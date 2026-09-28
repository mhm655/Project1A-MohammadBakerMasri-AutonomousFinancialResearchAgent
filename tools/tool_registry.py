"""The tool registry: the agent's catalogue of tools and the single entry point for calling them.

Responsibilities (spec A2.2):

1. **Metadata for the LLM** — :meth:`ToolRegistry.openai_tools` returns the available tools in OpenAI
   function-calling format, and :meth:`ToolRegistry.describe` renders them for the system prompt.
2. **Execution** — :meth:`ToolRegistry.dispatch` validates arguments against the tool's JSON Schema,
   serves repeated calls from cache, runs the handler, validates the output, and wraps the result in
   a :class:`~tools.base.ToolResult` carrying provenance and an evidence id.
3. **Bookkeeping** — every call, successful or not, is appended to :attr:`ToolRegistry.call_log`.

Tools are defined declaratively by JSON files in ``tools/schemas/``. Each file holds the OpenAI-format
definition plus an ``x-ara`` block with ARA-1 metadata, including the dotted path of the handler
function for each run mode. Adding a tool therefore needs no code change to the registry or the agent
(spec B4.4): drop a schema file into the directory and it is picked up on the next load.

Retries, circuit breakers and fallback chains are layered on top of :meth:`dispatch` on Day 9
(``agent/error_handler.py``); this module deliberately stays synchronous and single-attempt.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import logging
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from tools.base import (
    CallRecord,
    ToolError,
    ToolExecutionError,
    ToolHandler,
    ToolLockedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolResult,
    ToolSpec,
    ToolUnavailableError,
    ToolValidationError,
    UnlockRule,
)

SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Run modes. "mock" uses stub handlers with canned data; "live" uses real APIs (Day 5 onwards).
MODES = ("mock", "live")

# OpenAI requires function names to match this pattern.
_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def resolve_handler(path: str) -> ToolHandler:
    """Import ``"package.module:function"`` and return the function."""
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Handler path must look like 'module:function', got {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Handler {path!r}: module has no attribute {attr!r}") from exc


def spec_from_schema(schema: dict[str, Any], mode: str) -> ToolSpec:
    """Build a :class:`ToolSpec` from a schema document (the contents of one ``schemas/*.json``).

    The handler for ``mode`` is used if present, otherwise the mode-independent ``"any"`` handler.
    If neither exists the spec is still returned, with ``handler=None``, and the tool is reported as
    unavailable in that mode.
    """
    _check_definition(schema)
    meta = schema.get("x-ara", {})
    handlers: dict[str, str | None] = meta.get("handlers", {})

    handler_mode = mode if handlers.get(mode) else ("any" if handlers.get("any") else None)
    handler = resolve_handler(handlers[handler_mode]) if handler_mode else None

    unlock = meta.get("unlock")
    return ToolSpec(
        name=schema["name"],
        description=schema["description"],
        parameters=schema["parameters"],
        handler=handler,
        handler_mode=handler_mode,
        output_schema=meta.get("output_schema"),
        source_tier=meta.get("source_tier"),
        cache_ttl_s=int(meta.get("cache_ttl_s", 0)),
        fallbacks=list(meta.get("fallbacks", [])),
        domains=list(meta.get("domains", [])),
        unlock=UnlockRule(**unlock) if unlock else None,
    )


def _check_definition(schema: dict[str, Any]) -> None:
    """Reject tool definitions that the LLM providers would refuse or misread."""
    name = schema.get("name", "")
    if not name or len(name) > 64 or not set(name) <= _NAME_CHARS:
        raise ValueError(f"Invalid tool name {name!r}: use 1-64 chars of [a-zA-Z0-9_-]")
    if not schema.get("description"):
        raise ValueError(f"Tool {name!r} has no description")
    params = schema.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object":
        raise ValueError(f"Tool {name!r}: 'parameters' must be a JSON Schema of type 'object'")
    missing = set(params.get("required", [])) - set(params.get("properties", {}))
    if missing:
        raise ValueError(f"Tool {name!r}: required fields not declared in properties: {missing}")
    for label, sub in (("parameters", params), ("output_schema", schema.get("x-ara", {}).get("output_schema"))):
        if sub is None:
            continue
        try:
            Draft202012Validator.check_schema(sub)
        except SchemaError as exc:
            raise ValueError(f"Tool {name!r}: {label} is not a valid JSON Schema: {exc.message}") from exc


class ToolRegistry:
    """Catalogue and dispatcher for the agent's tools.

    Args:
        mode: ``"mock"`` or ``"live"``; selects which handler each tool uses.
        completed_challenges: how many research challenges have been completed, for unlock rules.
        enforce_unlocks: when False (the default) every tool is available regardless of unlock
            rules. The spec's unlock schedule contradicts the tools its challenges expect
            (ERROR_LOG #7), so enforcement is opt-in.
        clock: monotonic time source in seconds; injectable so cache expiry can be tested.
    """

    def __init__(
        self,
        mode: str = "mock",
        *,
        completed_challenges: int = 0,
        enforce_unlocks: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.completed_challenges = completed_challenges
        self.enforce_unlocks = enforce_unlocks
        self._clock = clock
        self._specs: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._output_validators: dict[str, Draft202012Validator] = {}
        self._cache: dict[tuple[str, str], tuple[float, ToolResult]] = {}
        self._evidence_ids = itertools.count(1)
        self._listeners: list[Callable[[CallRecord], None]] = []
        self.call_log: list[CallRecord] = []

    # ------------------------------------------------------------------ registration

    def register(self, spec: ToolSpec) -> None:
        """Add a tool, or replace an existing tool with the same name."""
        self._specs[spec.name] = spec
        self._validators[spec.name] = Draft202012Validator(spec.parameters, format_checker=FormatChecker())
        if spec.output_schema is not None:
            self._output_validators[spec.name] = Draft202012Validator(spec.output_schema)
        else:
            self._output_validators.pop(spec.name, None)
        self._drop_cache(spec.name)

    def register_schema(self, schema: dict[str, Any], handler: ToolHandler | None = None) -> ToolSpec:
        """Register a tool from a schema document; ``handler`` overrides the one named in the schema."""
        spec = spec_from_schema(schema, self.mode)
        if handler is not None:
            spec.handler, spec.handler_mode = handler, self.mode
        self.register(spec)
        return spec

    def load_directory(self, directory: str | Path = SCHEMA_DIR) -> list[str]:
        """Register every ``*.json`` schema in ``directory``; returns the tool names, sorted."""
        names = []
        for path in sorted(Path(directory).glob("*.json")):
            with path.open(encoding="utf-8") as fh:
                schema = json.load(fh)
            try:
                names.append(self.register_schema(schema).name)
            except ValueError as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
        return names

    def bind_handler(self, name: str, handler: ToolHandler) -> None:
        """Replace the handler of an existing tool (e.g. to inject a test double or a failure injector)."""
        spec = self.get(name)
        spec.handler, spec.handler_mode = handler, self.mode
        self._drop_cache(name)

    def add_listener(self, listener: Callable[[CallRecord], None]) -> None:
        """Call ``listener(record)`` after every dispatch, e.g. to write the call to a trace."""
        self._listeners.append(listener)

    # ------------------------------------------------------------------ lookup

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolNotFoundError(f"Unknown tool {name!r}. Known tools: {sorted(self._specs)}", tool=name) from None

    @property
    def names(self) -> list[str]:
        """All registered tool names, including locked or unimplemented ones."""
        return sorted(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def is_unlocked(self, name: str) -> bool:
        spec = self.get(name)
        return not (self.enforce_unlocks and spec.unlock and not spec.unlock.is_met(self.completed_challenges))

    def is_available(self, name: str) -> bool:
        """True if the tool is unlocked and has a handler for the current mode."""
        return self.is_unlocked(name) and self.get(name).handler is not None

    def available(self) -> list[ToolSpec]:
        """Tools the agent may use right now, sorted by name."""
        return [self._specs[n] for n in self.names if self.is_available(n)]

    def set_progress(self, completed_challenges: int) -> list[str]:
        """Record challenge progress; returns the names of tools that just became unlocked."""
        before = {s.name for s in self.available()}
        self.completed_challenges = completed_challenges
        return sorted({s.name for s in self.available()} - before)

    # ------------------------------------------------------------------ LLM-facing views

    def openai_tools(self) -> list[dict[str, Any]]:
        """Available tools in OpenAI function-calling format, ready to pass as ``tools=``."""
        return [copy.deepcopy(spec.to_openai()) for spec in self.available()]

    def describe(self, exclude: set[str] | frozenset[str] = frozenset()) -> str:
        """Plain-text tool signatures for the system prompt, with exact argument names, allowed values
        and defaults. Without these, models guess argument names and the calls fail validation."""
        lines = []
        for spec in self.available():
            if spec.name in exclude:
                continue
            props = spec.parameters.get("properties", {})
            required = set(spec.parameters.get("required", []))
            args = ", ".join(_signature(name, schema, name in required) for name, schema in props.items())
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ execution

    def validate_args(self, name: str, args: dict[str, Any] | str | None) -> dict[str, Any]:
        """Parse and validate arguments; returns a copy with schema defaults filled in.

        ``args`` may be a JSON string, because that is how LLM APIs deliver tool-call arguments.
        """
        spec = self.get(name)
        if args is None:
            args = {}
        elif isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolValidationError(f"{name}: arguments are not valid JSON ({exc.msg})", tool=name) from exc
        if not isinstance(args, dict):
            raise ToolValidationError(f"{name}: arguments must be a JSON object", tool=name)

        filled = coerce_to_schema(copy.deepcopy(args), spec.parameters)
        for prop, prop_schema in spec.parameters.get("properties", {}).items():
            if prop not in filled and "default" in prop_schema:
                filled[prop] = copy.deepcopy(prop_schema["default"])
        if isinstance(filled.get("ticker"), str):
            filled["ticker"] = filled["ticker"].strip().upper()  # one cache entry per company

        errors = sorted(self._validators[name].iter_errors(filled), key=lambda e: list(e.path))
        if errors:
            details = "; ".join(_format_error(e) for e in errors)
            raise ToolValidationError(f"{name}: invalid arguments: {details}", tool=name)
        return filled

    def dispatch(self, name: str, args: dict[str, Any] | str | None = None) -> ToolResult:
        """Validate, run and wrap a single tool call. Raises a :class:`ToolError` subclass on failure."""
        started = time.perf_counter()
        logged_args: dict[str, Any] = args if isinstance(args, dict) else {"_raw": args}
        try:
            spec = self.get(name)
            if not self.is_unlocked(name):
                raise ToolLockedError(
                    f"{name} unlocks after challenge {spec.unlock.after_challenge}"  # type: ignore[union-attr]
                    f" (completed: {self.completed_challenges})",
                    tool=name,
                )
            if spec.handler is None:
                raise ToolUnavailableError(f"{name} has no handler for mode {self.mode!r}", tool=name)

            clean = self.validate_args(name, args)
            logged_args = clean

            cached = self._cache_get(name, clean)
            if cached is not None:
                self._log(name, clean, started, result=cached)
                return cached

            try:
                data = spec.handler(**clean)
            except ToolError as exc:
                exc.tool = exc.tool or name
                raise
            except Exception as exc:  # a handler bug or an unexpected library error
                raise ToolExecutionError(f"{name} failed: {type(exc).__name__}: {exc}", tool=name) from exc

            self._check_output(name, data)
            result = ToolResult(
                evidence_id=f"E{next(self._evidence_ids)}",
                tool=name,
                args=clean,
                data=data,
                source_tier=spec.source_tier,
                retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
                latency_ms=_elapsed_ms(started),
                is_mock=spec.handler_mode == "mock",
            )
            self._cache_put(name, clean, result)
            self._log(name, clean, started, result=result)
            return result
        except ToolError as exc:
            self._log(name, logged_args, started, error=exc)
            raise

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ internals

    def _check_output(self, name: str, data: Any) -> None:
        if not isinstance(data, dict):
            raise ToolOutputError(f"{name} returned {type(data).__name__}, expected an object", tool=name)
        validator = self._output_validators.get(name)
        if validator is None:
            return
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        if errors:
            raise ToolOutputError(f"{name}: malformed output: {_format_error(errors[0])}", tool=name)

    def _cache_key(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        return name, json.dumps(args, sort_keys=True, default=str)

    def _cache_get(self, name: str, args: dict[str, Any]) -> ToolResult | None:
        if self._specs[name].cache_ttl_s <= 0:
            return None
        entry = self._cache.get(self._cache_key(name, args))
        if entry is None:
            return None
        expires_at, result = entry
        if self._clock() >= expires_at:
            del self._cache[self._cache_key(name, args)]
            return None
        hit = copy.deepcopy(result)  # callers may mutate their copy; the cached entry must not change
        hit.cache_hit = True
        return hit

    def _cache_put(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        ttl = self._specs[name].cache_ttl_s
        if ttl > 0:
            self._cache[self._cache_key(name, args)] = (self._clock() + ttl, copy.deepcopy(result))

    def _drop_cache(self, name: str) -> None:
        for key in [k for k in self._cache if k[0] == name]:
            del self._cache[key]

    def _log(
        self,
        name: str,
        args: dict[str, Any],
        started: float,
        *,
        result: ToolResult | None = None,
        error: ToolError | None = None,
    ) -> None:
        record = CallRecord(
            tool=name,
            args=args,
            ok=error is None,
            latency_ms=_elapsed_ms(started),
            evidence_id=result.evidence_id if result else None,
            cache_hit=bool(result and result.cache_hit),
            error_kind=error.kind if error else None,
            error_message=str(error) if error else None,
        )
        self.call_log.append(record)
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:  # a broken listener must never break a tool call
                logging.getLogger("ara").exception("Tool-call listener failed")


def _signature(name: str, schema: dict[str, Any], required: bool) -> str:
    """``statement_type: income_statement|balance_sheet``, ``years?: integer = 3``, ``metrics?: [revenue|roe]``."""
    kind = schema.get("type", "any")
    if "enum" in schema:
        kind = "|".join(str(v) for v in schema["enum"])
    elif kind == "array" and isinstance(schema.get("items"), dict):
        items = schema["items"]
        kind = "[" + ("|".join(str(v) for v in items["enum"]) if "enum" in items else items.get("type", "any")) + "]"
    elif kind == "object" and schema.get("properties"):
        kind = "{" + ", ".join(schema["properties"]) + "}"
    default = f" = {json.dumps(schema['default'])}" if "default" in schema else ""
    return f"{name}{'' if required else '?'}: {kind}{default}"


_INTEGER = re.compile(r"[+-]?\d+")
_BOOLEANS = {"true": True, "false": False}


def coerce_to_schema(value: Any, schema: dict[str, Any]) -> Any:
    """Repair the harmless mistakes LLMs make in tool arguments, guided by the schema.

    * ``null`` for an optional property means "not given": the property is dropped.
    * Scalars sent as strings (``"5"``, ``"true"``) become the declared type when unambiguous.
    * Objects and arrays sent as JSON-encoded strings are decoded. Some providers only accept
      free-form objects as strings (see ``agent/llm.py``), and models stringify nested JSON anyway.

    Anything that cannot be repaired is left alone for the validator to reject with a clear message.
    """
    kind = schema.get("type")
    if isinstance(value, str):
        text = value.strip()
        if kind in ("object", "array"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return value
        elif kind == "integer" and _INTEGER.fullmatch(text):
            return int(text)
        elif kind == "number":
            try:
                number = float(text)
            except ValueError:
                return value
            return number if math.isfinite(number) else value
        elif kind == "boolean" and text.lower() in _BOOLEANS:
            return _BOOLEANS[text.lower()]
        else:
            return value

    if isinstance(value, dict) and kind == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        return {
            key: coerce_to_schema(item, properties[key]) if key in properties else item
            for key, item in value.items()
            if not (item is None and key in properties and key not in required)
        }
    if isinstance(value, list) and kind == "array" and isinstance(schema.get("items"), dict):
        return [coerce_to_schema(item, schema["items"]) for item in value]
    return value


def build_default_registry(mode: str = "mock", **kwargs: Any) -> ToolRegistry:
    """A registry with every tool in ``tools/schemas/`` loaded."""
    registry = ToolRegistry(mode, **kwargs)
    registry.load_directory(SCHEMA_DIR)
    return registry


def _format_error(error: Any) -> str:
    location = ".".join(str(p) for p in error.absolute_path) or "(root)"
    return f"{location}: {error.message}"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = [
    "MODES",
    "SCHEMA_DIR",
    "ToolRegistry",
    "build_default_registry",
    "coerce_to_schema",
    "resolve_handler",
    "spec_from_schema",
]
