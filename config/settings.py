"""Typed, validated configuration for ARA-1.

Settings come from three layers, each overriding the one before:

1. ``config/default.toml``
2. ``config/<mode>.toml`` (optional, e.g. ``config/local.toml``)
3. environment variables, including those in a git-ignored ``.env`` file

API keys are never stored in configuration files. Each provider names the environment variable that
holds its key (``api_key_env``), and the key is excluded from ``repr`` so it cannot leak into logs.
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent

MODES = ("mock", "local", "free")
DATA_MODES = ("mock", "live")
PROVIDER_KINDS = ("openai_compatible", "anthropic", "mock")
SCHEMA_DIALECTS = ("openai", "gemini")


class ConfigError(ValueError):
    """The configuration is invalid or incomplete."""


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    kind: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    schema_dialect: str = "openai"
    #: Default thinking level for reasoning models ("none", "minimal", "low", "medium", "high").
    #: Empty = never send the parameter, for providers that do not accept it.
    reasoning_effort: str = ""
    #: Cap on output tokens per request (0 = no cap); some free tiers reject larger requests outright.
    max_output_tokens: int = 0
    api_key: str | None = field(default=None, repr=False)

    @property
    def needs_key(self) -> bool:
        return bool(self.api_key_env)

    @property
    def is_configured(self) -> bool:
        """True if the provider can be called: it needs no key, or its key is set."""
        return not self.needs_key or bool(self.api_key)


@dataclass(frozen=True)
class LLMSettings:
    provider_order: tuple[str, ...]
    providers: dict[str, ProviderSettings]
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_s: float = 60.0
    max_retries: int = 2

    def chain(self) -> list[ProviderSettings]:
        """Providers in fallback order, including unconfigured ones (callers decide to skip them)."""
        return [self.providers[name] for name in self.provider_order]

    def configured_chain(self) -> list[ProviderSettings]:
        return [p for p in self.chain() if p.is_configured]


@dataclass(frozen=True)
class PathSettings:
    data_dir: Path
    chroma_dir: Path
    episodic_db: Path
    traces_dir: Path


@dataclass(frozen=True)
class AgentSettings:
    max_tool_calls: int = 20
    max_calls_per_step: int = 3
    max_replans: int = 3
    max_reflections: int = 2


@dataclass(frozen=True)
class RetrySettings:
    base_delay_s: float = 1.0
    max_retries: int = 5
    max_jitter_s: float = 0.5


@dataclass(frozen=True)
class MemorySettings:
    collection_prefix: str = "ara_findings"
    embedding_model: str = "default"


@dataclass(frozen=True)
class SecSettings:
    app_name: str
    contact_email: str | None
    max_requests_per_second: int = 8

    @property
    def user_agent(self) -> str:
        """EDGAR's required User-Agent: application name plus a contact email."""
        if not self.contact_email:
            raise ConfigError("SEC_CONTACT_EMAIL is not set; EDGAR requires a contact email in the User-Agent.")
        return f"{self.app_name} {self.contact_email}"


@dataclass(frozen=True)
class Settings:
    mode: str
    data_mode: str
    llm: LLMSettings
    paths: PathSettings
    agent: AgentSettings
    retry: RetrySettings
    memory: MemorySettings
    sec: SecSettings
    log_level: str
    project_root: Path = PROJECT_ROOT


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    config_dir: Path = CONFIG_DIR,
    project_root: Path = PROJECT_ROOT,
    load_dotenv_file: bool = True,
) -> Settings:
    """Load and validate settings.

    Args:
        env: environment to read overrides and keys from. Defaults to ``os.environ`` (after loading
            ``.env``). Tests pass an explicit mapping so they never depend on the real environment.
    """
    if env is None:
        if load_dotenv_file:
            _load_dotenv(project_root / ".env")
        env = os.environ

    raw = _read_toml(config_dir / "default.toml")
    mode = env.get("ARA_MODE") or raw.get("mode", "free")
    if mode not in MODES:
        raise ConfigError(f"ARA_MODE must be one of {MODES}, got {mode!r}")
    mode_file = config_dir / f"{mode}.toml"
    if mode_file.exists():
        raw = _deep_merge(raw, _read_toml(mode_file))

    data_mode = env.get("ARA_DATA_MODE") or raw.get("data_mode") or ("mock" if mode == "mock" else "live")
    if data_mode not in DATA_MODES:
        raise ConfigError(f"ARA_DATA_MODE must be one of {DATA_MODES}, got {data_mode!r}")

    return Settings(
        mode=mode,
        data_mode=data_mode,
        llm=_llm_settings(raw.get("llm", {}), mode, env),
        paths=_path_settings(raw.get("paths", {}), project_root, env),
        agent=AgentSettings(**raw.get("agent", {})),
        retry=RetrySettings(**raw.get("retry", {})),
        memory=MemorySettings(**raw.get("memory", {})),
        sec=SecSettings(
            app_name=raw.get("sec", {}).get("app_name", "ARA-1 Research Agent"),
            contact_email=env.get("SEC_CONTACT_EMAIL") or None,
            max_requests_per_second=raw.get("sec", {}).get("max_requests_per_second", 8),
        ),
        log_level=(env.get("ARA_LOG_LEVEL") or raw.get("logging", {}).get("level", "INFO")).upper(),
        project_root=project_root,
    )


def _llm_settings(raw: dict[str, Any], mode: str, env: Mapping[str, str]) -> LLMSettings:
    providers: dict[str, ProviderSettings] = {}
    for name, p in raw.get("providers", {}).items():
        kind = p.get("kind", "openai_compatible")
        if kind not in PROVIDER_KINDS:
            raise ConfigError(f"Provider {name!r}: unknown kind {kind!r}")
        dialect = p.get("schema_dialect", "openai")
        if dialect not in SCHEMA_DIALECTS:
            raise ConfigError(f"Provider {name!r}: unknown schema_dialect {dialect!r}")
        key_env = p.get("api_key_env", "")
        prefix = f"ARA_{name.upper()}_"
        providers[name] = ProviderSettings(
            name=name,
            kind=kind,
            model=env.get(prefix + "MODEL") or p["model"],
            base_url=env.get(prefix + "BASE_URL") or p.get("base_url", ""),
            api_key_env=key_env,
            schema_dialect=dialect,
            reasoning_effort=env.get(prefix + "REASONING_EFFORT") or p.get("reasoning_effort", ""),
            max_output_tokens=int(p.get("max_output_tokens", 0)),
            api_key=(env.get(key_env) or None) if key_env else None,
        )
    providers["mock"] = ProviderSettings(name="mock", kind="mock", model="mock-llm")

    if mode == "mock":
        order: tuple[str, ...] = ("mock",)
    elif mode == "local":
        order = ("ollama",)
    elif env.get("ARA_PROVIDER_ORDER"):
        order = tuple(s.strip() for s in env["ARA_PROVIDER_ORDER"].split(",") if s.strip())
    else:
        order = tuple(raw.get("provider_order", ["gemini", "groq", "ollama"]))

    unknown = [name for name in order if name not in providers]
    if unknown:
        raise ConfigError(f"provider_order names unknown providers: {unknown}")
    if not order:
        raise ConfigError("provider_order is empty")

    return LLMSettings(
        provider_order=order,
        providers=providers,
        temperature=float(raw.get("temperature", 0.0)),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        timeout_s=float(raw.get("timeout_s", 60.0)),
        max_retries=int(raw.get("max_retries", 2)),
    )


def _path_settings(raw: dict[str, Any], root: Path, env: Mapping[str, str]) -> PathSettings:
    def resolve(key: str, default: str) -> Path:
        path = Path(env.get(f"ARA_{key.upper()}") or raw.get(key, default))
        return path if path.is_absolute() else root / path

    return PathSettings(
        data_dir=resolve("data_dir", "data"),
        chroma_dir=resolve("chroma_dir", "data/chroma"),
        episodic_db=resolve("episodic_db", "data/episodic.sqlite"),
        traces_dir=resolve("traces_dir", "traces"),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name}: invalid TOML: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed: fall back to the process environment only
        return
    load_dotenv(path, override=False)
