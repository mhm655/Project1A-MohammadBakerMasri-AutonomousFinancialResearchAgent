"""Environment check: ``python -m config.doctor``.

Verifies everything ARA-1 needs and explains how to fix what is missing:

* Python version and installed packages
* ``.env`` and which keys are set (values are never printed)
* configuration validity
* the tool registry
* the vector database (Chroma + the local embedding model)
* each LLM provider in the chain: reachable, model exists, answers a prompt, and accepts ARA-1's
  real tool schemas and returns a tool call

Options:
    --offline          skip every network check
    --provider NAME    check only this LLM provider
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

REQUIRED_PACKAGES = [
    ("openai", "openai"), ("langgraph", "langgraph"), ("chromadb", "chromadb"), ("jsonschema", "jsonschema"),
    ("pydantic", "pydantic"), ("dotenv", "python-dotenv"), ("httpx", "httpx"), ("tenacity", "tenacity"),
    ("rich", "rich"), ("tiktoken", "tiktoken"), ("yfinance", "yfinance"), ("ddgs", "ddgs"),
    ("feedparser", "feedparser"), ("vaderSentiment", "vaderSentiment"), ("bs4", "beautifulsoup4"),
    ("rank_bm25", "rank-bm25"),
]


@dataclass
class Check:
    """One environment check: its name, a status (ok, warn, fail or skip) and a human-readable detail."""
    name: str
    status: str  # "ok", "warn", "fail", "skip"
    detail: str


def _run(name: str, fn: Callable[[], tuple[str, str]]) -> Check:
    try:
        status, detail = fn()
    except Exception as exc:  # the doctor must report problems, not crash on them
        status, detail = "fail", f"{type(exc).__name__}: {exc}"
    return Check(name, status, detail)


def check_python() -> tuple[str, str]:
    """Require Python 3.11 or newer."""
    v = sys.version_info
    return ("ok" if v >= (3, 11) else "fail"), f"Python {v.major}.{v.minor}.{v.micro}"


def check_packages() -> tuple[str, str]:
    """Verify every package in REQUIRED_PACKAGES can be imported; list the missing ones."""
    missing = []
    for module, dist in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(dist)
    if missing:
        return "fail", "missing: " + ", ".join(missing) + "  ->  python -m pip install -r requirements.txt"
    return "ok", f"{len(REQUIRED_PACKAGES)} packages importable"


def check_env_file() -> tuple[str, str]:
    """Warn if the project has no .env file."""
    from config.settings import PROJECT_ROOT

    if not (PROJECT_ROOT / ".env").exists():
        return "warn", "no .env file  ->  copy .env.example to .env and fill it in"
    return "ok", ".env found"


def check_settings() -> tuple[str, str]:
    """Load the settings and summarise mode, data mode, provider chain and API-key presence."""
    from config.settings import load_settings

    s = load_settings()
    keys = []
    for p in s.llm.chain():
        if p.needs_key:
            keys.append(f"{p.api_key_env}={'set' if p.api_key else 'missing'}")
    return "ok", f"mode={s.mode} data_mode={s.data_mode} chain={','.join(s.llm.provider_order)} " + " ".join(keys)


def check_llm_configured() -> tuple[str, str]:
    """Check that at least one cloud LLM key is set (always ok in mock mode)."""
    from config.settings import load_settings

    s = load_settings()
    configured = [p.name for p in s.llm.configured_chain()]
    if s.mode == "mock":
        return "ok", "mock mode: scripted LLM"
    keyed = [p for p in s.llm.configured_chain() if p.needs_key]
    if not keyed:
        return "warn", ("no cloud LLM key set; only Ollama is available. Get a free Gemini key at "
                        "https://aistudio.google.com/apikey and put it in .env as GEMINI_API_KEY")
    return "ok", "configured: " + ", ".join(configured)


def check_sec_email() -> tuple[str, str]:
    """Warn if SEC_CONTACT_EMAIL is unset (EDGAR requires it in the User-Agent)."""
    from config.settings import load_settings

    s = load_settings()
    if not s.sec.contact_email:
        return "warn", "SEC_CONTACT_EMAIL not set (required by EDGAR from Day 5)  ->  add it to .env"
    return "ok", f"User-Agent: {s.sec.user_agent}"


def check_registry() -> tuple[str, str]:
    """Build the default tool registry and require at least 10 registered tools."""
    from config.settings import load_settings
    from tools import build_default_registry

    s = load_settings()
    registry = build_default_registry(s.data_mode)
    available = [t.name for t in registry.available()]
    status = "ok" if len(registry) >= 10 else "fail"
    return status, f"{len(registry)} tools registered, {len(available)} available in {s.data_mode} mode"


def check_vector_db(offline: bool) -> tuple[str, str]:
    """Round-trip one record through the Chroma vector store (hashing embeddings when offline)."""
    from memory.vector_store import ChromaVectorStore

    model = "hashing" if offline else "default"
    # Chroma keeps its files open on Windows, so the temporary folder may not delete cleanly.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        try:
            store = ChromaVectorStore(None if offline else tmp, embedding_model=model)
            store.store("NVIDIA data-center revenue grew strongly.",
                        {"ticker": "NVDA", "date": "2025-02-26", "source_type": "10-K"})
            hit = store.search("data center growth")["results"][0]
        except Exception as exc:
            if model == "default" and ("Timeout" in type(exc).__name__ or "Connect" in type(exc).__name__):
                return "fail", (f"the embedding model (~80 MB, first use only) could not be downloaded: "
                                f"{type(exc).__name__}. Check the connection and re-run; memory works offline "
                                f"meanwhile with embedding_model = 'hashing' in config/default.toml")
            raise
    note = " (offline: hashing embeddings; the ONNX model downloads on first online run)" if offline else ""
    return "ok", f"Chroma round trip OK with '{model}' embeddings, score {hit['score']}{note}"


def check_provider(name: str) -> list[Check]:
    """Check one LLM provider: model listing, a plain completion and a full tool-call round trip. Returns one Check per step."""
    from agent.llm import LLMClient, assistant_message, build_provider, tool_result_message
    from config.settings import load_settings
    from tools import build_default_registry

    s = load_settings()
    settings = s.llm.providers[name]
    label = f"llm:{name} ({settings.model})"
    if not settings.is_configured:
        return [Check(label, "skip", f"{settings.api_key_env} not set")]

    provider = build_provider(settings, timeout_s=s.llm.timeout_s)
    client = LLMClient([provider], max_retries=3)  # free tiers return transient 503s under load
    checks = []

    # 1. Is the model listed? (Not every server implements /models; that is only a warning. Being
    #    listed does not prove the model is usable: Gemini lists retired models. Step 2 confirms.)
    if hasattr(provider, "list_models"):
        try:
            ids = provider.list_models()
            present = settings.model in ids
            sample = ", ".join(sorted(i for i in ids if any(w in i for w in ("flash", "llama", "qwen", "gpt")))[:6])
            checks.append(Check(f"{label} model", "ok" if present else "fail",
                                "listed by the server" if present else f"not found; available include: {sample}"))
            if not present:
                return checks
        except Exception as exc:
            if "Connection" in type(exc).__name__:
                hint = " (is Ollama installed and running?)" if name == "ollama" else ""
                return checks + [Check(label, "fail", f"cannot connect to {settings.base_url}{hint}")]
            checks.append(Check(f"{label} model", "warn", f"could not list models: {str(exc)[:200]}"))

    # 2. Plain completion.
    try:
        reply = client.complete([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=256)
        if not (reply.content or "").strip():
            checks.append(Check(f"{label} chat", "warn",
                                f"empty reply (finish_reason={reply.finish_reason}, {reply.usage.reasoning_tokens} "
                                "hidden reasoning tokens): lower reasoning_effort or raise max_output_tokens"))
        else:
            thinking = f", {reply.usage.reasoning_tokens} reasoning tokens" if reply.usage.reasoning_tokens else ""
            checks.append(Check(f"{label} chat", "ok",
                                f"replied {reply.content.strip()!r} in {reply.latency_ms:.0f} ms{thinking}"))
    except Exception as exc:
        return checks + [Check(f"{label} chat", "fail", str(exc)[:300])]

    # 3. The real tool schemas, and a full tool-call round trip.
    registry = build_default_registry("mock")
    tools = registry.openai_tools()
    messages = [
        {"role": "system", "content": "You are a financial research agent. Use tools to get data."},
        {"role": "user", "content": "Get the company profile for ticker MSFT."},
    ]
    try:
        first = client.complete(messages, tools=tools, tool_choice="auto", max_tokens=1024)
    except Exception as exc:
        return checks + [Check(f"{label} tools", "fail", f"provider rejected the tool schemas: {str(exc)[:300]}")]
    if not first.tool_calls:
        return checks + [Check(f"{label} tools", "warn", f"accepted {len(tools)} tool schemas but made no tool call: {first.content!r:.120}")]
    # Like the real agent: every tool call gets a result, and a tool error goes back to the model as
    # the result so it can correct itself. What is under test here is the protocol (tool-call ids,
    # Gemini thought signatures), not whether the model's first guess was valid.
    from tools import ToolError

    messages.append(assistant_message(first))
    outcomes = []
    for call in first.tool_calls:
        try:
            content = registry.dispatch(call.name, call.arguments).data
            outcomes.append(f"{call.name}: ok")
        except ToolError as exc:
            content = {"error": str(exc)}
            outcomes.append(f"{call.name}: tool error returned to the model")
        messages.append(tool_result_message(call.id, content))
    try:
        final = client.complete(messages, tools=tools, max_tokens=1024)
    except Exception as exc:
        return checks + [Check(f"{label} tools", "fail", f"the follow-up turn was rejected: {str(exc)[:250]}")]
    follow_up = (f"then called {', '.join(tc.name for tc in final.tool_calls)}" if final.tool_calls
                 else f"answered {(final.content or '').strip()[:60]!r}")
    checks.append(Check(f"{label} tools", "ok",
                        f"accepted {len(tools)} tool schemas; round trip OK ({'; '.join(outcomes)}); {follow_up}"))
    return checks


def run(offline: bool = False, only_provider: str | None = None) -> list[Check]:
    """Run every environment check and return the results; offline mode skips the live LLM provider checks."""
    checks = [
        _run("python", check_python),
        _run("packages", check_packages),
        _run(".env", check_env_file),
        _run("settings", check_settings),
        _run("llm keys", check_llm_configured),
        _run("sec email", check_sec_email),
        _run("tool registry", check_registry),
        _run("vector db", lambda: check_vector_db(offline)),
    ]
    if offline:
        checks.append(Check("llm providers", "skip", "offline mode"))
        return checks
    try:
        from config.settings import load_settings

        s = load_settings()
        names = [only_provider] if only_provider else [n for n in s.llm.provider_order if n != "mock"]
        by_provider: dict[str, list[Check]] = {}
        for name in names:
            try:
                by_provider[name] = check_provider(name)
            except Exception as exc:
                by_provider[name] = [Check(f"llm:{name}", "fail", f"{type(exc).__name__}: {exc}")]
        # One working provider is enough to run; a broken fallback is then a warning, not a failure.
        # A provider's failures are downgraded only if ANOTHER provider works: when the only working
        # provider rejects the tool schemas, nothing covers for it.
        working = {n for n, cs in by_provider.items() if any(c.name.endswith(" chat") and c.status == "ok" for c in cs)}
        for name, provider_checks in by_provider.items():
            if working - {name}:
                for c in provider_checks:
                    if c.status == "fail":
                        c.status, c.detail = "warn", f"{c.detail} (a fallback provider; another provider works)"
            checks.extend(provider_checks)
    except Exception as exc:
        checks.append(Check("llm providers", "fail", str(exc)))
    return checks


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns 1 if any check failed, else 0."""
    parser = argparse.ArgumentParser(description="Check the ARA-1 environment.")
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--provider", help="check only this LLM provider")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    checks = run(offline=args.offline, only_provider=args.provider)
    if args.json:
        print(json.dumps([vars(c) for c in checks], indent=2))
    else:
        _print_table(checks)
    return 1 if any(c.status == "fail" for c in checks) else 0


def _print_table(checks: list[Check]) -> None:
    icons = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    try:
        from rich.console import Console
        from rich.table import Table

        colours = {"ok": "green", "warn": "yellow", "fail": "red", "skip": "dim"}
        table = Table(title="ARA-1 environment check", show_lines=False)
        table.add_column("Status")
        table.add_column("Check")
        table.add_column("Detail", overflow="fold")
        for c in checks:
            table.add_row(f"[{colours[c.status]}]{icons[c.status]}[/]", c.name, c.detail)
        Console().print(table)
    except ImportError:
        for c in checks:
            print(f"[{icons[c.status]}] {c.name}: {c.detail}")


if __name__ == "__main__":
    sys.exit(main())
