# ARA-1: Autonomous Financial Research Agent

ARA-1 is an autonomous agent that takes a research question about a public company and writes a
cited investment research report. It plans the research, gathers data from SEC filings, market
data, earnings communications and the news, and resolves conflicting figures along the way. It
needs no step-by-step guidance.

Zetheta Project 1A · Mohammad Baker Masri

> **Status: Day 2.** The tool registry, the 12 tool schemas and mock tool implementations are done,
> with 80 unit tests. Full setup instructions arrive on Day 3, and the quick-start guide on Day 14.

## Running the tests

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest
```

## Highlights

- **Hybrid agent loop.** A Plan-and-Execute outer graph (LangGraph) runs a bounded ReAct executor for
  each step and re-plans as it learns.
- **12 tools** behind one dispatch pipeline: schema validation, caching, rate limiting, retries with
  jitter, per-provider circuit breakers and fallback chains.
- **Three-layer memory:**
  - short-term context management;
  - long-term vector store (Chroma);
  - episodic strategy memory, plus a structured facts store.
- **A claim ledger** in which every number carries a period, a unit, evidence ids and a confidence
  score. Conflicts are resolved with a tiered source hierarchy and always disclosed.
- **Costs nothing to run.** The LLM chain is Gemini free tier → Groq free tier → local Ollama,
  called through the `openai` SDK. The data sources are SEC EDGAR/XBRL, yfinance, DuckDuckGo and
  Google News RSS, and embeddings are computed locally.

## Documents

| Document | Purpose |
|---|---|
| [docs/architecture_specification.md](docs/architecture_specification.md) | The full design: cognitive loop, tools, memory, synthesis, error handling, evaluation |
| [docs/architecture_diagram.svg](docs/architecture_diagram.svg) | Architecture diagram |
| [ERROR_LOG.md](ERROR_LOG.md) | The 7 deliberate errors found in the project document |
| [tools/tool_registry.py](tools/tool_registry.py) | Tool registry: schema validation, dispatch, caching, provenance, unlocks |
| [tools/schemas/](tools/schemas/) | The 12 tool definitions (OpenAI function format plus `x-ara` metadata) |

![Architecture](docs/architecture_diagram.svg)

## Use of AI tools

Following the project's AI Assistance Policy (E5.3), the AI tools used in this project are listed
below.

| Tool | Used for |
|---|---|
| Claude Code (Anthropic) | Drafting the architecture specification, diagram and error log; scaffolding code. All output is reviewed and validated against the project requirements. |
