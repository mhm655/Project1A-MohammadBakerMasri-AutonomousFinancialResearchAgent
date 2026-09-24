# ARA-1: Autonomous Financial Research Agent

ARA-1 is an autonomous agent that takes a research question about a public company and writes a
cited investment research report. It plans the research, gathers data from SEC filings, market
data, earnings communications and the news, and resolves conflicting figures along the way. It
needs no step-by-step guidance.

Zetheta Project 1A · Mohammad Baker Masri

> **Status: Day 3.** Done so far:
> - the architecture specification;
> - the tool registry (12 tools);
> - configuration, the LLM client with its provider fallback chain, the Chroma vector store, and
>   tracing.
>
> The agent loop itself arrives on Day 4.

**Runs at zero cost.** Every dependency and data source is free, and no credit card is required
anywhere.

## What is built and what is planned

| Component | Status |
|---|---|
| **Tool registry**: 12 tools, schema validation, repair of common LLM argument mistakes, caching, evidence ids, call log | Built (Day 2) |
| **Local tools**: `calculation_engine`, `fact_checker`, `report_generator` | Built (Day 2) |
| **Network tools** (SEC, market data, search, news, transcripts, peers) | Mock data now; live on Days 5–7 |
| **LLM client**: Gemini → Groq → Ollama through the `openai` SDK, with retries, reasoning-token accounting and tracing. Verified against the live Gemini API. | Built (Day 3) |
| **Long-term memory**: Chroma with local embeddings | Built (Day 3) |
| **Hybrid agent loop**: a Plan-and-Execute graph (LangGraph) running a bounded ReAct executor per step | Day 4 |
| **Short-term context management and episodic memory** | Day 6 |
| **Synthesis**: a claim ledger (period, unit, evidence, confidence) and tiered conflict resolution | Day 8 |
| **Tool fallback chains, circuit breakers, failure injection** | Day 9 (fallbacks are already declared per tool) |
| **Evaluation**: 22 metrics, LLM judge, dashboard | Day 11 |

The full design is in the [architecture specification](docs/architecture_specification.md).

## Setup

### 1. Prerequisites

- **Python 3.11 or newer** (developed on 3.14).
- **Git.**
- About 1 GB of free disk space for the dependencies and the embedding model.

### 2. Install

```bash
git clone <this repository>
cd Project1A-MohammadBakerMasri-AutonomousFinancialResearchAgent
python -m venv .venv
```

Activate the environment:

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Git Bash) | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in the following. `.env` is git-ignored, so your keys never get committed.

| Variable | Required? | How to get it (all free, no card) |
|---|---|---|
| `GEMINI_API_KEY` | Recommended: the primary LLM | Sign in at <https://aistudio.google.com/apikey> with a Google account, then click **Create API key** |
| `GROQ_API_KEY` | Optional: backup LLM | Create an account at <https://console.groq.com/keys> and create a key |
| `SEC_CONTACT_EMAIL` | Required from Day 5 | Any email you read. The SEC requires a contact address in the User-Agent of every EDGAR request; no account is needed |

**Optional: an offline LLM with Ollama.** Install it from <https://ollama.com>, then run
`ollama pull qwen2.5:3b` (about 2 GB). ARA-1 falls back to it automatically when the cloud providers
are unavailable. Run with `ARA_MODE=local` to use only Ollama.

### 4. Check the setup

```bash
python -m config.doctor
```

The doctor checks the following, and says how to fix anything that is missing:

- the Python version and installed packages;
- your `.env` file and which keys are set (it never prints key values);
- the configuration;
- the tool registry;
- the vector database. The embedding model, about 80 MB, is downloaded once on the first run.

For each LLM provider it also checks that:

- the model exists;
- the model answers a prompt;
- it accepts ARA-1's real tool schemas and completes a tool-call round trip.

Use `--offline` to skip the network checks.

### 5. Run the tests

```bash
python -m pytest
```

The tests need no network access and no API keys.

## Run modes

Set these in `.env` or as environment variables. Defaults live in [config/default.toml](config/default.toml).

| `ARA_MODE` | LLM | Data | Use |
|---|---|---|---|
| `mock` | Scripted mock | Mock fixtures | Tests, and demos with no network |
| `local` | Ollama only | Live | Offline or unlimited development |
| `free` *(default)* | Gemini → Groq → Ollama | Live | Challenge runs |

`ARA_DATA_MODE=mock|live` overrides the data source independently of the LLM. The live data tools
arrive on Day 5; until then, keep `ARA_DATA_MODE=mock`.

## Project layout

```
agent/        reasoning loop (Day 4), LLM client (llm.py), tracing (tracing.py)
tools/        tool registry, 12 JSON tool schemas, tool implementations
memory/       vector store (Chroma); context manager and episodic memory (Day 6)
synthesis/    claim ledger, conflict resolution, narrative (Day 8)
evaluation/   metrics, benchmarks, dashboard (Day 11)
config/       default.toml, settings loader, environment doctor
docs/         architecture specification and diagram
results/      challenge outputs
tests/        pytest suite
```

## Documents

| Document | Purpose |
|---|---|
| [docs/architecture_specification.md](docs/architecture_specification.md) | The full design: cognitive loop, tools, memory, synthesis, error handling, evaluation |
| [docs/architecture_diagram.svg](docs/architecture_diagram.svg) | Architecture diagram |
| [ERROR_LOG.md](ERROR_LOG.md) | The 7 deliberate errors found in the project document |
| [tools/schemas/](tools/schemas/) | The 12 tool definitions (OpenAI function format plus `x-ara` metadata) |

![Architecture](docs/architecture_diagram.svg)

## Use of AI tools

Following the project's AI Assistance Policy (E5.3), the AI tools used in this project are listed
below.

| Tool | Used for |
|---|---|
| Claude Code (Anthropic) | Drafting the architecture specification, diagram and error log; implementing and testing code. All output is reviewed and validated against the project requirements. |
