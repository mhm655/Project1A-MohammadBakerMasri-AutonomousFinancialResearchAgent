# Optimization Log (Day 13)

Each entry gives the problem and its evidence, the change, the numbers before and after, and the side effects. Sources: `results/token_usage_analysis.md` (Day 12), `results/stress_test_report.md` (Day 12), `results/evaluation_report.md` and `results/eval/scores.json` (Day 11).

## How the numbers were measured

- **Mock token measurement (exact, repeatable).** `python -m evaluation.token_compare measure <dir>` runs C1-C8 with mock data and a scripted LLM, in a throw-away data directory. `python -m evaluation.token_compare table <before> <after>` prints calls and tokens per challenge and per call purpose. Baseline: `results/runs_day13/baseline_mock/` (code before any Day 13 change). After: `results/runs_day13/after_mock/`. Table: `results/runs_day13/token_before_after.md`.
- **Why a scripted LLM.** The stock mock model answers `{}` to everything, so no step ever needed an execute call and the largest cost did not show. The scripted model in `evaluation/token_compare.py` returns a plan (two direct and two ReAct steps per company), a tool call for each ReAct step, "done" for reviews and one short section per required section. It counts the tool schemas in the prompt size, as a real provider does. Mock numbers show structure (what is sent), not model behaviour.
- **Live re-run.** All eight challenges, `results/runs_day13/live/`, scored into `results/runs_day13/eval/`. Providers were mixed (see `evaluation_report_v2.md`), so live differences are indicative only.

## Entries

### T1. Execute calls send one tool schema

- **Problem.** Every ReAct execute call attached the schemas of all research tools (about 2,000 tokens). Execute calls were 56% of all tokens in the Day 11 runs (`token_usage_analysis.md`, recommendation 1).
- **Change.** `agent/core.py` `_run_step`: `tools` holds only the schema of `step.tool`. If that tool is not among the research tools, all schemas are sent. `agent/prompts.py` `execute_step_prompt` lists the other research tool names on one line. Dispatch still validates whatever tool the model calls.
- **Before -> after (mock, 22 execute calls).** Execute prompt tokens 68,343 -> 37,169 (-45.6%); mean 3,106 -> 1,689 per call. Total mock tokens 132,856 -> 92,055 (-30.7%, all Day 13 changes together).
- **Side effects.** A model that wants a different tool must say so in text; the step then ends as "not needed". No Day 11 run showed this need. Existing test fake keyed on the word "calculation" anywhere in the prompt; the new tool-name line contains it, so the fake now keys on `Preferred tool: calculation_engine` (`tests/test_core.py`).

### T2. Synthesis budget depends on the provider

- **Problem.** One cap (6,000 tokens, sized for Groq) applied to every provider. Gemini takes far more, so the writer saw little evidence. One Qwen prompt measured 107% of its cap.
- **Change.** `config/default.toml` `[agent.prompt_tokens_by_provider]` (gemini 24000, groq 6000, ollama 6000); `max_prompt_tokens` stays the default for other providers; `prompt_budget_share = 0.9`. `config/settings.py` `AgentSettings.prompt_budget(provider)`. `agent/llm.py` `LLMClient.next_provider_name()` returns the first provider `complete` would try now (same ready/cooling order; `_ready_and_cooling` is shared). `_node_synthesize` uses 90% of that provider's cap and traces `synthesis_prompt_budget`.
- **Before -> after.** Live: budget 21,600 on Gemini (C1, C2) and 5,400 on Groq (C3-C8). Largest Groq synthesis prompt 6,171 tokens (cap 6,000), so the provider's own count still runs about 14% above our estimate. Mock (default cap): synthesis prompts 27,053 -> 26,313 tokens in total, because the budget fell from 6,000 to 5,400 for C4 and C6.
- **Side effects.** On Groq the evidence budget is 10% smaller than before. Which provider answers can change between planning and writing; the budget follows the provider chosen at writing time.

### T3. No review call when there is nothing to review

- **Problem.** The review ("REPLAN") returned "done" in most Day 11 runs; each call cost about 1,400-1,500 prompt tokens in mock.
- **Change.** `_node_review`: with no pending gaps and no failed or skipped step, record `research_complete` ("deterministic check: ...") and return the same shape as the no-new-steps branch. There is no step-to-section mapping in the code, so only these two conditions are used.
- **Before -> after.** Mock review calls 14 -> 6, review prompt tokens 20,300 -> 9,760. Live: the check fired in C1, C2, C3, C5, C6 and C7 (6 of 8 runs).
- **Side effects.** A review that would have added a step for an uncovered section is no longer asked when all steps succeeded. The reflect step still triggers a gap review if a section is missing after writing. `tests/test_core.py` expectations of the LLM call sequence changed (4 calls -> 3) in four tests; the intent of each test is unchanged.

### T4. No review when no LLM provider is available; writing reserve

- **Problem.** Day 12 stress finding: when every provider is cooling down, a review call fails or waits, and steps it adds may need the LLM.
- **Change.** `LLMClient.any_provider_ready()`; `_node_review` skips the call when it is false and records why in the decision text.
- **Writing reserve.** Already correct, not rewritten. `core.py` keeps a quarter of the budget (at least 60 s, at most half) for writing, research calls use `deadline=_research_deadline`, and `llm.py` clips every rate-limit wait to that deadline. Regression tests: `test_rate_limit_waits_never_pass_the_research_deadline`, `test_the_writing_reserve_is_a_quarter_of_the_budget`. One request already in flight can still run to the provider timeout (60 s), which is not longer than the reserve.
- **Before -> after.** Live: the skip fired in C4 and C8 when Gemini and one Groq model were rate-limited.
- **Side effects.** If every provider is cooling the report is written from what was gathered (the writing call still tries the cooling providers last).

### P1. Forward-looking content in the report

- **Problem.** AD-4 Forward-Looking Analysis: 0-1 sections in 7 of 8 reports; target 2 (`evaluation_report.md`, "Priorities").
- **Change.** `SYNTHESIZE_PROMPT` rule: in at least two sections state the outlook, catalysts, risks to the thesis and what to watch next quarter, grounded in cited guidance, targets or estimates. With none, give a conditional view marked "Analysis:" and say no guidance was found. No invented forecast figures. AD-4 counts a section as forward-looking when its title names outlook/guidance/scenario or it has two sentences with forward wording (`evaluation/metrics.py` `ad4_forward_looking`); the metric was not changed.
- **Before -> after (live).** Sections counted: 4 across eight reports (mean 0.50) -> 7 (mean 0.88). Reports meeting the target of 2: 1 -> 2 (C2, C8). Reports with at least one: 3 -> 5.
- **Side effects.** FA-3 Temporal Accuracy fell from 8/8 to 4/8 and CS-3 Executive Summary Quality from 3/8 to 0/8. Forward sentences such as "Q4 FY2026 revenue growth of 9% to 11%" name periods the cited evidence does not contain, which FA-3 counts as errors. This is a probable link, not a proven one (different providers and judge variance). See `evaluation_report_v2.md`.

### P2. Tickers for company-specific steps

- **Problem.** C7 (cross-memory query) scored AB-3 2/5. The planner and the review scheduled `earnings_transcript` steps with no ticker; the executor answered "no ticker was specified". In `results/runs/C7.json`, 11 of 25 steps were skipped (9 `earnings_transcript`, 2 `web_search`), and the run took 23 LLM calls and 71,351 tokens.
- **Change.** Prompt: `_node_recall` passes the tickers recalled from memory (`memory_tickers`, without the `MULTI` label) and the plan prompt lists them and says company-specific tools need a ticker. Code: `enforce_tickers` in `agent/core.py` runs on the plan and on review steps. A step for `company_profile`, `financial_data_api`, `sec_filing_search`, `earnings_transcript` or `peer_comparison` with open arguments must name a known ticker in its goal. One candidate ticker is added to the goal; several candidates with no choice, or none, drop the step with a trace note.
- **Before -> after (live, C7).** Skipped steps 11 of 25 -> 0 of 16; LLM calls 23 -> 4; tokens 71,351 -> 28,447; AB-3 2 -> 2 (the judge still marked it down for repeated steps and missing news and peer steps). Also in C8 a review step was given its ticker (`set to NVDA`).
- **Side effects.** A step with open arguments that is genuinely company-independent but uses one of these tools would be dropped; none exists in the current tool set.

### Tool descriptions

Misses in the Day 11 bundles (`results/runs/C*.json`) and scores (`results/eval/scores.json`, AB-3 "expected tool not used"):

| Tool | Evidence | Edit to `tools/schemas/<tool>.json` |
|---|---|---|
| `earnings_transcript` | 11 of 21 steps skipped, 10 with "no ticker was specified" (C3, C4, C7) | Requires a ticker, never call without one, once per company and quarter |
| `calculation_engine` | 10 of 19 steps skipped, no numeric inputs available (C4, C5, C6) | Call only when the numbers are in the evidence; pass values, not names; skip when figures are missing |
| `web_search` | 6 of 9 steps skipped "no query, company, or topic"; AB-3 "expected tool not used" in C4, C8 | Always give a concrete query naming company and topic |
| `news_sentiment` | AB-3 "expected tool not used" in C2 | Call once per company, company name as the query, for any report with a sentiment or narrative discussion |

`fact_checker` and `vector_db_search`, also listed by AB-3, are system tools run by the agent, so their descriptions were not changed.

- **Cost.** About 650 characters added in total (110-160 per tool); this text is now only sent for the planned tool in execute calls, and in full in the plan prompt.
- **Before -> after (live).** Skipped steps across the eight runs: `earnings_transcript` 11 -> 0, `web_search` 6 -> 0, `calculation_engine` 10 -> 6 (the remaining six lack inputs: C4 and C8 plans referenced figures that were never retrieved). AB-3 mean 3.62 -> 3.75; plans passing AB-3: 5 of 8 -> 6 of 8. Several effects (P2, T1) change the same runs, so the share due to the descriptions cannot be separated.

### M1. Section-targeted memory recall

- **Problem.** AB-4 Memory Utilization failed in 7 of 8 runs (0-21%, mean 0.29). `_node_recall` made one query-wide search per ticker.
- **Change.** `_node_recall` runs, per ticker, the query-wide search and then up to two searches built from the template's first two content sections ("MSFT Business Overview", "MSFT Financial Summary"; `recall_topics`). Targeted hits must score at least `RECALL_MIN_SCORE` (0.30, similarity = 1 - cosine distance). Hits are de-duplicated by record id across all searches. Only searches with surviving hits are stored as evidence. The targeted searches are skipped when the query-wide search found nothing for that company. All are local; none uses the API budget.
- **Before -> after (live).** AB-4 mean 0.29 -> 0.60; runs meeting the 0.3 target 1 of 8 -> 7 of 8 (C6 stayed at 0: memory held nothing for the banks).
- **Side effects.** The AB-4 gain comes from the extra local recalls counted as memory hits; the number of external calls did not fall. It shows that the planner and writer now get more prior findings, not that fewer external calls were made. The metric was not changed. The relevance floor made no measurable difference to precision (see M2).

### M2. Chunk size and relevance floor

- **Method.** `python -m evaluation.memory_probe` (output `results/runs_day13/memory_probe.md` and `.json`; a first 24-probe run is kept in `memory_probe_first24.md`). 48 probes: a 9-word phrase from the middle of a stored chunk, filtered by ticker; a hit means one of the top k chunks contains the phrase. Stored chunks are re-split to try smaller sizes. 2,032 stored chunks in the live store. Probe phrases are lexically close to their chunk, so absolute rates are optimistic; the comparison is what counts.
- **Result.**

| Chunk size | hit@3 | hit@5 |
|---|---|---|
| 1500 chars (before) | 35% | 42% |
| 800 chars | 33% | 42% |
| 500 chars | 52% | 56% |
| 350 chars | 60% | 60% |
| 500 chars, 100 overlap | 60% | 65% |
| 800 chars, 150 overlap | 42% | 50% |

- **Change.** `memory/chunking.py` `MAX_CHARS` 1500 -> 500. The default embedding model reads at most about 256 tokens (about 1,000 characters), so the tail of a 1,500-character chunk is not embedded; this is a likely reason for the gain. 350 characters and overlap scored higher still but 350 chunks stop standing alone, and overlap needs changes in every chunker; both are left as options.
- **Relevance floor (negative result).** For "<TICKER> <topic>" queries (210 hits), the share of keyword-relevant hits was 62% at floors 0.00-0.30, 63% at 0.35-0.40 and 61% at 0.50, while the share of hits kept fell from 100% to 63%. The floor does not separate relevant from irrelevant hits by this proxy. 0.30 was kept as a guard against near-zero matches; it removes 3% of hits.
- **Alpha.** The architecture specification asked whether α of a hybrid re-ranker needs tuning. No hybrid re-ranker exists in the code, so there is no α. The open-question marker was updated.
- **Side effects.** New chunks are about 3x smaller and more numerous (5,469 against 2,387 for the same text). Chunks already in the store stay at their old size until re-ingested. Two unit tests that call `split_paragraphs` on text written for 1,500-character chunks now pass `1500` explicitly (`tests/test_memory.py`).

### Fix found on the way

`_node_execute` raised `ValueError: max_workers must be greater than 0` when no step was ready (an empty plan on a cross-memory query with no recalled company). The guard is now `len(ready) <= 1`.

## Summary

| ID | Change | Evidence before | Result |
|---|---|---|---|
| T1 | One tool schema per execute call | Execute = 56% of tokens | Mock execute tokens -45.6% |
| T2 | Per-provider synthesis budget, 90% headroom | One cap for all providers | Gemini 21,600, Groq 5,400 (live) |
| T3 | Deterministic review check | Review mostly "done" | Mock review calls 14 -> 6; live skipped in 6 of 8 |
| T4 | Skip review with no provider; reserve verified | Day 12 stress finding | Skipped in C4, C8; reserve held, tests added |
| P1 | Forward-view rule in synthesis prompt | AD-4 0-1 sections in 7 of 8 | Sections 4 -> 7; FA-3 8/8 -> 4/8 (regression) |
| P2 | Tickers for company-specific steps | C7: 11 of 25 steps skipped | C7: 0 of 16 skipped, 23 -> 4 LLM calls |
| Tools | Sharper descriptions for 4 tools | 27 skipped steps across 3 tools | 27 -> 6 (calculation_engine only) |
| M1 | Section-targeted recall | AB-4 0.29, 1 of 8 passing | AB-4 0.60, 7 of 8 passing |
| M2 | Chunk size 500 | hit@3 35% | hit@3 52%; floor made no difference |

Mock total tokens for eight challenges: 132,856 -> 92,055 (-30.7%); LLM calls 52 -> 44. Live total tokens: 257,229 -> 150,656; LLM calls 78 -> 37 (different providers and the Day 12 reference resolver also contribute).
