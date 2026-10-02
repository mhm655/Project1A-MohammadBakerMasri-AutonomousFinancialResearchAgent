# Agent Trace Gallery

Seven reasoning traces from ARA-1, chosen because each shows one capability or one failure that was later fixed. Three of the seven show the agent doing something poorly. All excerpts are real trace lines. No event was edited. Long lines are cut at the width shown and end with "…". Lines that were skipped are not marked, so the `#` numbers (event sequence numbers) jump.

## How to read and reproduce

Every entry names a trace file in `traces/` (one JSON line per event) and, where one exists, the run bundle that holds the same events in `trace_events`. Print any trace with the trace viewer:

```text
python -m agent.trace_viewer traces/session-20261001-211346-e773.jsonl
python -m agent.trace_viewer results/runs_day13/live/C6.json --types decision,error
python -m agent.trace_viewer traces/session-20261001-211346-e773.jsonl --step s1
```

`--types` keeps only the listed event types (`thought`, `decision`, `action`, `observation`, `error`, `llm_call`, `note`, `metric`). `--step` keeps events of one plan step. `--limit N` stops after N events. The excerpts below were taken from this output by sequence number.

Event types, in short: `thought` is free text from the agent. `decision` records a choice and its rationale. `action` is a tool call. `observation` is a tool result, given as an evidence id (E1, E2, ...). `error` is a failed tool or LLM call. `llm_call` is one answered model request with its token counts. `note` is a progress or resilience message. `metric` is a computed number.

| # | Capability | Trace | Quality |
|---|---|---|---|
| 1 | Query disambiguation | `session-20261001-211346-e773` | good |
| 2 | Memory-first recall and tickers from memory | `session-20261001-211503-a7a3`, `session-20260930-235312-ead7` | good, with an earlier failure that was fixed |
| 3 | Failure injection, retry, circuit breaker, fallback source | `session-20260930-235802-900b` | good, with costs |
| 4 | Provider fallback, and the whole chain failing | `session-20260928-163614-4cbc`, `session-20261001-002628-167b` | good, then poor; fixed |
| 5 | Verifier and an invented executive name | `session-20260930-235103-9899` | poor; fixed |
| 6 | Deterministic resolution of calculation inputs | `session-20261001-210053-defa`, `session-20260930-234336-1588` | good, with an earlier failure that was fixed |
| 7 | Synthesis continuation after an output cut-off | `session-20261001-211131-74ab` | good, with limits |

---

## 1. Query disambiguation: "What's happening with the banks?"

- **Capability.** The agent scores how vague a request is, states how it reads it, and goes on. It does not stop to ask.
- **Query.** `What's happening with the banks?`
- **Source.** `traces/session-20261001-211346-e773.jsonl`, Day 13 live run of challenge C6 (`results/runs_day13/live/C6.json`).

```text
#   2 thought      Report type 'company_profile' (ambiguous, sector); companies not identified; ambiguity 1.00 (high: +0.45: no company or ticker could be resolved; +0.20: the request names a group (…
#   3 decision     {"chosen": "disambiguate", "rationale": "Reading the request as US large-cap banks (JPM, BAC, C, WFC, GS, MS), for the latest reported period as of 2026-10-02, covering earnings an…
#   4 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "JPM"})
#   5 observation  vector_db_search -> E1
#   6 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "BAC"})
#   7 observation  vector_db_search -> E2
#   8 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "C"})
#   9 observation  vector_db_search -> E3
#  10 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "WFC"})
#  11 observation  vector_db_search -> E4
#  12 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "GS"})
#  13 observation  vector_db_search -> E5
#  14 action       vector_db_search({"query": "What's happening with the banks?", "top_k": 3, "ticker": "MS"})
```

**Annotation.**
- Event 2 gives the score (1.00, "high") and the reasons: no company named, a group named instead, no aspect, no period. The score is computed from rules, not guessed by the model.
- Event 3 turns the vague request into a concrete reading: six US large-cap banks, the latest reported period, four themes. The decision also stores three clarifying questions that were not asked, because the run was autonomous. The final report prints the reading and its assumptions under "Scope and assumptions", so the reader can see what was assumed.
- Events 4 to 14 start with memory lookups for each of the six tickers.
- Flaw. Memory held nothing about banks (event 16 in the full trace: "Memory recall found 0 prior finding(s)"), so these lookups added calls and no evidence. A different reading, such as regional banks, would have produced a report that answers a question nobody asked. One sentence flagged by the verifier in this run's bundle also says "five institutions" while the scope names six.

---

## 2. Memory-first recall, and a failure that was fixed

- **Capability.** For "the companies you've already researched", the agent reads the stored findings first and plans around the tickers it finds there.
- **Query.** `Based on the companies you've already researched, what themes emerge across the technology sector? Identify cross-cutting risks and opportunities.`
- **Source (good run).** `traces/session-20261001-211503-a7a3.jsonl`, Day 13 live C7 (`results/runs_day13/live/C7.json`).
- **Source (earlier failure).** `traces/session-20260930-235312-ead7.jsonl`, Day 11 C7 (`results/runs/C7.json`).

Day 13 run, good:

```text
#   2 thought      Report type 'thematic' (thematic, cross_memory); companies not identified; ambiguity 0.55 (medium: +0.45: no company or ticker could be resolved; +0.20: the request names a group (…
#   3 decision     {"chosen": "disambiguate", "rationale": "Reading the request as US mega-cap technology companies (MSFT, AAPL, NVDA, GOOGL, AMZN, META), for the latest reported period as of 2026-10…
#   4 thought      No company named; memory holds research on MSFT, TSLA, AMZN, GOOGL, PLTR, AAPL, MULTI, NVDA.
#   5 action       vector_db_search({"query": "Based on the companies you've already researched, what themes emerge across the technology sector? Identify cross-cutting risks and opportunities.", "to…
#   6 observation  vector_db_search -> E1
#   7 action       vector_db_search({"query": "MSFT Themes", "top_k": 3, "ticker": "MSFT"})
#  53 thought      Memory recall found 62 prior finding(s) and strategy notes from earlier runs.
#  54 error        llm:gemini quota_exhausted: [gemini] quota_exhausted: Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing det…
#  55 llm_call     groq/openai/gpt-oss-120b 5337+2040 tok, 4918.8 ms
#  57 decision     {"chosen": "plan", "rationale": "Identify common strategic themes, risks, and opportunities across leading technology firms by extracting recent financial performance, management c…
#  58 decision     {"chosen": "company_profile", "rationale": "Establish fiscal year end and key executives for consistent period alignment", "step_id": "s1", "args": {"ticker": "MSFT"}, "depends_on"…
#  59 decision     {"chosen": "company_profile", "rationale": "Establish fiscal year end and key executives for consistent period alignment", "step_id": "s2", "args": {"ticker": "AAPL"}, "depends_on"…
#  60 decision     {"chosen": "company_profile", "rationale": "Establish fiscal year end and key executives for consistent period alignment", "step_id": "s3", "args": {"ticker": "GOOGL"}, "depends_on…
```

Day 11 run, same query, before the fix (only the skipped-step notes):

```text
#  91 note         {"text": "step r2 skipped: not needed: Cannot call earnings_transcript because the target company's ticker symbol was not provided in the prompt.", "step_id": "r2"}
#  97 note         {"text": "step r6 skipped: not needed: No specific company or topic was provided for the web_search, so the step cannot be executed without a defined query.", "step_id": "r6"}
# 104 note         {"text": "step r4 skipped: not needed: No ticker was specified, so the earnings_transcript tool cannot be called.", "step_id": "r4"}
# 126 note         {"text": "step r9 skipped: not needed: No ticker was specified, so the earnings_transcript tool cannot be called.", "step_id": "r9"}
# 137 note         {"text": "step r14 skipped: not needed: No specific query was provided for the web_search step, so the tool cannot be called without a target company or topic.", "step_id": "r14"}
# 140 note         {"text": "step r12 skipped: not needed: Cannot call earnings_transcript because the target company's ticker symbol was not provided in the prompt.", "step_id": "r12"}
```

**Annotation.**
- Event 4 reads the stored tickers from memory. Event 53 reports 62 prior findings. The plan (event 57) then gives each recalled company a `company_profile` step with its ticker.
- The Day 11 run did not pass tickers on. Its plan had 25 steps and 11 were skipped, mostly `earnings_transcript` with "No ticker was specified". The run took 23 LLM calls and 71,351 tokens (prompt plus completion).
- The Day 13 fix passes recalled tickers into the plan prompt and drops company steps that have none (`enforce_tickers`, `docs/optimization_log.md` entry P2). The same challenge then ran 16 steps with none skipped, 4 LLM calls and 28,447 tokens.
- Remaining flaw. Memory also held Palantir (PLTR), but the plan covers six other tickers (MSFT, AAPL, GOOGL, AMZN, TSLA, NVDA). Every company step in the plan uses one of those six tickers, and the disambiguation reading in event 3 names a slightly different set (it lists META, which memory does not hold). So memory fed the evidence but did not fully set the company list. The judge still gave this run 2 of 5 for tool use (AB-3) in Day 11 and the fix did not raise it (`docs/optimization_log.md`, P2).

---

## 3. Failure injection, retry, circuit breaker and fallback source

- **Capability.** With 50% of calls to two tools failing on purpose, the agent retries, trips a circuit breaker, switches to another source and says so in the report.
- **Query.** `Produce a complete investment research report on NVIDIA Corporation. Note: The financial data API and SEC filing search tools are currently experiencing intermittent failures (simulate 50% failure rate).`
- **Source.** `traces/session-20260930-235802-900b.jsonl`, Day 11 official run of challenge C8 (`results/runs/C8.json`, `results/challenge_8.md`).

```text
#  18 action       financial_data_api({"ticker": "NVDA", "statement_type": "income_statement", "period": "annual", "years": 3})
#  19 error        tool:financial_data_api execution_error: HTTP 503 from injected failure for financial_data_api
#  20 note         {"text": "resilience retry: tool=financial_data_api, attempt=1, error_class=transient, delay_s=1.1354949290359497"}
#  24 action       sec_filing_search({"ticker": "NVDA", "filing_type": "10-K"})
#  25 error        tool:sec_filing_search execution_error: sec_filing_search: request timed out (injected)
#  26 note         {"text": "resilience breaker_open: tool=sec_filing_search, provider=sec_edgar, reason=tripped"}
#  29 note         {"text": "resilience fallback_failed: tool=financial_data_api, fallback=sec_filing_search, error_class=unavailable, message=circuit open for sec_edgar"}
#  36 action       web_search({"query": "NVDA income statement financials", "num_results": 10})
#  37 observation  web_search -> E3
#  38 note         {"text": "resilience fallback_used: tool=financial_data_api, fallback=web_search, error_class=transient"}
#  42 note         {"text": "resilience fallback_used: tool=sec_filing_search, fallback=web_search, error_class=timeout"}
```

**Annotation.**
- Event 19 is an injected HTTP 503. Event 20 is a retry with a delay (the retry text names the error class: transient). Event 22 (not shown) is an injected malformed reply, retried the same way.
- The timeout in event 25 trips the breaker for `sec_edgar` (event 26). Both tools use that provider, so later calls to either are refused while the breaker is open (events 27 and 28, not shown).
- Event 29 shows a cost of the design. `financial_data_api` tried `sec_filing_search` as a fallback, and the breaker refused it. The agent then used `web_search` (events 36 to 38 and 42). A search result is weaker than a statement table, and the report marks fallback data as lower confidence.
- Result from the run bundle: 4 injected failures, 4 recovered, 4 gaps named in the report's "Data Gaps and Limitations", all 7 template sections present (`results/stress_test_report.md`). Honest limit: the report is thin, about 1,500 words, because the free Groq model wrote it.

---

## 4. Provider fallback, and the whole chain failing

- **Capability.** When the preferred model fails, the agent moves down a chain (Gemini, Groq gpt-oss, Groq Qwen, local Ollama) and keeps working. When every provider fails, it should degrade without crashing.
- **Queries.** `Create a comprehensive profile of Microsoft Corporation ...` (first block) and `What's happening with the banks?` (second block).
- **Sources.** `traces/session-20260928-163614-4cbc.jsonl` (first Microsoft run, before the bundle format existed). `traces/session-20261001-002628-167b.jsonl` (Day 12 re-run C6_rerun2, `results/runs_day12/C6_rerun2.json`). `traces/session-20261001-211753-ef30.jsonl` (Day 13 C8) for the fix.

Fallback working (planning call):

```text
#   6 error        llm:gemini server: [gemini] server: Error code: 503 - [{'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in…
#   7 error        llm:gemini server: [gemini] server: Error code: 503 - [{'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in…
#   8 error        llm:gemini server: [gemini] server: Error code: 503 - [{'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in…
#   9 llm_call     groq/openai/gpt-oss-120b 1237+1692 tok, 4076.5 ms
#  10 decision     {"chosen": "plan", "rationale": "Gather core company data, financial statements, executive details, and recent news to build a complete profile of Mic…
```

Chain exhausted (Day 12, last provider errors and review) and the fix (Day 13):

```text
#  73 error        llm:groq quota_exhausted: [groq] quota_exhausted: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m3m8s…
#  82 error        llm:groq_qwen quota_exhausted: [groq_qwen] quota_exhausted: Error code: 429 - {'error': {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_…
#  83 error        llm:ollama connection: [ollama] connection: Connection error.
#  86 error        review: AllProvidersFailedError: All LLM providers failed (4 attempts): [gemini] quota_exhausted: daily quota used up; skipped; [groq] quota_exhausted: daily quota used u…
#  87 decision     {"chosen": "research_complete", "rationale": "review skipped: no LLM provider is available right now (all are rate-limited or out of quota); writing the report with what was gather…
```

**Annotation.**
- First block. Gemini answers 503 three times ("high demand"). The plan call then succeeds on Groq (event 9, 1,237 prompt and 1,692 completion tokens). The user sees no difference except latency.
- Second block, flaw. On Day 12 all free quotas were used up. Gemini, Groq and Qwen answered `quota_exhausted`, Ollama had no server, and the review step failed with `AllProvidersFailedError`. That run produced no report text, and the stress report lists it as not judged.
- The fix is in the last line (Day 13, event 87). The review step now checks whether any provider is ready, skips the review if none is, and writes the report from what was gathered (`docs/optimization_log.md`, T4). The decision text states why, so the skip is visible in the trace.
- Remaining flaw. A provider that is out of quota is still tried on every call in some runs, which costs latency. This was logged as "found, not yet fixed" in `results/stress_test_report.md`.

---

## 5. Verifier and an invented executive name

- **Capability.** After writing, the agent checks the report against its sources. This entry shows what the check missed before Day 12 and what it checks now.
- **Query.** `What's happening with the banks?`
- **Source.** `traces/session-20260930-235103-9899.jsonl`, Day 11 official run of C6 (`results/runs/C6.json`). Compare `results/runs_day13/live/C6.json` (Day 13).

```text
#  80 note         {"text": "step s12 done: financial_data_api returned E19", "step_id": "s12"}
#  86 note         {"text": "step s13 skipped: not needed: The available income-statement data only includes total revenue and net income; it does not provide net interest income, which is …
# 100 note         {"text": "step r1 skipped: not needed: The step goal is underspecified \u2014 no query, company, or topic was provided, so I cannot construct a meaningful web_search call…
# 102 note         {"text": "step r2 skipped: not needed: The step goal is underspecified \u2014 no query, company, or topic was provided, so I cannot construct a meaningful web_search call…
# 112 error        llm:groq rate_limit: [groq] rate_limit: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m3m8s4dne45twcs…
# 113 llm_call     groq/openai/gpt-oss-120b 5471+1583 tok, 4119.0 ms
# 114 metric       {"name": "verification", "value": {"checked": 9, "supported": 6, "partially_supported": 0, "unsupported": 3, "unverifiable": 0, "uncited": 15, "auto_cited": 5}}
# 115 thought      Reflection: 0 section(s) missing
# 116 action       report_generator({"template": "company_profile", "title": "JPM, BAC, C, WFC, GS, MS: Company Profile", "sections": {"executive_summary": "Bank of America, JPMorgan Chase,…
# 116 action       report_generator(… Sullivan, Chairman & CEO [E11].  \n- **Morgan Stanley** \u2013 James G. P. Hunt, Chairman & CEO [E12].  \n- **Wells Fargo** \u2013 Charlie Schwab, CEO [E8].\n\n*(Executive roles are derived from the company profiles; specific titles are as listed in the profiles.)* …
```

**Annotation.**
- Event 86 shows a step skipped because the data lacked a figure (net interest income). Events 100 and 102 show the executor refusing to invent a search query ("without inventing content"). Event 114 is the verifier: 9 claims checked, 3 unsupported, 15 uncited figures. It looked at numbers only.
- The last line is a cut from event 116, the report text sent to the report generator. It names "Charlie Schwab" as CEO of Wells Fargo. The company profile in the evidence (E8 in `results/runs/C6.json`) lists "Charles W. Scharf". The report also gives "David M. Sullivan" as head of Goldman Sachs. The factual-accuracy metric FA-4 found both and no check in the agent did.
- The fix (Day 12, `tools/fact_checker.unsupported_names`, called from `agent/core.py`) flags an executive named next to a title who appears in no source. When a name is flagged, the verification metric gains an `unsupported_names` count. The Day 13 run of the same query (`results/runs_day13/live/C6.json`) contains neither "Schwab" nor "Scharf", so it does not repeat the error. On the eight live Day 11 reports the check flags exactly the three real errors in C6 and nothing else (`results/stress_test_report.md`).
- Honest limit. No saved trace shows the new check firing on a live run. The evidence is its test (`test_verifier_flags_executives_that_no_source_names`) and the stress report's replay over the eight reports. The check also covers only executives, not other proper nouns.

---

## 6. Deterministic resolution of calculation inputs

- **Capability.** The planner names the numbers a calculation needs ("revenue FY2026"). The agent looks them up in the retrieved data in code, with no LLM call, and says where each value came from.
- **Query.** `Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, key executives, and recent developments.`
- **Sources.** `traces/session-20261001-210053-defa.jsonl`, Day 13 live C1 (`results/runs_day13/live/C1.json`). Earlier failure: `traces/session-20260930-234336-1588.jsonl`, Day 11 C4 (`results/runs/C4.json`).

```text
#  35 decision     {"chosen": "resolve_inputs", "rationale": "inputs taken from the financial data: current = revenue FY2026, previous = revenue FY2025", "step_id": "s5"}
#  36 decision     {"chosen": "resolve_inputs", "rationale": "inputs taken from the financial data: numerator = operating_income FY2026, revenue = revenue FY2026", "step_id":…
#  37 decision     {"chosen": "resolve_inputs", "rationale": "inputs taken from the financial data: numerator = net_income FY2026, revenue = revenue FY2026", "step_id": "s7"}
#  38 action       calculation_engine({"calculation_type": "growth_rate", "inputs": {"current": 331839.0, "previous": 281724.0}})
#  39 observation  calculation_engine -> E8
#  40 action       calculation_engine({"calculation_type": "margin", "inputs": {"numerator": 133749.0, "revenue": 331839.0}})
#  41 observation  calculation_engine -> E10
#  54 decision     {"chosen": "add_steps", "rationale": "The core gap is that the 10-K excerpts (E7, E8, E9) returned only risk-factor narratives and did not contain the segm…
#  55 action       calculation_engine({"calculation_type": "growth_rate", "inputs": {"current": "AWS FY2025 revenue", "previous": "AWS FY2024 revenue"}})
#  56 error        tool:calculation_engine invalid_arguments: calculation_engine: input 'current' must be a plain number, got 'AWS FY2025 revenue'
#  61 action       calculation_engine({"calculation_type": "growth_rate", "inputs": {"current": "Azure FY2025 revenue", "previous": "Azure FY2024 revenue"}})
#  62 error        tool:calculation_engine invalid_arguments: calculation_engine: input 'current' must be a plain number, got 'Azure FY2025 revenue'
```

**Annotation.**
- Events 35 to 37 are `resolve_inputs` decisions. Each lists the source of each value. Events 38 and 40 then call the calculator with real numbers (growth from 281,724 to 331,839, which is 17.8%). No model call was needed.
- If a year is not reported yet, `agent/references.py` moves all years back together and the report says so. Here FY2026 is available, so no shift was needed.
- The last five lines are Day 11 C4, which asked for AWS, Azure and Google Cloud growth. The planner passed text ("AWS FY2025 revenue") where numbers were required. The calculator refused three times in this trace and the review step added steps to repair it. That cost an LLM call each time and it failed in C4 and C5 (`results/stress_test_report.md`, bug 2).
- Limit. Segment names such as AWS and Azure are left alone on purpose, because company-level statements do not hold them. The 10-K excerpts in C4 held no segment revenue, so deterministic resolution cannot help there. In the Day 13 C8 trace (`session-20261001-211753-ef30`, events 54 to 59) the plan still passed `null` inputs after the data calls had failed, and the calculator rejected them.

---

## 7. Synthesis continuation after an output cut-off

- **Capability.** When the writer is cut off by its output limit, the agent notices which sections are missing and asks for them, up to twice, without overwriting finished sections.
- **Query.** `Research Palantir Technologies. Note: Recent news reports suggest the company is struggling ...` (challenge C5, a conflicting-narrative query).
- **Source.** `traces/session-20261001-211131-74ab.jsonl`, Day 13 live C5 (`results/runs_day13/live/C5.json`).

```text
#  51 metric       {"name": "synthesis_ledger", "value": {"claims": 50, "by_status": {"single_source": 50}, "conflicts": 0, "conflict_methods": [], "threads": ["growth", "profitability", "c…
#  52 metric       {"name": "synthesis_prompt_budget", "value": {"provider": "groq", "budget_tokens": 5400, "overhead_tokens": 1545}}
#  53 error        llm:groq rate_limit: [groq] rate_limit: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m3m8s4dne45twcs…
#  54 llm_call     groq_qwen/qwen/qwen3.8-27b 5749+1000 tok, 2637.3 ms
#  55 thought      Synthesis was cut off by the output limit; continuing with ['implications']
#  56 error        llm:groq_qwen rate_limit: [groq_qwen] rate_limit: Error code: 429 - {'error': {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_01m3m8s4dn…
#  61 llm_call     groq_qwen/qwen/qwen3.8-27b 5833+656 tok, 2041.4 ms
#  62 metric       {"name": "verification", "value": {"checked": 25, "supported": 19, "partially_supported": 3, "unsupported": 3, "unverifiable": 0, "uncited": 0, "auto_cited": 0}}
#  63 thought      Reflection: 0 section(s) missing
#  64 action       report_generator({"template": "investigative", "title": "Palantir: Investigative", "sections": {"executive_summary": "Palantir Technologies is a software infrastructure c…
```

**Annotation.**
- Event 54 shows the writer call stopping at exactly 1,000 completion tokens. That is the output cap of the Groq Qwen model used after the larger models were rate-limited.
- Event 55 names the missing section: `implications`. The continuation (event 61) writes only that section, 656 tokens. The verifier then runs on the whole report (event 62).
- Before Day 12 a cut report simply ended mid-sentence. In the Day 12 re-runs, C8 lost 4 of 7 sections this way. The fix is `test_a_report_cut_off_by_the_output_limit_is_continued` in the tests.
- Flaws. The continuation call hit rate limits four times (events 56 to 60) before it succeeded, and about 60 seconds passed between events 55 and 61. The verifier then found 3 of 25 claims unsupported and 3 partially supported, so a complete report is not the same as a fully checked one. A continuation also needs time left before the deadline. In the second Day 12 C8 re-run the time was gone and the report stayed cut (`results/stress_test_report.md`).
