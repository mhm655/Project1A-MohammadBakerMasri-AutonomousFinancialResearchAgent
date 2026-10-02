# ARA-1 Final Evaluation Report

This report summarises every metric result for the Autonomous Financial Research Agent (ARA-1). It covers the Day 11 baseline, the Day 13 re-run after the optimisations, the stress tests and the token use. Numbers taken from `scores.json` were recomputed with a script for this report. Numbers that come only from a Markdown report are marked "copied".

Sources:

| Item | File |
|---|---|
| Day 11 scores | `results/eval/scores.json`, `results/evaluation_report.md` |
| Day 13 scores | `results/runs_day13/eval/scores.json`, `results/evaluation_report_v2.md` |
| Stress tests | `results/stress_test_report.md` |
| Token use | `results/token_usage_analysis.md`, `results/runs_day13/token_before_after.md` |
| Changes between the runs | `docs/optimization_log.md` |
| Example traces | `docs/trace_gallery.md` |

## 1. Executive summary

ARA-1 takes a research question about a listed company and writes a cited report. We tested it on eight standard challenges (C1 to C8), twice, with 23 metrics each. Between the two runs we changed the agent to cut cost and fix the weak points the first run showed.

- **Overall.** 53% of the measured checks passed in the first run (93 of 174). 61% passed in the second (103 of 169). This is an improvement, but less than two thirds of checks still pass.
- **What works well.** The numbers in the reports are almost always traceable to a source. Every report had every required section. Every run finished well inside its time limit. When we made two data tools fail on purpose, the agent recovered, said what had failed, and still produced a full report.
- **What does not work well yet.** The reports are short and shallow. They give few original calculations and few insights per page (0 of 8 reports reached the insight target in either run). Executive summaries scored only a middling mark from the judge. Benchmark recall of key facts is low (about 55% to 61% against a 70% target).
- **Headline targets.** Tool efficiency (AB-1, target above 70%) averaged 69.9% in the first run and 71.5% in the second. It is met on the average only in the second run, and only 5 of 8 runs pass individually. The hallucination rate (FA-5) is 1.4% and 1.6% of numeric claims, under the 2% line, but the sample is small and the spec's own per-report target is zero, which 6 of 8 reports meet.
- **Cost.** The live runs used 41% fewer LLM tokens (257,229 to 150,656) and 53% fewer LLM calls (78 to 37).
- **Caution.** Each challenge was run once. Different AI providers answered in the two runs. Differences of one or two checks are within the noise. Section 7 lists the limits.

## 2. Results by category

Pass counts include only metrics that could be measured in a run. A metric cannot be measured when its input is missing (for example AB-2 needs a tool error to recover from, and CO-4 needs a risk-factor section in the source filing). "Rate" is passes divided by measured checks. A mean of metric values across a category would mix units (ratios, counts, seconds), so means are given per metric in the next table.

| Category | Metrics | Day 11 pass | Day 11 rate | Day 13 pass | Day 13 rate |
|---|---|---|---|---|---|
| Factual Accuracy (FA) | 5 | 31/40 | 78% | 31/40 | 78% |
| Completeness (CO) | 4 | 18/28 | 64% | 20/26 | 77% |
| Analytical Depth (AD) | 4 | 6/32 | 19% | 11/32 | 34% |
| Coherence and Structure (CS) | 4 | 15/32 | 47% | 11/32 | 34% |
| Agent Behaviour (AB) | 5 | 23/39 | 59% | 29/36 | 81% |
| Benchmark (BM) | 1 | 0/3 | 0% | 1/3 | 33% |
| All | 23 | 93/174 | 53% | 103/169 | 61% |

Per challenge (checks passed out of measured):

| Challenge | Day 11 | Day 13 |
|---|---|---|
| C1 Single-company profile | 13/21 | 16/21 |
| C2 Earnings analysis | 13/22 | 13/21 |
| C3 Risk assessment | 13/23 | 17/23 |
| C4 Industry comparison | 9/22 | 11/22 |
| C5 Contradictory data | 14/22 | 13/21 |
| C6 Ambiguous query | 8/21 | 9/20 |
| C7 Sector analysis with memory | 11/22 | 10/20 |
| C8 Full report with degradation | 12/21 | 14/21 |

The benchmark (BM-1) ran for three challenges only (C1, C2 and C3, the ones with gold facts in `evaluation/benchmarks/`).

### Per metric

Pass is the number of runs meeting the target. Mean is the mean value over the runs where the metric was measured. Targets are those in `evaluation/metrics.py`.

| Metric | Target | D11 pass | D13 pass | D11 mean | D13 mean |
|---|---|---|---|---|---|
| FA-1 Numerical Accuracy Rate | >= 98% | 6/8 | 6/8 | 0.99 | 0.98 |
| FA-2 Citation Accuracy | 100% | 4/8 | 7/8 | 0.91 | 0.98 |
| FA-3 Temporal Accuracy | 100% | 8/8 | 4/8 | 1.00 | 0.94 |
| FA-4 Entity Accuracy | 100% | 7/8 | 8/8 | 0.98 | 1.00 |
| FA-5 Hallucination Rate (count) | 0 | 6/8 | 6/8 | 0.25 | 0.38 |
| CO-1 Section Coverage | 100% | 8/8 | 8/8 | 1.00 | 1.00 |
| CO-2 Data Source Diversity | >= 4 types | 4/8 | 6/8 | 3.50 | 4.12 |
| CO-3 Temporal Coverage | >= 3 years | 6/8 | 6/8 | 3.12 | 3.00 |
| CO-4 Risk Factor Coverage | >= 80% | 0/4 | 0/2 | 0.49 | 0.42 |
| AD-1 Insight Density | >= 3 per page | 0/8 | 0/8 | 1.52 | 1.46 |
| AD-2 Cross-Source Synthesis | >= 5 | 4/8 | 7/8 | 4.50 | 8.75 |
| AD-3 Quantitative Reasoning | >= 10 calculations | 1/8 | 2/8 | 5.50 | 8.12 |
| AD-4 Forward-Looking Analysis | >= 2 sections | 1/8 | 2/8 | 0.50 | 0.88 |
| CS-1 Logical Flow | >= 4 of 5 | 4/8 | 2/8 | 3.50 | 3.25 |
| CS-2 Internal Consistency (count) | 0 | 6/8 | 6/8 | 0.25 | 0.50 |
| CS-3 Executive Summary Quality | >= 4 of 5 | 3/8 | 0/8 | 3.38 | 3.00 |
| CS-4 Professional Formatting | >= 80% | 2/8 | 3/8 | 0.75 | 0.79 |
| AB-1 Tool Efficiency | >= 70% | 4/8 | 5/8 | 0.70 | 0.72 |
| AB-2 Error Recovery Rate | >= 90% | 5/7 | 3/4 | 0.71 | 0.90 |
| AB-3 Planning Quality | >= 4 of 5 | 5/8 | 6/8 | 3.62 | 3.75 |
| AB-4 Memory Utilization | >= 0.3 | 1/8 | 7/8 | 0.29 | 0.60 |
| AB-5 Latency (seconds) | < 300 s | 8/8 | 8/8 | 110 | 95 |
| BM-1 Benchmark Fact Recall | >= 70% | 0/3 | 1/3 | 0.55 | 0.61 |

Notes on the table:
- AB-5 uses a longer limit (600 s) for multi-company challenges. The slowest Day 11 run was C7 at 246 s.
- The AB-1 means are 0.699 (Day 11) and 0.715 (Day 13) before rounding.
- Day 13 measured fewer checks (169 against 174) because fewer runs had a tool error (AB-2 measured in 4 runs instead of 7) or a risk-factor section (CO-4 in 2 runs instead of 4).

## 3. The two headline targets

### Tool efficiency above 70% (AB-1)

AB-1 in this project is defined in `evaluation/metrics.py` as the number of research evidence records cited in the report body divided by the number of tool calls. A call whose result is never cited counts against the agent.

| | Day 11 | Day 13 |
|---|---|---|
| Mean over 8 runs | 0.699 | 0.715 |
| Runs at or above 0.70 | 4 of 8 | 5 of 8 |
| Per run, C1 to C8 | 1.00, 0.83, 0.70, 0.59, 0.56, 0.82, 0.47, 0.63 | 1.00, 1.00, 0.75, 0.50, 0.78, 0.86, 0.44, 0.40 |

Verdict: **not met on the mean in Day 11 (69.9%), met on the mean in Day 13 (71.5%), but only just, and not in 3 of the 8 runs.** The low runs are the long multi-company ones (C4, C7, C8). The gain is within what a change of provider could cause, so it should not be called a firm improvement.

### Hallucination below 2% (FA-5)

FA-5 is defined in `evaluation/metrics.py` as a count of numeric claims that are neither found in any evidence record nor derivable from it (a ratio or growth of two source amounts). The spec target is 0 per report. The 2% target needs a rate, so it is computed here as untraceable numeric claims divided by all numeric claims checked.

| | Day 11 | Day 13 |
|---|---|---|
| Untraceable numeric claims | 2 | 3 |
| Numeric claims checked | 141 | 193 |
| Rate | 1.4% | 1.6% |
| 95% interval for the rate (Wilson) | 0.4% to 5.0% | 0.5% to 4.5% |
| Reports with count 0 | 6 of 8 | 6 of 8 |

Verdict: **below 2% in both runs on the pooled rate, but the evidence is thin.** The interval includes values above 2%, and single reports exceeded 2% (for example C6 on Day 13 had 2 of 16 claims, 12.5%). The strict per-report target of zero is met in 6 of 8 reports in each run. The two Day 11 cases were reviewed by hand in `results/evaluation_report.md`: one was a correct figure with a wrong citation, one could not be confirmed. FA-5 counts numbers only. Invented names are measured by FA-4, which was 100% in Day 13 after the Day 12 verifier fix. Other false statements in prose are not measured by any automated metric.

## 4. Stress tests (copied from `results/stress_test_report.md`)

The stress tests ran in mock mode (fake LLM and fake data) on 2026-10-01, so they test the code, not the free live providers.

| Test | Result | Key numbers |
|---|---|---|
| Concurrency, 5 agents at once | Pass | 5 of 5 completed, 0.619 s concurrent against 1.703 s sequential, no cross-talk between runs |
| Maximum context use | Pass | largest request 3,580 of 6,000 tokens, none over |
| All external APIs failing | Pass | C1 and C8 completed in under 5 s with 3 LLM calls each |
| Challenge 8, 50% failure injection (live) | Pass | 4 failures injected, 4 recovered, 4 gaps disclosed, all 7 sections present |

Day 12 also found and fixed bugs through these tests and the live re-runs: invented executive names (the verifier checked numbers only), calculation inputs written as names, reports cut off by an output limit, and failure injection that hit more tools than the spec named. The live re-runs on Day 12 could not show an overall gain because the free provider quotas ran out during them. The Day 13 re-run is the like-for-like check.

Not tested: five concurrent live runs (they would exhaust the free quotas) and shared memory under concurrency beyond the mock store, which the report notes is shared mutable state.

## 5. Token use

| | Day 11 | Day 13 | Change |
|---|---|---|---|
| Live tokens (prompt + completion) | 257,229 | 150,656 | -41.4% |
| Live LLM calls | 78 | 37 | -52.6% |
| Live tool calls | 89 | 92 | +3 |
| Live total run time, 8 challenges | 877 s | 759 s | -13.4% |
| Mock tokens (scripted LLM, same code paths) | 132,856 | 92,055 | -30.7% |
| Mock LLM calls | 52 | 44 | -8 |

Where the tokens went in Day 11 (copied from `results/token_usage_analysis.md`, which counts reasoning tokens, total 260,476): execute calls 55.8%, synthesis 22.3%, review 12.3%, planning 9.6%. The largest cost was sending every tool schema in every execute call. Day 13 sends only the schema of the planned tool, skips the review call when nothing needs reviewing, and resolves calculation inputs in code. The mock figure isolates these changes. The live figure also reflects a different provider mix, so it is indicative only.

Example: C7 went from 25 planned steps with 11 skipped, 23 LLM calls and 71,351 tokens, to 16 steps with none skipped, 4 calls and 28,447 tokens. The report itself did not get better (see weaknesses).

## 6. Strengths and weaknesses

### Strengths

1. **Figures are grounded.** Numerical accuracy averages about 98% (0.985 and 0.979). Temporal accuracy was 100% in all eight Day 11 reports. Entity accuracy was 100% in all eight Day 13 reports. Untraceable numbers are 1.4% to 1.6% of all numeric claims.
2. **Structure and reliability.** CO-1 (all required sections) is 8 of 8 in both runs. Latency is within target in 16 of 16 runs. Mean run time fell from 110 s to 95 s.
3. **Graceful degradation.** With injected tool failures the agent retried, opened a circuit breaker, used a lower-confidence fallback source and disclosed each gap. Error recovery (AB-2) averaged 0.90 on Day 13 (3 of 4 measured runs pass).
4. **Memory use.** AB-4 passes in 7 of 8 runs, up from 1 of 8, after the section-targeted recall change. Caveat: the gain comes from more local recalls being counted, not from fewer external calls.
5. **Cost.** 41% fewer live tokens and 53% fewer LLM calls, with no loss of section coverage.
6. **Handling of vague and conflicting queries.** The agent states its reading of an ambiguous question and records its assumptions in the report. C5 (Palantir's strong numbers against negative news) produced a cross-source explanation with 8 synthesis claims in the Day 11 run, and AD-2 passes in 7 of 8 Day 13 runs. See `docs/trace_gallery.md`.

### Weaknesses and likely causes

| Weakness | Evidence | Likely cause |
|---|---|---|
| Shallow analysis | AD category 19% to 34% pass. AD-1 0 of 8 in both runs (mean about 1.5 per page against 3). AD-3 2 of 8 in Day 13. | Small writing budget on the free Groq models (6,000-token prompts, 1,000-token output on Qwen). Reports are 651 to 1,651 words against a 3 to 20 page spec. |
| Weak executive summaries and flow | CS-3 0 of 8 in Day 13 (3 of 8 in Day 11). CS-1 2 of 8 (was 4 of 8). | Probably the new forward-looking text in summaries and judge variance. Not proven. |
| Temporal errors rose | FA-3 8 of 8 to 4 of 8. | Forward-looking sentences name periods the cited evidence does not contain (for example "Q4 FY2026"). Probable link to the outlook prompt, not isolated. |
| Low benchmark recall | BM-1 mean 0.55 and 0.61 against 0.70. | Reports skip core lines (operating income, net income, EPS) that the gold summaries contain. No reported figure contradicted a gold value. |
| Risk coverage | CO-4 0 of 4 and 0 of 2 measured runs. | The check is a 12-category keyword match against the 10-K risk section. The writer covers fewer categories than the filing. |
| Cross-memory report got narrower | C7 passed 11 of 22, then 10 of 20. AD-2 9 to 0, AD-3 20 to 2. | After the ticker fix the plan used only three tools and no calculation, news or peer step. The process became cheaper and valid, the report less rich. |
| Tool efficiency in long runs | AB-1 0.40 to 0.50 in C4, C7 and C8 on Day 13. | Many calls whose results are not cited, and skipped calculation steps when data was missing. |
| Dependence on free providers | Day 11 used Groq for 75 of 78 calls (copied). Gemini and Groq quotas ran out during Day 12. One re-run produced no report text. | Free-tier daily and per-minute limits. A skip-review rule now keeps the agent writing when no provider is ready. |
| Invented executive names (Day 11) | C6 named the wrong CEOs for Wells Fargo and Goldman Sachs. | The verifier checked numbers only. Fixed on Day 12 for executives. Other proper nouns are still unchecked. |

### What would come next

1. Re-run both sets on one model with a full quota, several times per challenge, so that provider and judge noise can be separated from real change.
2. Give the writer a larger budget (a stronger model or a split of the report into two writing calls) and aim at AD-1, AD-3 and BM-1, the three weakest groups.
3. Keep the forward-looking text but require each forward claim to name a period that appears in the evidence. Re-check FA-3 and CS-3.
4. Give C7-type questions a plan that includes calculation, news and peer steps for recalled companies, so memory improves the report and not only the cost.
5. Extend the name check to other proper nouns (companies, products, places) and add a prose-claim check for FA-5.
6. Raise AB-1 by not planning steps whose result is unlikely to be cited, and by retrying skipped calculations when a provider returns.
7. Re-ingest memory with the new chunk size, so retrieval gains apply to old records.

## 7. Methodology and limitations

**How the scores were produced.** `evaluation/runner.py` ran C1 to C8 on live data and wrote one run bundle per challenge. `evaluation/evaluate.py` scored each bundle on 23 metrics: the 22 in the spec plus BM-1, recall of gold-standard facts. Seventeen metrics are automated, three use an LLM judge (AD-1, CS-1, CS-3), two are hybrids (CS-2, AB-3) and BM-1 is a benchmark comparison. The Day 11 scores were checked in two review passes that fixed 13 measurement errors, each with a test. The pass rate moved from 51% to 53% and no check was loosened to improve a score.

**Limitations.**

- **One run per challenge.** There are no repeats, so there is no variance estimate. A difference of one or two runs on a metric is not evidence of change.
- **Mixed providers.** Day 11 used Groq (`gpt-oss-120b` and `qwen3.8-27b`) for 75 of 78 calls because the Gemini quota was spent. Day 13 used Gemini for 6 of 37 calls (C1 to C3) and Groq and Qwen for the rest. Gemini is the stronger model, so part of any difference in C1 to C3 may come from the provider. Counts of calls per provider are copied from `results/evaluation_report_v2.md`.
- **LLM-judge variance.** The judge was a single call per report. In Day 11 it was the Qwen model, the same family that wrote some reports, so scores may be lenient. Judge scores move by a point between runs of the same bundle (copied from the Day 13 report). AD-1, CS-1, CS-3 and AB-3 are the affected metrics.
- **Metric definitions.** FA-2 treats a sentence as supported if 30% of its content words appear in the cited source. CO-4 is a keyword taxonomy. AB-1 rewards citation of evidence, not usefulness of the call. AB-4 can be raised by more local recalls without fewer external calls. FA-5 covers numbers only.
- **Deliberate spec error in AB-4.** The project document gives the formula as memory hits multiplied by total API calls. That is not a ratio, grows as the agent makes more calls, and cannot meet a target of 0.3 in a meaningful way. We implemented memory hits divided by external API calls and recorded this as Error 1 in `ERROR_LOG.md`. AB-4 scores here are for the corrected formula and are not comparable with a score from the spec's formula.
- **Small benchmark.** BM-1 has gold facts for three challenges only. The memory-retrieval probe set (48 probes) is small and lexically close to its chunks.
- **Stress tests in mock mode.** They test code paths and concurrency, not live provider behaviour.
- **Optimisation effects are entangled.** Several Day 13 changes touched the same runs, so the share of any gain due to one change cannot be separated. Mock token figures are exact but come from a scripted model.
- **Live data.** Live sources (SEC EDGAR, Yahoo Finance, news) change over time, so a re-run will not give identical figures.

## 8. Reproduce

```text
python -m evaluation.runner --data live --out-dir results/runs_day13/live --report-dir results/runs_day13/live/reports
python -m evaluation.evaluate --runs results/runs_day13/live --out results/runs_day13/eval/scores.json
python -m evaluation.token_compare table results/runs_day13/baseline_mock results/runs_day13/after_mock
python -m evaluation.stress
python -m agent.trace_viewer traces/session-20261001-211753-ef30.jsonl --types decision,error
```

Live runs depend on provider quotas and will not match the numbers above exactly.
