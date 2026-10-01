# ARA-1 Evaluation Report v2 (Day 13)

Generated: 2026-10-02  
Data mode: live (token tables: mock)  
LLM judge: enabled

Reproduce: `python -m evaluation.runner --data live --out-dir results/runs_day13/live --report-dir results/runs_day13/live/reports`, then `python -m evaluation.evaluate --runs results/runs_day13/live --out results/runs_day13/eval/scores.json --dashboard results/runs_day13/eval/dashboard.html --report results/runs_day13/eval/evaluation_report.md`. Mock tokens: `python -m evaluation.token_compare`. Changes and reasons: `docs/optimization_log.md`.

## Summary

- Pass rate over 23 metrics x 8 challenges: **53% (Day 11) -> 61% (Day 13)**.
- Memory Utilization (AB-4) passes in 7 of 8 runs, up from 1 of 8 (mean 0.29 -> 0.60).
- Forward-Looking Analysis (AD-4): 7 sections in total, up from 4; 2 of 8 reports meet the target of 2, up from 1.
- Tool Efficiency (AB-1) mean 0.70 -> 0.72; 5 of 8 runs pass, up from 4.
- Total live LLM tokens 257,229 -> 150,656 and LLM calls 78 -> 37. Mock tokens for the same eight challenges: 132,856 -> 92,055 (-30.7%).
- Regressions: Temporal Accuracy (FA-3) 8/8 -> 4/8 and Executive Summary Quality (CS-3) 3/8 -> 0/8. Both are probably linked to the new forward-looking text (see below).

## Providers (this confounds the comparison)

The Day 11 runs used Groq for 75 of 78 calls (the Gemini daily quota was spent). The Day 13 runs used Gemini for 6 of 37 calls (C1-C3) and Groq for the rest. Gemini (`gemini-3.6-flash`) is stronger than Groq's `gpt-oss-120b` and `qwen3.8-27b`, so part of any quality difference in C1-C3 may come from the provider, not from the changes. Groq column: `gpt-oss-120b`; Q: Qwen; G: Gemini.

| Challenge | Day 11 providers (calls) | Day 13 providers (calls) |
|---|---|---|
| C1 | G 2, Gr 2 | G 1, Gr 1 |
| C2 | Gr 3, Q 1 | G 3 |
| C3 | Gr 5, Q 4 | G 2, Gr 3 |
| C4 | Gr 9, Q 6 | Gr 5, Q 3 |
| C5 | Gr 5, Q 6 | Gr 3, Q 2 |
| C6 | Gr 5, Q 3 | Gr 2 |
| C7 | Gr 17, Q 6 | Gr 2, Q 2 |
| C8 | G 1, Gr 3 | Gr 6, Q 2 |

## Per challenge: Day 11 -> Day 13

`x` marks a value that misses its target. Values are `Day 11 -> Day 13`.

| Challenge | Passed | Tokens | LLM calls | Duration (s) | AB-1 | AB-3 | AB-4 | AD-4 | AD-2 | AD-3 | FA-1 | FA-2 | FA-3 | CO-2 | CS-1 | CS-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 13/21 -> 16/21 | 13,150 -> 11,384 | 4 -> 2 | 44 -> 35 | 1.00 -> 1.00 | 4 -> 4 | 0.17 x -> 0.75 | 0 x -> 0 x | 2 x -> 17 | 2 x -> 5 x | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.00 -> 0.92 x | 3 x -> 4 | 4 -> 4 | 3 x -> 3 x |
| C2 | 13/22 -> 13/21 | 11,183 -> 12,690 | 4 -> 3 | 17 -> 51 | 0.83 -> 1.00 | 4 -> 4 | 0.00 x -> 0.40 | 1 x -> 2 | 1 x -> 11 | 2 x -> 4 x | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.00 -> 0.83 x | 3 x -> 4 | 4 -> 3 x | 4 -> 3 x |
| C3 | 13/23 -> 17/23 | 29,107 -> 16,432 | 9 -> 5 | 76 -> 140 | 0.70 -> 0.75 | 5 -> 5 | 0.17 x -> 0.43 | 0 x -> 0 x | 7 -> 7 | 6 x -> 16 | 1.00 -> 1.00 | 0.86 x -> 1.00 | 1.00 -> 1.00 | 4 -> 4 | 4 -> 4 | 3 x -> 3 x |
| C4 | 9/22 -> 11/22 | 53,746 -> 24,089 | 15 -> 8 | 219 -> 220 | 0.59 x -> 0.50 x | 3 x -> 3 x | 0.21 x -> 0.75 | 0 x -> 0 x | 5 -> 10 | 1 x -> 1 x | 0.92 x -> 1.00 | 0.67 x -> 1.00 | 1.00 -> 0.89 x | 5 -> 4 | 3 x -> 3 x | 4 -> 3 x |
| C5 | 14/22 -> 13/21 | 35,629 -> 20,506 | 11 -> 5 | 138 -> 85 | 0.56 x -> 0.78 | 5 -> 4 | 0.14 x -> 0.60 | 1 x -> 1 x | 8 -> 8 | 3 x -> 3 x | 1.00 -> 0.96 x | 1.00 -> 1.00 | 1.00 -> 1.00 | 4 -> 7 | 4 -> 3 x | 4 -> 3 x |
| C6 | 8/21 -> 9/20 | 29,196 -> 10,206 | 8 -> 2 | 84 -> 28 | 0.82 -> 0.86 | 2 x -> 4 | 0.00 x -> 0.00 x | 0 x -> 1 x | 3 x -> 9 | 5 x -> 7 x | 0.96 x -> 0.88 x | 0.93 x -> 0.83 x | 1.00 -> 0.88 x | 3 x -> 3 x | 3 x -> 3 x | 3 x -> 3 x |
| C7 | 11/22 -> 10/20 | 71,351 -> 28,447 | 23 -> 4 | 246 -> 122 | 0.47 x -> 0.44 x | 2 x -> 2 x | 1.50 -> 1.38 | 0 x -> 1 x | 9 -> 0 x | 20 -> 2 x | 1.00 -> 1.00 | 0.86 x -> 1.00 | 1.00 -> 1.00 | 2 x -> 4 | 3 x -> 3 x | 3 x -> 3 x |
| C8 | 12/21 -> 14/21 | 13,867 -> 26,902 | 4 -> 8 | 54 -> 79 | 0.62 x -> 0.40 x | 4 -> 4 | 0.09 x -> 0.50 | 2 -> 2 | 1 x -> 8 | 5 x -> 27 | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.00 -> 1.00 | 4 -> 3 x | 3 x -> 3 x | 3 x -> 3 x |

Metric names: AB-1 Tool Efficiency; AB-3 Planning Quality; AB-4 Memory Utilization; AD-4 Forward-Looking Analysis; AD-2 Cross-Source Synthesis; AD-3 Quantitative Reasoning; FA-1 Numerical Accuracy Rate; FA-2 Citation Accuracy; FA-3 Temporal Accuracy; CO-2 Data Source Diversity; CS-1 Logical Flow; CS-3 Executive Summary Quality.

## Aggregate per metric

| Metric | Day 11 passed | Day 13 passed | Day 11 mean | Day 13 mean |
|---|---|---|---|---|
| FA-1 Numerical Accuracy Rate | 6/8 | 6/8 | 0.99 | 0.98 |
| FA-2 Citation Accuracy | 4/8 | 7/8 | 0.91 | 0.98 |
| FA-3 Temporal Accuracy | 8/8 | 4/8 | 1.00 | 0.94 |
| FA-4 Entity Accuracy | 7/8 | 8/8 | 0.98 | 1.00 |
| FA-5 Hallucination Rate | 6/8 | 6/8 | 0.25 | 0.38 |
| CO-1 Section Coverage | 8/8 | 8/8 | 1.00 | 1.00 |
| CO-2 Data Source Diversity | 4/8 | 6/8 | 3.50 | 4.12 |
| CO-3 Temporal Coverage | 6/8 | 6/8 | 3.12 | 3.00 |
| CO-4 Risk Factor Coverage | 0/4 | 0/2 | 0.49 | 0.42 |
| AD-1 Insight Density | 0/8 | 0/8 | 1.52 | 1.46 |
| AD-2 Cross-Source Synthesis | 4/8 | 7/8 | 4.50 | 8.75 |
| AD-3 Quantitative Reasoning | 1/8 | 2/8 | 5.50 | 8.12 |
| AD-4 Forward-Looking Analysis | 1/8 | 2/8 | 0.50 | 0.88 |
| CS-1 Logical Flow | 4/8 | 2/8 | 3.50 | 3.25 |
| CS-2 Internal Consistency | 6/8 | 6/8 | 0.25 | 0.50 |
| CS-3 Executive Summary Quality | 3/8 | 0/8 | 3.38 | 3.00 |
| CS-4 Professional Formatting | 2/8 | 3/8 | 0.75 | 0.79 |
| AB-1 Tool Efficiency | 4/8 | 5/8 | 0.70 | 0.72 |
| AB-2 Error Recovery Rate | 5/7 | 3/4 | 0.71 | 0.90 |
| AB-3 Planning Quality | 5/8 | 6/8 | 3.62 | 3.75 |
| AB-4 Memory Utilization | 1/8 | 7/8 | 0.29 | 0.60 |
| AB-5 Latency | 8/8 | 8/8 | 109.60 | 94.91 |
| BM-1 Benchmark Fact Recall | 0/3 | 1/3 | 0.55 | 0.61 |

## Mock token before and after

Mock data and a scripted LLM (see `docs/optimization_log.md`, "How the numbers were measured"). Before: code at the start of Day 13. After: all Day 13 changes. Numbers are deterministic. Prompt, completion and totals are tokens; the mock provider is unlisted in `prompt_tokens_by_provider`, so it uses the default cap.

### Tokens per challenge

| Challenge | Calls before | Calls after | Prompt before | Prompt after | Completion before | Completion after | Total before | Total after | Total change |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 5 | 4 | 11,310 | 7,423 | 288 | 284 | 11,598 | 7,707 | -33.5% |
| C2 | 6 | 5 | 12,785 | 9,027 | 216 | 212 | 13,001 | 9,239 | -28.9% |
| C3 | 6 | 5 | 12,788 | 9,036 | 216 | 212 | 13,004 | 9,248 | -28.9% |
| C4 | 10 | 9 | 29,518 | 19,213 | 580 | 576 | 30,098 | 19,789 | -34.3% |
| C5 | 6 | 5 | 13,748 | 9,988 | 216 | 212 | 13,964 | 10,200 | -27.0% |
| C6 | 9 | 8 | 27,796 | 17,116 | 637 | 633 | 28,433 | 17,749 | -37.6% |
| C7 | 4 | 3 | 8,916 | 8,063 | 113 | 109 | 9,029 | 8,172 | -9.5% |
| C8 | 6 | 5 | 13,513 | 9,739 | 216 | 212 | 13,729 | 9,951 | -27.5% |
| ALL | 52 | 44 | 130,374 | 89,605 | 2,482 | 2,450 | 132,856 | 92,055 | -30.7% |

### Calls and tokens per challenge and purpose (calls / prompt / completion)

| Challenge | Purpose | Calls before | Calls after | Prompt before | Prompt after | Completion before | Completion after |
|---|---|---|---|---|---|---|---|
| C1 | plan | 1 | 1 | 1,745 | 1,953 | 167 | 167 |
| C1 | execute | 2 | 2 | 6,213 | 3,379 | 22 | 22 |
| C1 | review | 1 | 0 | 1,392 | 0 | 4 | 0 |
| C1 | synthesize | 1 | 1 | 1,960 | 2,091 | 95 | 95 |
| C2 | plan | 1 | 1 | 1,752 | 1,959 | 167 | 167 |
| C2 | execute | 2 | 2 | 6,213 | 3,379 | 22 | 22 |
| C2 | review | 2 | 1 | 2,867 | 1,605 | 8 | 4 |
| C2 | synthesize | 1 | 1 | 1,953 | 2,084 | 19 | 19 |
| C3 | plan | 1 | 1 | 1,747 | 1,955 | 167 | 167 |
| C3 | execute | 2 | 2 | 6,213 | 3,379 | 22 | 22 |
| C3 | review | 2 | 1 | 2,866 | 1,609 | 8 | 4 |
| C3 | synthesize | 1 | 1 | 1,962 | 2,093 | 19 | 19 |
| C4 | plan | 1 | 1 | 1,921 | 2,129 | 487 | 487 |
| C4 | execute | 6 | 6 | 18,640 | 10,138 | 66 | 66 |
| C4 | review | 2 | 1 | 3,143 | 1,753 | 8 | 4 |
| C4 | synthesize | 1 | 1 | 5,814 | 5,193 | 19 | 19 |
| C5 | plan | 1 | 1 | 1,754 | 1,962 | 167 | 167 |
| C5 | execute | 2 | 2 | 6,213 | 3,379 | 22 | 22 |
| C5 | review | 2 | 1 | 2,865 | 1,600 | 8 | 4 |
| C5 | synthesize | 1 | 1 | 2,916 | 3,047 | 19 | 19 |
| C6 | plan | 1 | 1 | 1,722 | 1,930 | 472 | 472 |
| C6 | execute | 6 | 6 | 18,638 | 10,136 | 66 | 66 |
| C6 | review | 1 | 0 | 1,483 | 0 | 4 | 0 |
| C6 | synthesize | 1 | 1 | 5,953 | 5,050 | 95 | 95 |
| C7 | plan | 1 | 1 | 2,270 | 2,500 | 86 | 86 |
| C7 | review | 2 | 1 | 2,762 | 1,549 | 8 | 4 |
| C7 | synthesize | 1 | 1 | 3,884 | 4,014 | 19 | 19 |
| C8 | plan | 1 | 1 | 1,767 | 1,975 | 167 | 167 |
| C8 | execute | 2 | 2 | 6,213 | 3,379 | 22 | 22 |
| C8 | review | 2 | 1 | 2,922 | 1,644 | 8 | 4 |
| C8 | synthesize | 1 | 1 | 2,611 | 2,741 | 19 | 19 |
| ALL | plan | 8 | 8 | 14,678 | 16,363 | 1,880 | 1,880 |
| ALL | execute | 22 | 22 | 68,343 | 37,169 | 242 | 242 |
| ALL | review | 14 | 6 | 20,300 | 9,760 | 56 | 24 |
| ALL | synthesize | 8 | 8 | 27,053 | 26,313 | 304 | 304 |

## What improved

- **Memory Utilization (AB-4):** 0.29 -> 0.60, 7 of 8 passing. Cause: section-targeted recall (M1). The gain is in recalls with results; external calls did not fall. C6 stays at 0 because memory has nothing on the banks.
- **Tokens and calls (live):** 257,229 -> 150,656 tokens, 78 -> 37 LLM calls. Contributors: one tool schema per execute call (T1), skipped reviews (T3, T4), and fewer repair calls (Day 12 reference resolver). Provider mix also differs, so the live figure is indicative; the mock figure (-30.7%) isolates the Day 13 changes.
- **C7 (cross-memory):** all 16 planned steps done, none skipped (Day 11: 11 of 25 skipped); 71,351 -> 28,447 tokens; 23 -> 4 LLM calls (P2, tool descriptions).
- **Citation Accuracy (FA-2):** 4/8 -> 7/8. Cross-Source Synthesis (AD-2): 4/8 -> 7/8. Data Source Diversity (CO-2): 4/8 -> 6/8. These were not targets; they may reflect the provider mix and judge variance.
- **Forward-Looking Analysis (AD-4):** 4 -> 7 sections in total (P1). C2 reaches 2 (executive summary and guidance and outlook). Five of eight reports now have at least one such section (Day 11: three); six still miss the target of 2.
- **Planning Quality (AB-3):** 5 of 8 -> 6 of 8 passing. C6 rose from 2 to 4; C7 stays at 2; C5 fell from 5 to 4.
- **Mean latency (AB-5):** 110 s -> 95 s.

## What regressed, and why

- **Temporal Accuracy (FA-3): 8/8 -> 4/8.** The failing claims are mostly forward-looking or "as of" statements whose period is not in the cited evidence, for example guidance for "Q4 FY2026" and employee counts "as of FY2026". P1 asks for forward-looking text, so more such sentences exist. Not proven: the providers also changed.
- **Executive Summary Quality (CS-3): 3/8 -> 0/8** (mean 3.38 -> 3.00); **Logical Flow (CS-1): 4/8 -> 2/8.** The forward view was asked for in two sections, usually the executive summary. The judge may score longer summaries lower. Judge scores move by a point between runs of the same bundle, so these are small differences.
- **Tool Efficiency (AB-1)** fell in C4 (0.59 -> 0.50), C7 (0.47 -> 0.44) and C8 (0.62 -> 0.40). C8 had two calculation steps skipped and a review skipped for lack of a provider.
- **Numerical Accuracy (FA-1)** fell in C5 (1.00 -> 0.96) and C6 (0.96 -> 0.88). Hallucination Rate (FA-5) mean 0.25 -> 0.38. Both are within the range seen between providers; no cause was isolated.
- **Quantitative Reasoning (AD-3)** rose in the mean (5.5 -> 8.1) but fell to 2 in C7, where fewer calculations ran.

## Remaining limitations

- The comparison is one run per challenge on mixed providers. A clean comparison needs both runs on the same model.
- Insight Density (AD-1) passes in 0 of 8 runs and Risk Factor Coverage (CO-4) in 0 of 2; none of the Day 13 changes targeted them.
- AB-4 can be raised by extra local recalls. It is a measure of memory use, not of fewer external calls.
- Mock token numbers come from a scripted model with fixed behaviour. They show what is sent, not how a real model reacts to the shorter execute prompt.
- T1 has no live check of how often a model asks for a different tool than the planned one. In the eight live runs the only execute steps that ended without a tool call were calculations with no inputs.
- The 500-character chunks apply to new ingests only. The store still holds mostly 1,500-character chunks; the retrieval gain will grow as it is re-ingested.
- The probe set (48 probes) is small and lexically close to its chunks.
- Judge results were cached in `results/eval/judge_cache.json` (entries added for the new bundles; Day 11 entries unchanged).
