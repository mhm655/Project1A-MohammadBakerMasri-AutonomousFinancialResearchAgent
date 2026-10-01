# ARA-1 Evaluation Report

Generated: 2026-10-01T21:23:08+00:00  
Data mode: live  
LLM judge: enabled

Reproduce: `python -m evaluation.runner`, then `python -m evaluation.evaluate`.

## Summary

Overall pass rate: **61%** (23 metrics x 8 challenges). Total LLM tokens: 150,656.

| Category | Passed | Measured | Rate |
|---|---|---|---|
| Factual Accuracy | 31 | 40 | 78% |
| Completeness | 20 | 26 | 77% |
| Analytical Depth | 11 | 32 | 34% |
| Coherence and Structure | 11 | 32 | 34% |
| Agent Behaviour | 29 | 36 | 81% |
| Benchmark | 1 | 3 | 33% |

## Metric Matrix

Legend: ✓ meets target, ✗ misses target, — not measured.

### Factual Accuracy

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| FA-1 | Numerical Accuracy Rate | >= 98% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 96% ✗ | 88% ✗ | 100% ✓ | 100% ✓ |
| FA-2 | Citation Accuracy | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 83% ✗ | 100% ✓ | 100% ✓ |
| FA-3 | Temporal Accuracy | 100% | 92% ✗ | 83% ✗ | 100% ✓ | 89% ✗ | 100% ✓ | 88% ✗ | 100% ✓ | 100% ✓ |
| FA-4 | Entity Accuracy | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ |
| FA-5 | Hallucination Rate | 0 | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 1 ✗ | 2 ✗ | 0 ✓ | 0 ✓ |

### Completeness

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| CO-1 | Section Coverage | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ |
| CO-2 | Data Source Diversity | >= 4 source types | 4 ✓ | 4 ✓ | 4 ✓ | 4 ✓ | 7 ✓ | 3 ✗ | 4 ✓ | 3 ✗ |
| CO-3 | Temporal Coverage | >= 3 years | 3 ✓ | 2 ✗ | 3 ✓ | 3 ✓ | 4 ✓ | 4 ✓ | 3 ✓ | 2 ✗ |
| CO-4 | Risk Factor Coverage | >= 80% | — | — | 67% ✗ | 18% ✗ | — | — | — | — |

### Analytical Depth

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| AD-1 | Insight Density | >= 3 per page | 2.5/pg ✗ | 2.2/pg ✗ | 1.3/pg ✗ | 0.9/pg ✗ | 1.4/pg ✗ | 1.3/pg ✗ | 0.9/pg ✗ | 1.2/pg ✗ |
| AD-2 | Cross-Source Synthesis | >= 5 | 17 ✓ | 11 ✓ | 7 ✓ | 10 ✓ | 8 ✓ | 9 ✓ | 0 ✗ | 8 ✓ |
| AD-3 | Quantitative Reasoning | >= 10 original calculations | 5 ✗ | 4 ✗ | 16 ✓ | 1 ✗ | 3 ✗ | 7 ✗ | 2 ✗ | 27 ✓ |
| AD-4 | Forward-Looking Analysis | >= 2 sections | 0 ✗ | 2 ✓ | 0 ✗ | 0 ✗ | 1 ✗ | 1 ✗ | 1 ✗ | 2 ✓ |

### Coherence and Structure

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| CS-1 | Logical Flow | >= 4 / 5 | 4/5 ✓ | 3/5 ✗ | 4/5 ✓ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ |
| CS-2 | Internal Consistency | 0 contradictions | 0 ✓ | 2 ✗ | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 2 ✗ | 0 ✓ |
| CS-3 | Executive Summary Quality | >= 4 / 5 | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ |
| CS-4 | Professional Formatting | >= 80% of checks | 100% ✓ | 86% ✓ | 71% ✗ | 71% ✗ | 71% ✗ | 86% ✓ | 71% ✗ | 71% ✗ |

### Agent Behaviour

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| AB-1 | Tool Efficiency | >= 70% | 100% ✓ | 100% ✓ | 75% ✓ | 50% ✗ | 78% ✓ | 86% ✓ | 44% ✗ | 40% ✗ |
| AB-2 | Error Recovery Rate | >= 90% | — | — | 100% ✓ | 60% ✗ | 100% ✓ | — | — | 100% ✓ |
| AB-3 | Planning Quality | >= 4 / 5 | 4/5 ✓ | 4/5 ✓ | 5/5 ✓ | 3/5 ✗ | 4/5 ✓ | 4/5 ✓ | 2/5 ✗ | 4/5 ✓ |
| AB-4 | Memory Utilization | >= 0.3 | 75% ✓ | 40% ✓ | 43% ✓ | 75% ✓ | 60% ✓ | 0% ✗ | 138% ✓ | 50% ✓ |
| AB-5 | Latency | < 300 s | 35 s ✓ | 51 s ✓ | 140 s ✓ | 220 s ✓ | 85 s ✓ | 28 s ✓ | 122 s ✓ | 79 s ✓ |

### Benchmark

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM-1 | Benchmark Fact Recall | >= 70% | 85% ✓ | 29% ✗ | 69% ✗ | — | — | — | — | — |

## Per-Challenge Results

### C1 — Single-Company Profile

Query: Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, key executives, and recent developments.

Run: template `company_profile`, stop: completed · 35 s · 7 tool calls · 2 LLM calls · 11,384 tokens

Failing metrics:
- ✗ **FA-3** Temporal Accuracy: 92% (target 100%) — 11 of 12 period-bearing numeric claims use a period present in their cited evidence
- ✗ **AD-1** Insight Density: 2.5/pg (target >= 3 per page) — 5 insights over 2.0 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 5 (target >= 10 original calculations) — 5 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — Summary captures revenue growth, margin expansion, and peer ranking, but omits recent executive changes and the security update noted in the body. Automated: 6 of 6 numbers in the executive summary also appear in the body

### C2 — Earnings Analysis

Query: Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates and identify key takeaways from the earnings call.

Run: template `earnings_analysis`, stop: completed · 51 s · 6 tool calls · 3 LLM calls · 12,690 tokens

Failing metrics:
- ✗ **FA-3** Temporal Accuracy: 83% (target 100%) — 5 of 6 period-bearing numeric claims use a period present in their cited evidence
- ✗ **CO-3** Temporal Coverage: 2 (target >= 3 years) — years 2025-2026 appear in numeric claims/tables
- ✗ **AD-1** Insight Density: 2.2/pg (target >= 3 per page) — 5 insights over 2.2 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 4 (target >= 10 original calculations) — 4 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections are generally ordered, but the report repeats EPS commentary, lacks consensus numbers, and some conclusions (e.g., margin strength) are not fully supported by the evidence.
- ✗ **CS-2** Internal Consistency: 2 (target 0 contradictions) — 0 automated (same metric/period/entity differing >2%) + 2 from the judge
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue growth and guidance but omits key points such as the missing consensus estimates, the FX headwind, and the margin compression implied by guidance. Automated: 9 of 9 numbers in the executive summary also appear in the body
- ✗ **BM-1** Benchmark Fact Recall: 29% (target >= 70%) — 4 of 14 gold facts recalled; 0 conflicting (missed: aapl-gm-q3fy26, aapl-rev-fy25, aapl-rev-growth-fy25, aapl-gm-fy25, aapl-opinc-fy25, aapl-ni-fy25, aapl-eps-fy25, aapl-ceo, aapl-cfo, aapl-risk-supply)

### C3 — Risk Assessment

Query: Produce a comprehensive risk assessment for Tesla Inc. covering financial risks, operational risks, regulatory risks, and competitive risks.

Run: template `risk_assessment`, stop: completed · 140 s · 12 tool calls · 5 LLM calls · 16,432 tokens

Failing metrics:
- ✗ **CO-4** Risk Factor Coverage: 67% (target >= 80%) — 6 of 9 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 1.3/pg (target >= 3 per page) — 4 insights over 3.1 pages (~500 words each)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — Summary captures revenue and margin trends and mentions competitive pressure, but omits key operational and regulatory risk details such as supply-chain and credit dependence. Automated: 8 of 12 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1527 words, 9/15 paragraphs cited)
- ✗ **BM-1** Benchmark Fact Recall: 69% (target >= 70%) — 9 of 13 gold facts recalled; 0 conflicting (missed: tsla-eps-fy25, tsla-ceo, tsla-cfo, tsla-risk-demand)

### C4 — Industry Comparison

Query: Compare the cloud computing divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP). Analyze revenue growth, market share, margins, and competitive advantages.

Run: template `comparative_analysis`, stop: completed · 220 s · 18 tool calls · 8 LLM calls · 24,089 tokens

Failing metrics:
- ✗ **FA-3** Temporal Accuracy: 89% (target 100%) — 8 of 9 period-bearing numeric claims use a period present in their cited evidence
- ✗ **CO-4** Risk Factor Coverage: 18% (target >= 80%) — 2 of 11 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 0.9/pg (target >= 3 per page) — 3 insights over 3.4 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 1 (target >= 10 original calculations) — 1 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — The report follows a reasonable structure, but several conclusions rely on unverified or missing segment data, and the market-share section admits lack of evidence, weakening logical coherence.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures the main growth and margin points, but it omits the explicit acknowledgement that market-share data is unavailable and overstates confidence in unverified figures. Automated: 10 of 10 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1721 words, 7/9 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 50% (target >= 70%) — 9 research evidence records cited in the body over 18 tool calls
- ✗ **AB-2** Error Recovery Rate: 60% (target >= 90%) — 3 of 5 error episodes recovered (same call later succeeded, arguments were repaired, or a fallback tool's evidence exists)
- ✗ **AB-3** Planning Quality: 3/5 (target >= 4 / 5) — The plan includes most required sources (transcripts, 10-Ks, news sentiment) but contains redundant steps and does not resolve failed calculations, indicatin... Automated: expected-tool coverage 5/6; 17 plan steps, 0 failed, 4 skipped

### C5 — Contradictory Data Handling

Query: Research Palantir Technologies. Note: Recent news reports suggest the company is struggling, but their financial statements show strong growth. Investigate and explain the apparent contradiction.

Run: template `investigative`, stop: completed · 85 s · 9 tool calls · 5 LLM calls · 20,506 tokens

Failing metrics:
- ✗ **FA-1** Numerical Accuracy Rate: 96% (target >= 98%) — 24 of 25 numeric claims match their cited source (or, if uncited, any evidence)
- ✗ **FA-5** Hallucination Rate: 1 (target 0) — 1 of 25 numeric claims can be neither found in nor derived (ratio or growth of two source amounts) from any evidence record
- ✗ **AD-1** Insight Density: 1.4/pg (target >= 3 per page) — 4 insights over 2.8 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 3 (target >= 10 original calculations) — 3 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 1 (target >= 2 sections) — 1 forward-looking sections: implications
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections follow a logical order, but conclusions about valuation and margin sustainability are repeated and some claims lack direct evidence, reducing coherence.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures the growth vs. narrative conflict but omits the risk that margin expansion may be non-recurring and the extreme valuation expectations. Automated: 5 of 5 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1402 words, 7/7 paragraphs cited)

### C6 — Ambiguous Query Handling

Query: What's happening with the banks?

Run: template `company_profile`, stop: completed · 28 s · 14 tool calls · 2 LLM calls · 10,206 tokens

Failing metrics:
- ✗ **FA-1** Numerical Accuracy Rate: 88% (target >= 98%) — 14 of 16 numeric claims match their cited source (or, if uncited, any evidence)
- ✗ **FA-2** Citation Accuracy: 83% (target 100%) — 10 of 12 cited ids are real and support a claim; accessible = evidence record present with retrievable data (no URL fetching)
- ✗ **FA-3** Temporal Accuracy: 88% (target 100%) — 7 of 8 period-bearing numeric claims use a period present in their cited evidence
- ✗ **FA-5** Hallucination Rate: 2 (target 0) — 2 of 16 numeric claims can be neither found in nor derived (ratio or growth of two source amounts) from any evidence record
- ✗ **CO-2** Data Source Diversity: 3 (target >= 4 source types) — 3 distinct source types cited: financial data, industry/peer data, news/web
- ✗ **AD-1** Insight Density: 1.3/pg (target >= 3 per page) — 4 insights over 3.2 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 7 (target >= 10 original calculations) — 7 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 1 (target >= 2 sections) — 1 forward-looking sections: recent_developments
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections are ordered sensibly, but several conclusions rely on unverified figures and the analysis repeats points without clear evidence linkage.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures the main margin trends and Wells Fargo weakness, but omits revenue growth rates and the recent negative sentiment for Citigroup. Automated: 3 of 6 numbers in the executive summary also appear in the body
- ✗ **AB-4** Memory Utilization: 0% (target >= 0.3) — 0 memory hits (0 recalls with results + 0 cache hits) / 14 external API calls

### C7 — Sector Analysis With Memory

Query: Based on the companies you've already researched, what themes emerge across the technology sector? Identify cross-cutting risks and opportunities.

Run: template `thematic`, stop: completed · 122 s · 16 tool calls · 4 LLM calls · 28,447 tokens

Failing metrics:
- ✗ **AD-1** Insight Density: 0.9/pg (target >= 3 per page) — 5 insights over 5.8 pages (~500 words each)
- ✗ **AD-2** Cross-Source Synthesis: 0 (target >= 5) — 0 claims cite evidence from two or more different tools
- ✗ **AD-3** Quantitative Reasoning: 2 (target >= 10 original calculations) — 2 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 1 (target >= 2 sections) — 1 forward-looking sections: cross_cutting_risks
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections are generally ordered, but the executive summary repeats margin figures without context and the risk discussion repeats inferred points, leading to some unsupported conclusions.
- ✗ **CS-2** Internal Consistency: 2 (target 0 contradictions) — 2 automated (same metric/period/entity differing >2%) + 0 from the judge
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures the profit bifurcation and AI-driven capex theme, but omits key cross-cutting risks such as supply-chain exposure, talent competition, and regulatory pressure discussed later. Automated: 4 of 4 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (2908 words, 9/19 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 44% (target >= 70%) — 7 research evidence records cited in the body over 16 tool calls
- ✗ **AB-3** Planning Quality: 2/5 (target >= 4 / 5) — The plan repeats the same data-gathering steps multiple times, showing redundancy, and does not include news or peer-comparison steps that would enrich thema... Automated: expected-tool coverage 0/2; 16 plan steps, 0 failed, 0 skipped

### C8 — Full Research Report With Degradation

Query: Produce a complete investment research report on NVIDIA Corporation. Note: The financial data API and SEC filing search tools are currently experiencing intermittent failures (simulate 50% failure rate).

Run: template `full_report`, stop: completed · 79 s · 10 tool calls · 8 LLM calls · 26,902 tokens · injected failures: financial_data_api=3

Failing metrics:
- ✗ **CO-2** Data Source Diversity: 3 (target >= 4 source types) — 3 distinct source types cited: financial data, industry/peer data, news/web
- ✗ **CO-3** Temporal Coverage: 2 (target >= 3 years) — years 2025-2026 appear in numeric claims/tables
- ✗ **AD-1** Insight Density: 1.2/pg (target >= 3 per page) — 5 insights over 4.2 pages (~500 words each)
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections are generally ordered, but conclusions about outlook and valuation rely on unverified statements and repeat earlier points, reducing coherence.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue, margin, and sentiment but omits key risk factors, valuation scenarios, and the lack of forward guidance discussed in the body. Automated: 4 of 5 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (2102 words, 9/18 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 40% (target >= 70%) — 4 research evidence records cited in the body over 10 tool calls

## Method

- **FA-1** Numerical Accuracy Rate — method: automated; target: >= 98%
- **FA-2** Citation Accuracy — method: automated; target: 100%
- **FA-3** Temporal Accuracy — method: automated; target: 100%
- **FA-4** Entity Accuracy — method: automated; target: 100%
- **FA-5** Hallucination Rate — method: automated; target: 0
- **CO-1** Section Coverage — method: automated; target: 100%
- **CO-2** Data Source Diversity — method: automated; target: >= 4 source types
- **CO-3** Temporal Coverage — method: automated; target: >= 3 years
- **CO-4** Risk Factor Coverage — method: automated; target: >= 80%
- **AD-1** Insight Density — method: llm_judge; target: >= 3 per page
- **AD-2** Cross-Source Synthesis — method: automated; target: >= 5
- **AD-3** Quantitative Reasoning — method: automated; target: >= 10 original calculations
- **AD-4** Forward-Looking Analysis — method: automated; target: >= 2 sections
- **CS-1** Logical Flow — method: llm_judge; target: >= 4 / 5
- **CS-2** Internal Consistency — method: hybrid; target: 0 contradictions
- **CS-3** Executive Summary Quality — method: llm_judge; target: >= 4 / 5
- **CS-4** Professional Formatting — method: automated; target: >= 80% of checks
- **AB-1** Tool Efficiency — method: automated; target: >= 70%
- **AB-2** Error Recovery Rate — method: automated; target: >= 90%
- **AB-3** Planning Quality — method: hybrid; target: >= 4 / 5
- **AB-4** Memory Utilization — method: automated; target: >= 0.3
- **AB-5** Latency — method: automated; target: < 300 s
- **BM-1** Benchmark Fact Recall — method: benchmark; target: >= 70%

Note: AB-4 uses memory_hits ÷ external_api_calls; the spec's "multiplied by" is a documented spec error (see ERROR_LOG.md Error 1).

<!-- ANALYSIS -->
## Analysis

_To be written after review._
