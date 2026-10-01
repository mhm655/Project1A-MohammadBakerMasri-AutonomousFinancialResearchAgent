# ARA-1 Evaluation Report

Generated: 2026-10-01T18:23:20+00:00  
Data mode: live  
LLM judge: enabled

Reproduce: `python -m evaluation.runner`, then `python -m evaluation.evaluate`.

## Summary

Overall pass rate: **53%** (23 metrics x 8 challenges). Total LLM tokens: 257,229.

| Category | Passed | Measured | Rate |
|---|---|---|---|
| Factual Accuracy | 31 | 40 | 78% |
| Completeness | 18 | 28 | 64% |
| Analytical Depth | 6 | 32 | 19% |
| Coherence and Structure | 15 | 32 | 47% |
| Agent Behaviour | 23 | 39 | 59% |
| Benchmark | 0 | 3 | 0% |

## Metric Matrix

Legend: ✓ meets target, ✗ misses target, — not measured.

### Factual Accuracy

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| FA-1 | Numerical Accuracy Rate | >= 98% | 100% ✓ | 100% ✓ | 100% ✓ | 92% ✗ | 100% ✓ | 96% ✗ | 100% ✓ | 100% ✓ |
| FA-2 | Citation Accuracy | 100% | 100% ✓ | 100% ✓ | 86% ✗ | 67% ✗ | 100% ✓ | 93% ✗ | 86% ✗ | 100% ✓ |
| FA-3 | Temporal Accuracy | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ |
| FA-4 | Entity Accuracy | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 85% ✗ | 100% ✓ | 100% ✓ |
| FA-5 | Hallucination Rate | 0 | 0 ✓ | 0 ✓ | 0 ✓ | 1 ✗ | 0 ✓ | 1 ✗ | 0 ✓ | 0 ✓ |

### Completeness

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| CO-1 | Section Coverage | 100% | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ | 100% ✓ |
| CO-2 | Data Source Diversity | >= 4 source types | 3 ✗ | 3 ✗ | 4 ✓ | 5 ✓ | 4 ✓ | 3 ✗ | 2 ✗ | 4 ✓ |
| CO-3 | Temporal Coverage | >= 3 years | 3 ✓ | 2 ✗ | 3 ✓ | 4 ✓ | 4 ✓ | 3 ✓ | 2 ✗ | 4 ✓ |
| CO-4 | Risk Factor Coverage | >= 80% | — | — | 78% ✗ | 27% ✗ | 27% ✗ | — | 64% ✗ | — |

### Analytical Depth

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| AD-1 | Insight Density | >= 3 per page | 2.3/pg ✗ | 2.2/pg ✗ | 1.9/pg ✗ | 0.7/pg ✗ | 1.2/pg ✗ | 0.7/pg ✗ | 1.6/pg ✗ | 1.7/pg ✗ |
| AD-2 | Cross-Source Synthesis | >= 5 | 2 ✗ | 1 ✗ | 7 ✓ | 5 ✓ | 8 ✓ | 3 ✗ | 9 ✓ | 1 ✗ |
| AD-3 | Quantitative Reasoning | >= 10 original calculations | 2 ✗ | 2 ✗ | 6 ✗ | 1 ✗ | 3 ✗ | 5 ✗ | 20 ✓ | 5 ✗ |
| AD-4 | Forward-Looking Analysis | >= 2 sections | 0 ✗ | 1 ✗ | 0 ✗ | 0 ✗ | 1 ✗ | 0 ✗ | 0 ✗ | 2 ✓ |

### Coherence and Structure

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| CS-1 | Logical Flow | >= 4 / 5 | 4/5 ✓ | 4/5 ✓ | 4/5 ✓ | 3/5 ✗ | 4/5 ✓ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ |
| CS-2 | Internal Consistency | 0 contradictions | 0 ✓ | 0 ✓ | 1 ✗ | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ | 1 ✗ |
| CS-3 | Executive Summary Quality | >= 4 / 5 | 3/5 ✗ | 4/5 ✓ | 3/5 ✗ | 4/5 ✓ | 4/5 ✓ | 3/5 ✗ | 3/5 ✗ | 3/5 ✗ |
| CS-4 | Professional Formatting | >= 80% of checks | 100% ✓ | 71% ✗ | 57% ✗ | 71% ✗ | 71% ✗ | 86% ✓ | 71% ✗ | 71% ✗ |

### Agent Behaviour

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| AB-1 | Tool Efficiency | >= 70% | 100% ✓ | 83% ✓ | 70% ✓ | 59% ✗ | 56% ✗ | 82% ✓ | 47% ✗ | 62% ✗ |
| AB-2 | Error Recovery Rate | >= 90% | — | 100% ✓ | 100% ✓ | 0% ✗ | 0% ✗ | 100% ✓ | 100% ✓ | 100% ✓ |
| AB-3 | Planning Quality | >= 4 / 5 | 4/5 ✓ | 4/5 ✓ | 5/5 ✓ | 3/5 ✗ | 5/5 ✓ | 2/5 ✗ | 2/5 ✗ | 4/5 ✓ |
| AB-4 | Memory Utilization | >= 0.3 | 17% ✗ | 0% ✗ | 17% ✗ | 21% ✗ | 14% ✗ | 0% ✗ | 150% ✓ | 9% ✗ |
| AB-5 | Latency | < 300 s | 44 s ✓ | 17 s ✓ | 76 s ✓ | 220 s ✓ | 138 s ✓ | 84 s ✓ | 246 s ✓ | 54 s ✓ |

### Benchmark

| ID | Metric | Target | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| BM-1 | Benchmark Fact Recall | >= 70% | 62% ✗ | 36% ✗ | 69% ✗ | — | — | — | — | — |

## Per-Challenge Results

### C1 — Single-Company Profile

Query: Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, key executives, and recent developments.

Run: template `company_profile`, stop: completed · 44 s · 7 tool calls · 4 LLM calls · 13,150 tokens

Failing metrics:
- ✗ **CO-2** Data Source Diversity: 3 (target >= 4 source types) — 3 distinct source types cited: financial data, industry/peer data, news/web
- ✗ **AD-1** Insight Density: 2.3/pg (target >= 3 per page) — 5 insights over 2.2 pages (~500 words each)
- ✗ **AD-2** Cross-Source Synthesis: 2 (target >= 5) — 2 claims cite evidence from two or more different tools
- ✗ **AD-3** Quantitative Reasoning: 2 (target >= 10 original calculations) — 2 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue growth, margin improvement, and cash generation, but omits balance-sheet strength, detailed capital allocation, and the executive team, which are notable findings in th... Automated: 5 of 5 numbers in the executive summary also appear in the body
- ✗ **AB-4** Memory Utilization: 17% (target >= 0.3) — 1 memory hits (1 recalls with results + 0 cache hits) / 6 external API calls
- ✗ **BM-1** Benchmark Fact Recall: 62% (target >= 70%) — 8 of 13 gold facts recalled; 0 conflicting (missed: msft-opinc-fy26, msft-ni-fy26, msft-eps-fy26, msft-risk-competition, msft-risk-cyber)

### C2 — Earnings Analysis

Query: Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates and identify key takeaways from the earnings call.

Run: template `earnings_analysis`, stop: completed · 17 s · 6 tool calls · 4 LLM calls · 11,183 tokens

Failing metrics:
- ✗ **CO-2** Data Source Diversity: 3 (target >= 4 source types) — 3 distinct source types cited: earnings calls, financial data, news/web
- ✗ **CO-3** Temporal Coverage: 2 (target >= 3 years) — years 2025-2026 appear in numeric claims/tables
- ✗ **AD-1** Insight Density: 2.2/pg (target >= 3 per page) — 3 insights over 1.4 pages (~500 words each)
- ✗ **AD-2** Cross-Source Synthesis: 1 (target >= 5) — 1 claims cite evidence from two or more different tools
- ✗ **AD-3** Quantitative Reasoning: 2 (target >= 10 original calculations) — 2 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 1 (target >= 2 sections) — 1 forward-looking sections: guidance_and_outlook
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (692 words, 6/6 paragraphs cited)
- ✗ **AB-4** Memory Utilization: 0% (target >= 0.3) — 0 memory hits (0 recalls with results + 0 cache hits) / 4 external API calls
- ✗ **BM-1** Benchmark Fact Recall: 36% (target >= 70%) — 5 of 14 gold facts recalled; 0 conflicting (missed: aapl-rev-growth-q3fy26, aapl-ni-q3fy26, aapl-rev-fy25, aapl-rev-growth-fy25, aapl-gm-fy25, aapl-opinc-fy25, aapl-ni-fy25, aapl-eps-fy25, aapl-ceo)

### C3 — Risk Assessment

Query: Produce a comprehensive risk assessment for Tesla Inc. covering financial risks, operational risks, regulatory risks, and competitive risks.

Run: template `risk_assessment`, stop: completed · 76 s · 10 tool calls · 9 LLM calls · 29,107 tokens

Failing metrics:
- ✗ **FA-2** Citation Accuracy: 86% (target 100%) — 6 of 7 cited ids are real and support a claim; accessible = evidence record present with retrievable data (no URL fetching)
- ✗ **CO-4** Risk Factor Coverage: 78% (target >= 80%) — 7 of 9 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 1.9/pg (target >= 3 per page) — 5 insights over 2.7 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 6 (target >= 10 original calculations) — 6 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-2** Internal Consistency: 1 (target 0 contradictions) — 0 automated (same metric/period/entity differing >2%) + 1 from the judge
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue decline, margin compression, and primary risk themes, but omits the disclosed $30 bn borrowing plan and the valuation pressure risk highlighted later in the body. Automated: 3 of 7 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 57% (target >= 80% of checks) — 4 of 7 checks passed (1337 words, 5/10 paragraphs cited)
- ✗ **AB-4** Memory Utilization: 17% (target >= 0.3) — 1 memory hits (1 recalls with results + 0 cache hits) / 6 external API calls
- ✗ **BM-1** Benchmark Fact Recall: 69% (target >= 70%) — 9 of 13 gold facts recalled; 0 conflicting (missed: tsla-opinc-fy25, tsla-eps-fy25, tsla-cfo, tsla-ai)

### C4 — Industry Comparison

Query: Compare the cloud computing divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP). Analyze revenue growth, market share, margins, and competitive advantages.

Run: template `comparative_analysis`, stop: completed · 219 s · 17 tool calls · 15 LLM calls · 53,746 tokens

Failing metrics:
- ✗ **FA-1** Numerical Accuracy Rate: 92% (target >= 98%) — 12 of 13 numeric claims match their cited source (or, if uncited, any evidence)
- ✗ **FA-2** Citation Accuracy: 67% (target 100%) — 8 of 12 cited ids are real and support a claim; accessible = evidence record present with retrievable data (no URL fetching)
- ✗ **FA-5** Hallucination Rate: 1 (target 0) — 1 of 13 numeric claims can be neither found in nor derived (ratio or growth of two source amounts) from any evidence record
- ✗ **CO-4** Risk Factor Coverage: 27% (target >= 80%) — 3 of 11 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 0.7/pg (target >= 3 per page) — 2 insights over 2.9 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 1 (target >= 10 original calculations) — 1 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections are ordered, but conclusions often rely on unverified figures and the market-share gap is noted without analysis, reducing coherence.
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1437 words, 6/10 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 59% (target >= 70%) — 10 research evidence records cited in the body over 17 tool calls
- ✗ **AB-2** Error Recovery Rate: 0% (target >= 90%) — 0 of 3 error episodes recovered (same call later succeeded, arguments were repaired, or a fallback tool's evidence exists)
- ✗ **AB-3** Planning Quality: 3/5 (target >= 4 / 5) — Plan includes necessary filings and peer data but contains redundant financial_data_api calls and several skipped calculations without clear remediation. Automated: expected-tool coverage 5/6; 20 plan steps, 0 failed, 6 skipped
- ✗ **AB-4** Memory Utilization: 21% (target >= 0.3) — 3 memory hits (3 recalls with results + 0 cache hits) / 14 external API calls

### C5 — Contradictory Data Handling

Query: Research Palantir Technologies. Note: Recent news reports suggest the company is struggling, but their financial statements show strong growth. Investigate and explain the apparent contradiction.

Run: template `investigative`, stop: completed · 138 s · 9 tool calls · 11 LLM calls · 35,629 tokens

Failing metrics:
- ✗ **CO-4** Risk Factor Coverage: 27% (target >= 80%) — 3 of 11 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 1.2/pg (target >= 3 per page) — 3 insights over 2.6 pages (~500 words each)
- ✗ **AD-3** Quantitative Reasoning: 3 (target >= 10 original calculations) — 3 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 1 (target >= 2 sections) — 1 forward-looking sections: evidence_review
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1288 words, 9/13 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 56% (target >= 70%) — 5 research evidence records cited in the body over 9 tool calls
- ✗ **AB-2** Error Recovery Rate: 0% (target >= 90%) — 0 of 2 error episodes recovered (same call later succeeded, arguments were repaired, or a fallback tool's evidence exists)
- ✗ **AB-4** Memory Utilization: 14% (target >= 0.3) — 1 memory hits (1 recalls with results + 0 cache hits) / 7 external API calls

### C6 — Ambiguous Query Handling

Query: What's happening with the banks?

Run: template `company_profile`, stop: completed · 84 s · 17 tool calls · 8 LLM calls · 29,196 tokens

Failing metrics:
- ✗ **FA-1** Numerical Accuracy Rate: 96% (target >= 98%) — 23 of 24 numeric claims match their cited source (or, if uncited, any evidence)
- ✗ **FA-2** Citation Accuracy: 93% (target 100%) — 13 of 14 cited ids are real and support a claim; accessible = evidence record present with retrievable data (no URL fetching)
- ✗ **FA-4** Entity Accuracy: 85% (target 100%) — 17 of 20 entities verified (6 tickers, 7 companies, 7 executive names)
- ✗ **FA-5** Hallucination Rate: 1 (target 0) — 1 of 24 numeric claims can be neither found in nor derived (ratio or growth of two source amounts) from any evidence record
- ✗ **CO-2** Data Source Diversity: 3 (target >= 4 source types) — 3 distinct source types cited: financial data, industry/peer data, news/web
- ✗ **AD-1** Insight Density: 0.7/pg (target >= 3 per page) — 2 insights over 3.0 pages (~500 words each)
- ✗ **AD-2** Cross-Source Synthesis: 3 (target >= 5) — 3 claims cite evidence from two or more different tools
- ✗ **AD-3** Quantitative Reasoning: 5 (target >= 10 original calculations) — 5 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections follow a reasonable order, but conclusions about margins and growth lack cited evidence, and some content is duplicated or vague, reducing coherence.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary reflects the body’s revenue and margin trends but omits the important observation that recent news coverage is sparse, limiting short-term catalyst analysis. Automated: 3 of 4 numbers in the executive summary also appear in the body
- ✗ **AB-3** Planning Quality: 2/5 (target >= 4 / 5) — The plan contains many redundant company_profile steps and skips essential web searches without justification, showing inefficiency and incomplete coverage o... Automated: expected-tool coverage 3/3; 18 plan steps, 0 failed, 3 skipped
- ✗ **AB-4** Memory Utilization: 0% (target >= 0.3) — 0 memory hits (0 recalls with results + 0 cache hits) / 14 external API calls

### C7 — Sector Analysis With Memory

Query: Based on the companies you've already researched, what themes emerge across the technology sector? Identify cross-cutting risks and opportunities.

Run: template `thematic`, stop: completed · 246 s · 15 tool calls · 23 LLM calls · 71,351 tokens

Failing metrics:
- ✗ **FA-2** Citation Accuracy: 86% (target 100%) — 6 of 7 cited ids are real and support a claim; accessible = evidence record present with retrievable data (no URL fetching)
- ✗ **CO-2** Data Source Diversity: 2 (target >= 4 source types) — 2 distinct source types cited: SEC filings, industry/peer data
- ✗ **CO-3** Temporal Coverage: 2 (target >= 3 years) — years 2025-2026 appear in numeric claims/tables
- ✗ **CO-4** Risk Factor Coverage: 64% (target >= 80%) — 7 of 11 risk categories in the 10-K are covered by the report
- ✗ **AD-1** Insight Density: 1.6/pg (target >= 3 per page) — 5 insights over 3.2 pages (~500 words each)
- ✗ **AD-4** Forward-Looking Analysis: 0 (target >= 2 sections) — 0 forward-looking sections: none
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections follow a logical order, but conclusions in the executive summary repeat points from the body and some risk statements lack direct citation, reducing coherence.
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue scale, margin differences, and growth themes, but omits key risks such as talent shortage and supply-chain dependency highlighted later in the report. Automated: 4 of 5 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1600 words, 3/5 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 47% (target >= 70%) — 7 research evidence records cited in the body over 15 tool calls
- ✗ **AB-3** Planning Quality: 2/5 (target >= 4 / 5) — The plan includes necessary filings, financials, news, and peer comparison, but contains many redundant earnings_transcript calls that were impossible withou... Automated: expected-tool coverage 1/2; 25 plan steps, 0 failed, 11 skipped

### C8 — Full Research Report With Degradation

Query: Produce a complete investment research report on NVIDIA Corporation. Note: The financial data API and SEC filing search tools are currently experiencing intermittent failures (simulate 50% failure rate).

Run: template `full_report`, stop: completed · 54 s · 8 tool calls · 4 LLM calls · 13,867 tokens · injected failures: financial_data_api=3, sec_filing_search=1

Failing metrics:
- ✗ **AD-1** Insight Density: 1.7/pg (target >= 3 per page) — 5 insights over 2.9 pages (~500 words each)
- ✗ **AD-2** Cross-Source Synthesis: 1 (target >= 5) — 1 claims cite evidence from two or more different tools
- ✗ **AD-3** Quantitative Reasoning: 5 (target >= 10 original calculations) — 5 distinct derived quantities (percentages, multiples, calculated table cells)
- ✗ **CS-1** Logical Flow: 3/5 (target >= 4 / 5) — Sections follow a typical structure, but conclusions sometimes lack supporting evidence and the report repeats contradictory statements, reducing coherence.
- ✗ **CS-2** Internal Consistency: 1 (target 0 contradictions) — 0 automated (same metric/period/entity differing >2%) + 1 from the judge
- ✗ **CS-3** Executive Summary Quality: 3/5 (target >= 4 / 5) — The summary captures revenue, margin, growth, and risk themes, but omits key valuation concerns, cash-flow figures, and the inconsistency in revenue-size ranking. Automated: 6 of 6 numbers in the executive summary also appear in the body
- ✗ **CS-4** Professional Formatting: 71% (target >= 80% of checks) — 5 of 7 checks passed (1435 words, 7/9 paragraphs cited)
- ✗ **AB-1** Tool Efficiency: 62% (target >= 70%) — 5 research evidence records cited in the body over 8 tool calls
- ✗ **AB-4** Memory Utilization: 9% (target >= 0.3) — 1 memory hits (0 recalls with results + 1 cache hits) / 11 external API calls

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

### Run conditions
- The eight challenges ran once each, on 2026-10-01, with live data: SEC EDGAR/XBRL, Yahoo Finance, Motley Fool transcripts, DuckDuckGo and Google News.
- The free Gemini quota for the day was already used up. 75 of 78 LLM calls therefore went to the Groq fallbacks (`gpt-oss-120b`, `qwen3.8-27b`), whose free tier holds `max_prompt_tokens` to 6,000.
- These are honest worst-case numbers for the free stack, not tuned results.
- The LLM judge (one call per report, cached) was `qwen3.8-27b`. That is the same model family that wrote some of the reports, so judged scores (AD-1, CS-1, CS-3, AB-3) may be slightly lenient. The judged contradictions are quoted in the per-challenge results; C3's ("gross margin 18.0%, down from 18.0%") is genuine.
- 23 metrics in total: the 22 in spec A5.2 plus BM-1, recall of the gold-standard facts in `evaluation/benchmarks/`.

### Strengths
- **Grounding of figures.**
  - Temporal accuracy (FA-3) is 100% in all 8 reports. Entity accuracy (FA-4) is 100% in 7 of 8, and numerical accuracy (FA-1) averages 98.5%.
  - FA-5 flags 2 numeric claims as untraceable, out of 141 numeric claims in eight reports.
  - Manual review of those two:
    - C4, MSFT operating-margin change: 155,237 / 331,839 = 46.8% against 128,528 / 281,724 = 45.6%, i.e. +1.2 pp. The figure is correct, but the claim cites the 10-K record (E9), not the income statement it was computed from.
    - C6, Goldman Sachs "+2.6 pp margin improvement": cannot be confirmed from the evidence.
  - Margins and growth rates computed from the cited statement rows (MSFT net margin 40.3%, TSLA net income −46.5%) are recognised as derived figures, not counted as hallucinations.
- **Structure and reliability.**
  - Every report has every template section (CO-1 = 100%).
  - Every run completed within its latency target. The slowest single-company run (C5) took 138 s against the 300 s target, and the slowest multi-company run (C7) took 246 s against 600 s.
  - C8 completed with 4 injected failures (3 on `financial_data_api`, 1 on `sec_filing_search`), all recovered (AB-2 = 100%), and the gaps are disclosed.
- **Synthesis across sources.**
  - C3, C4, C5 and C7 meet the cross-source synthesis target (AD-2 ≥ 5 claims citing two or more tools).
  - C5 explains the Palantir contradiction with 8 such claims.

### Weaknesses
1. **Analytical depth is the weakest category (19% of checks pass).**
   - Reports are 651–1,651 words against the spec's 3–20 pages.
   - Insight density (AD-1) is 0.7–2.3 per page against a target of 3.
   - Only C8 has two forward-looking sections (AD-4): five of the seven templates have no outlook section at all.
   - Original calculations (AD-3) fall short of 10 in seven reports.
   - Root causes:
     - the 6,000-token synthesis budget on Groq;
     - templates without outlook sections;
     - calculation steps that never ran (item 3).
2. **Invented names in C6.**
   - The banks report named "David M. Sullivan" as Goldman Sachs CEO and "Charlie Schwab" as Wells Fargo CEO. Neither name is in any source; the CEOs are David Solomon and Charles Scharf.
   - It also gave Bank of America's CEO the wrong middle initial: "Brian M." where the source has "Brian Thomas Moynihan".
   - FA-4 caught all three. The agent's own verifier did not, because it checked only numbers.
3. **Calculation inputs left as names.**
   - In six of eight runs (C2–C7) the planner wrote calculation inputs as names ("s2.revenue_FY2026"), 11 failed calculation calls in all.
   - C3's were repaired. C5's were then skipped as "not needed", because FY2026 is not reported yet.
   - C4's three calculations failed: they referenced segment revenue (AWS, Azure, GCP) that consolidated XBRL data does not contain.
   - The result is low AB-2 for C4 and C5 and missing derived figures.
4. **Benchmark recall is low (36–69%, target 70%).**
   - Reports skip core lines the gold summaries contain. C1 omits operating income, net income and EPS; C2 omits the FY2025 annual context.
   - No reported figure conflicts with a gold value.
5. **Formatting (CS-4) and memory use (AB-4).**
   - Six reports fail CS-4, mostly for a missing table or citations in fewer than 80% of paragraphs.
   - Memory utilisation is 0–0.21 in the company-specific runs: their companies had little stored memory. C7, which is built on memory, scores 1.5.

### What the evaluation changed
Before the scores were accepted, two review passes checked the metrics against these reports. They found and fixed 13 measurement errors, each covered by a test in `tests/test_evaluation_metrics.py` or `tests/test_evaluation_judge.py` that fails without the fix. The pass rate moved from 51% to 53%; no reported figure became "correct" by loosening a check.

**Lead review:**
- FA-5 counted correct derived percentages as hallucinations.
- CS-2 compared segment revenue with company revenue, and read "3M" (the company) as $3 million.
- AB-2 ignored calls recovered by corrected arguments.
- FA-4 missed names written with extra spaces or without a middle name.
- BM-1 reported a segment growth rate as a conflict.
- AB-3 ignored tools the agent runs outside the plan.

**Supervised debugging pass:**
- FA-1 and FA-5 now accept margins and growth rates computed from the cited statement rows, including point changes of a ratio.
- Derivation from figures in the report is limited to the claim's own paragraph. A whole section had let a dense peer table "explain" 62% of random percentages; within a paragraph the rate is 1.6% (4,250 trials).
- FA-3 no longer treats a cited calculator record as a temporal error.
- CO-3 counts only years within the 5-year horizon, so an incidental FY2014 mention no longer stretches C6 to 12 years.
- CS-2 and BM-1 no longer read "Microsoft's revenue" as a segment called "s".
- AB-5 classes C6 (several banks) as multi-company.
- FA-4 accepts familiar first names ("Jamie" for "James Dimon").

### Known limitations of the metrics
- CO-4 uses a 12-category keyword taxonomy, not a semantic comparison with Item 1A.
- AD-1, CS-1, CS-3 and AB-3 rest on a single judge call per report.
- FA-2's prose check is keyword overlap: a sentence is supported if 30% of its content words appear in the cited source.

### Priorities (carried into Day 12 and 13)
1. A verifier check for invented names. Done on Day 12.
2. Deterministic resolution of calculation inputs. Done on Day 12.
3. Synthesis budgets that depend on the provider, and outlook sections in every template.
4. Leaner execution prompts (see `results/token_usage_analysis.md`).
5. Re-running on Gemini when its quota allows.
