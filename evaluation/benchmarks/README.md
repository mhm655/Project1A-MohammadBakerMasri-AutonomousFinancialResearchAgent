# Gold benchmarks (metric BM-1)

`microsoft.md`, `apple.md`, `tesla.md` are gold-standard summaries for challenges C1, C2, C3.
`gold_facts.json` is the machine-checkable subset used by `evaluation/benchmark.py`.

**Status: drafted from SEC filings on 2026-10-01, pending human review and sign-off.**

## Method
- Financials: `tools.financial_api.live_financial_data_api` (SEC XBRL companyfacts), annual and, for Apple, quarterly. Growth and margins are computed from those figures.
- Business, segments and risk headings: `tools.sec_edgar.live_sec_filing_search` (latest 10-K, Items 1 and 1A).
- Executives and headcount: `tools.company_profile.live_company_profile`. These come from a market-data profile, not the filing; verify against the proxy statement.
- No figure was taken from memory. If a number is unavailable it is omitted.

## Review checklist
Spot-check each number against the filing, confirm executive names and titles, then remove the "pending human review" header.

Percent facts store percentage points (17.8 means 17.8%); USD facts are in whole dollars.
