"""Canned data used by the mock tool handlers.

These figures approximate the companies' reported results so that development runs look realistic,
but they are **mock data**: rounded, unaudited, and not to be cited. Every result produced from them
is flagged ``is_mock=True`` by the registry, and the report generator refuses to present mock
evidence as fact outside mock mode.

Monetary amounts are in USD millions unless noted; EPS is in USD per share (split-adjusted).
"""

from __future__ import annotations

MOCK_NOTICE = "MOCK DATA - approximate figures for development only; do not cite."

# --------------------------------------------------------------------------------------------------
# Company profiles
# --------------------------------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "MSFT": {
        "name": "Microsoft Corporation",
        "cik": "0000789019",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "sic_code": "7372",
        "headquarters": "Redmond, Washington, USA",
        "employees": 228000,
        "fiscal_year_end": "0630",
        "market_cap": 3.1e12,
        "description": (
            "Develops and licenses software, cloud services and devices. Reports three segments: "
            "Productivity and Business Processes, Intelligent Cloud (including Azure), and More "
            "Personal Computing."
        ),
        "executives": [
            {"name": "Satya Nadella", "title": "Chairman and Chief Executive Officer"},
            {"name": "Amy Hood", "title": "Executive Vice President and Chief Financial Officer"},
            {"name": "Brad Smith", "title": "Vice Chair and President"},
        ],
    },
    "AAPL": {
        "name": "Apple Inc.",
        "cik": "0000320193",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "sic_code": "3571",
        "headquarters": "Cupertino, California, USA",
        "employees": 164000,
        "fiscal_year_end": "0928",
        "market_cap": 3.4e12,
        "description": (
            "Designs and sells smartphones, personal computers, tablets, wearables and accessories, "
            "and a growing portfolio of services including the App Store, advertising, cloud, "
            "payments and subscriptions."
        ),
        "executives": [
            {"name": "Tim Cook", "title": "Chief Executive Officer"},
            {"name": "Kevan Parekh", "title": "Senior Vice President and Chief Financial Officer"},
        ],
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "cik": "0001318605",
        "exchange": "NASDAQ",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "sic_code": "3711",
        "headquarters": "Austin, Texas, USA",
        "employees": 125665,
        "fiscal_year_end": "1231",
        "market_cap": 1.0e12,
        "description": (
            "Designs, manufactures and sells electric vehicles and energy generation and storage "
            "systems, and offers related services including charging, insurance and software."
        ),
        "executives": [
            {"name": "Elon Musk", "title": "Chief Executive Officer"},
            {"name": "Vaibhav Taneja", "title": "Chief Financial Officer"},
        ],
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "cik": "0001045810",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Semiconductors",
        "sic_code": "3674",
        "headquarters": "Santa Clara, California, USA",
        "employees": 36000,
        "fiscal_year_end": "0126",
        "market_cap": 3.5e12,
        "description": (
            "Designs accelerated computing platforms: data-center GPUs, networking and software, "
            "plus gaming, professional visualisation and automotive products. Reports Compute & "
            "Networking and Graphics segments."
        ),
        "executives": [
            {"name": "Jensen Huang", "title": "President and Chief Executive Officer"},
            {"name": "Colette Kress", "title": "Executive Vice President and Chief Financial Officer"},
        ],
    },
}

# --------------------------------------------------------------------------------------------------
# Annual financial statements (most recent fiscal year first)
# --------------------------------------------------------------------------------------------------

_INCOME = ("revenue", "gross_profit", "operating_income", "net_income", "eps_diluted")
_BALANCE = ("total_assets", "total_liabilities", "shareholders_equity", "cash_and_equivalents")
_CASH = ("operating_cash_flow", "capital_expenditure")


def _year(fy: int, period_end: str, values: tuple) -> dict:
    income = dict(zip(_INCOME, values[0:5]))
    balance = dict(zip(_BALANCE, values[5:9]))
    cash = dict(zip(_CASH, values[9:11]))
    cash["free_cash_flow"] = cash["operating_cash_flow"] - cash["capital_expenditure"]
    return {
        "fiscal_year": fy,
        "fiscal_period": "FY",
        "period_end": period_end,
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cash,
    }


# values: revenue, gross profit, operating income, net income, diluted EPS,
#         total assets, total liabilities, equity, cash, operating cash flow, capex
ANNUAL: dict[str, list[dict]] = {
    "MSFT": [
        _year(2024, "2024-06-30", (245122, 171008, 109433, 88136, 11.80, 512163, 243686, 268477, 18315, 118548, 44477)),
        _year(2023, "2023-06-30", (211915, 146052, 88523, 72361, 9.68, 411976, 205753, 206223, 34704, 87582, 28107)),
        _year(2022, "2022-06-30", (198270, 135620, 83383, 72738, 9.65, 364840, 198298, 166542, 13931, 89035, 23886)),
    ],
    "AAPL": [
        _year(2024, "2024-09-28", (391035, 180683, 123216, 93736, 6.08, 364980, 308030, 56950, 29943, 118254, 9447)),
        _year(2023, "2023-09-30", (383285, 169148, 114301, 96995, 6.13, 352583, 290437, 62146, 29965, 110543, 10959)),
        _year(2022, "2022-09-24", (394328, 170782, 119437, 99803, 6.11, 352755, 302083, 50672, 23646, 122151, 10708)),
    ],
    "TSLA": [
        _year(2024, "2024-12-31", (97690, 17450, 7076, 7091, 2.04, 122070, 48390, 72913, 16139, 14923, 11339)),
        _year(2023, "2023-12-31", (96773, 17660, 8891, 14997, 4.30, 106618, 43009, 62634, 16398, 13256, 8898)),
        _year(2022, "2022-12-31", (81462, 20853, 13656, 12556, 3.62, 82338, 36440, 44704, 16253, 14724, 7158)),
    ],
    "NVDA": [
        _year(2025, "2025-01-26", (130497, 97858, 81453, 72880, 2.94, 111601, 32274, 79327, 8589, 64089, 3236)),
        _year(2024, "2024-01-28", (60922, 44301, 32972, 29760, 1.19, 65728, 22750, 42978, 7280, 28090, 1069)),
        _year(2023, "2023-01-29", (26974, 15356, 4224, 4368, 0.17, 41182, 19081, 22101, 3389, 5641, 1833)),
    ],
}

# Quarterly income statements (Challenge 2 needs Apple's most recent quarter).
QUARTERLY_INCOME: dict[str, list[dict]] = {
    "AAPL": [
        {"fiscal_year": 2024, "fiscal_period": "Q4", "period_end": "2024-09-28",
         "income_statement": dict(zip(_INCOME, (94930, 43879, 29591, 14736, 0.97)))},
        {"fiscal_year": 2024, "fiscal_period": "Q3", "period_end": "2024-06-29",
         "income_statement": dict(zip(_INCOME, (85777, 39678, 25352, 21448, 1.40)))},
        {"fiscal_year": 2024, "fiscal_period": "Q2", "period_end": "2024-03-30",
         "income_statement": dict(zip(_INCOME, (90753, 42271, 27900, 23636, 1.53)))},
        {"fiscal_year": 2024, "fiscal_period": "Q1", "period_end": "2023-12-30",
         "income_statement": dict(zip(_INCOME, (119575, 54855, 40373, 33916, 2.18)))},
    ],
}

# --------------------------------------------------------------------------------------------------
# Peer metrics (latest fiscal year; ratios as fractions; market cap in USD)
# --------------------------------------------------------------------------------------------------

PEER_MAP: dict[str, list[str]] = {
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "META"],
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "TSLA": ["GM", "F", "RIVN"],
    "NVDA": ["AMD", "AVGO", "QCOM", "INTC"],
}

# Tickers without full statements above get static metrics. pe_ratio is None when earnings are negative.
STATIC_METRICS: dict[str, dict] = {
    "GOOGL": {"name": "Alphabet Inc.", "revenue": 350018, "revenue_growth": 0.139, "gross_margin": 0.582,
              "operating_margin": 0.321, "net_margin": 0.286, "roe": 0.32, "pe_ratio": 23.0, "market_cap": 2.1e12},
    "AMZN": {"name": "Amazon.com, Inc.", "revenue": 637959, "revenue_growth": 0.110, "gross_margin": 0.488,
             "operating_margin": 0.108, "net_margin": 0.093, "roe": 0.24, "pe_ratio": 40.0, "market_cap": 2.2e12},
    "META": {"name": "Meta Platforms, Inc.", "revenue": 164501, "revenue_growth": 0.219, "gross_margin": 0.818,
             "operating_margin": 0.42, "net_margin": 0.38, "roe": 0.37, "pe_ratio": 27.0, "market_cap": 1.5e12},
    "ORCL": {"name": "Oracle Corporation", "revenue": 57399, "revenue_growth": 0.084, "gross_margin": 0.70,
             "operating_margin": 0.31, "net_margin": 0.22, "roe": 0.80, "pe_ratio": 40.0, "market_cap": 6.0e11},
    "GM": {"name": "General Motors Company", "revenue": 187442, "revenue_growth": 0.091, "gross_margin": 0.12,
           "operating_margin": 0.07, "net_margin": 0.032, "roe": 0.09, "pe_ratio": 6.0, "market_cap": 5.0e10},
    "F": {"name": "Ford Motor Company", "revenue": 184992, "revenue_growth": 0.05, "gross_margin": 0.08,
          "operating_margin": 0.03, "net_margin": 0.031, "roe": 0.13, "pe_ratio": 11.0, "market_cap": 4.0e10},
    "RIVN": {"name": "Rivian Automotive, Inc.", "revenue": 4970, "revenue_growth": 0.12, "gross_margin": -0.024,
             "operating_margin": -0.85, "net_margin": -0.95, "roe": -0.70, "pe_ratio": None, "market_cap": 1.3e10},
    "AMD": {"name": "Advanced Micro Devices, Inc.", "revenue": 25785, "revenue_growth": 0.137, "gross_margin": 0.49,
            "operating_margin": 0.07, "net_margin": 0.064, "roe": 0.03, "pe_ratio": 100.0, "market_cap": 2.5e11},
    "AVGO": {"name": "Broadcom Inc.", "revenue": 51574, "revenue_growth": 0.44, "gross_margin": 0.64,
             "operating_margin": 0.29, "net_margin": 0.11, "roe": 0.10, "pe_ratio": 90.0, "market_cap": 1.0e12},
    "QCOM": {"name": "QUALCOMM Incorporated", "revenue": 38962, "revenue_growth": 0.088, "gross_margin": 0.56,
             "operating_margin": 0.26, "net_margin": 0.26, "roe": 0.39, "pe_ratio": 17.0, "market_cap": 1.8e11},
    "INTC": {"name": "Intel Corporation", "revenue": 53101, "revenue_growth": -0.021, "gross_margin": 0.327,
             "operating_margin": -0.22, "net_margin": -0.354, "roe": -0.19, "pe_ratio": None, "market_cap": 9.0e10},
}

# --------------------------------------------------------------------------------------------------
# Filings (section summaries, paraphrased — not filing text)
# --------------------------------------------------------------------------------------------------

RISK_FACTORS: dict[str, list[str]] = {
    "TSLA": [
        "Production ramps of new vehicles and factories may face delays and cost overruns.",
        "Dependence on a limited number of suppliers for battery cells and other key components.",
        "Intensifying competition in electric vehicles, including from lower-cost manufacturers.",
        "Pricing actions to sustain demand may compress automotive gross margins.",
        "Regulatory scrutiny of Autopilot and Full Self-Driving features, including recalls.",
        "Changes to government incentives and regulatory credits for electric vehicles.",
        "Heavy dependence on the continued service of the Chief Executive Officer.",
        "Exposure to trade policy, tariffs and international operations, notably in China.",
        "Product liability and warranty claims could exceed reserves.",
        "Information-security incidents could disrupt operations or expose data.",
    ],
    "MSFT": [
        "Intense competition across cloud, productivity, gaming and devices.",
        "Heavy investment in AI infrastructure may not earn expected returns.",
        "Cybersecurity attacks on Microsoft or its customers could damage trust.",
        "Evolving regulation of AI, privacy and competition in multiple jurisdictions.",
        "Dependence on third-party hardware and datacenter capacity.",
    ],
    "AAPL": [
        "Revenue concentration in iPhone and exposure to consumer demand cycles.",
        "Supply-chain concentration, including manufacturing in China and India.",
        "Regulatory challenges to App Store practices and digital-markets legislation.",
        "Foreign-exchange exposure from a majority of revenue earned outside the US.",
        "Legal and regulatory risk around search-default and services agreements.",
    ],
    "NVDA": [
        "Export controls restricting sales of advanced computing products to certain countries.",
        "Customer concentration among a small number of hyperscale data-center buyers.",
        "Dependence on third-party foundries, notably for leading-edge process nodes.",
        "Demand volatility and inventory risk as product cycles accelerate.",
        "Competition from custom accelerators designed by large customers.",
    ],
}

# --------------------------------------------------------------------------------------------------
# Earnings-call transcripts (paraphrased summaries of speaker turns)
# --------------------------------------------------------------------------------------------------

TRANSCRIPTS: dict[tuple[str, str, int], dict] = {
    ("AAPL", "Q4", 2024): {
        "call_date": "2024-10-31",
        "segments": [
            {"speaker": "Suhasini Chandramouli", "role": "Director, Investor Relations", "section": "prepared",
             "text": "Welcome to the fiscal fourth-quarter call; remarks include forward-looking statements."},
            {"speaker": "Tim Cook", "role": "CEO", "section": "prepared",
             "text": "Reported a September-quarter revenue record, with growth in iPhone and an all-time high in Services."},
            {"speaker": "Luca Maestri", "role": "CFO", "section": "prepared",
             "text": "Net income included a one-time income-tax charge related to a European Commission state-aid decision."},
            {"speaker": "Analyst (Morgan Stanley)", "role": "Analyst", "section": "qa",
             "text": "Asked about the pace of iPhone upgrades tied to Apple Intelligence features."},
            {"speaker": "Tim Cook", "role": "CEO", "section": "qa",
             "text": "Said feature roll-out is staged by language and country and early adoption is encouraging."},
            {"speaker": "Analyst (Goldman Sachs)", "role": "Analyst", "section": "qa",
             "text": "Asked about gross-margin guidance for the December quarter."},
            {"speaker": "Luca Maestri", "role": "CFO", "section": "qa",
             "text": "Guided gross margin to a range broadly similar to the prior quarter, with foreign-exchange headwinds."},
        ],
    },
}
