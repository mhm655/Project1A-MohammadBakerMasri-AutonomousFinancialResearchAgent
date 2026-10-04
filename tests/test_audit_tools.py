"""Audit regression tests for the tool layer (tools/). Pure functions only; no network."""

from __future__ import annotations

import pytest

from tools.fact_checker import extract_quantities, fact_checker


def _values(text: str) -> list[float]:
    return [q.value for q in extract_quantities(text)]


def test_range_with_thousands_separators_keeps_the_unit_on_both_ends():
    # "$1,000-1,200 million" used to give [1000.0, 200_000_000.0]: the range regex could not read
    # "1,000" and the "000" inside it was mistaken for the number.
    assert _values("guidance of $1,000-1,200 million") == [1e9, 1.2e9]
    assert _values("from 1,000 to 1,200 million") == [1e9, 1.2e9]


def test_a_month_word_before_a_decimal_number_does_not_swallow_it():
    # "may 3.5 billion": the date pattern removed "may 3", leaving ".5 billion" (or nothing).
    assert _values("Revenue may 3.5 billion") == [3.5e9]
    assert _values("Oct 5 bn") == [5e9]


@pytest.mark.parametrize("minus", ["-", "−"])
def test_unicode_minus_sign_is_a_negative_sign(minus):
    # Typographic sources write negatives with U+2212; the claim and source must still agree.
    assert _values(f"operating margin of {minus}5.2%") == [pytest.approx(-0.052)]
    result = fact_checker("Operating margin was -5.2%", [f"operating margin {minus}5.2% in the year"])
    assert result["status"] == "supported"
