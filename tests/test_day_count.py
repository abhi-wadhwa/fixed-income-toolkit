"""Tests for day count conventions.

Reference values verified against ISDA definitions and Bloomberg.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.core.day_count import Convention, DayCountConvention


# ------------------------------------------------------------------
# 30/360
# ------------------------------------------------------------------

class TestThirty360:
    def test_six_months(self) -> None:
        """Jan 15 to Jul 15 = 180 days = 0.5 years."""
        dc = DayCountConvention(Convention.THIRTY_360)
        d1 = date(2024, 1, 15)
        d2 = date(2024, 7, 15)
        assert dc.day_count(d1, d2) == 180
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 0.5)

    def test_full_year(self) -> None:
        """Jan 1 to Jan 1 next year = 360 days = 1.0."""
        dc = DayCountConvention(Convention.THIRTY_360)
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        assert dc.day_count(d1, d2) == 360
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 1.0)

    def test_month_end_adjustment(self) -> None:
        """Day 31 adjustment: if d1=31, set to 30."""
        dc = DayCountConvention(Convention.THIRTY_360)
        d1 = date(2024, 1, 31)
        d2 = date(2024, 4, 30)
        # 30 + 30 + 30 + 0 = 90 days with adjustment
        expected = 360 * 0 + 30 * 3 + 0
        assert dc.day_count(d1, d2) == expected

    def test_same_date(self) -> None:
        """Same date should be 0."""
        dc = DayCountConvention(Convention.THIRTY_360)
        d = date(2024, 6, 15)
        assert dc.day_count(d, d) == 0
        assert dc.year_fraction(d, d) == 0.0


# ------------------------------------------------------------------
# ACT/360
# ------------------------------------------------------------------

class TestAct360:
    def test_full_year(self) -> None:
        """365 actual days / 360."""
        dc = DayCountConvention(Convention.ACT_360)
        d1 = date(2023, 1, 1)  # non-leap year
        d2 = date(2024, 1, 1)
        assert dc.day_count(d1, d2) == 365
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 365 / 360)

    def test_leap_year(self) -> None:
        """2024 is a leap year: 366/360."""
        dc = DayCountConvention(Convention.ACT_360)
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        assert dc.day_count(d1, d2) == 366
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 366 / 360)

    def test_short_period(self) -> None:
        """30 actual days / 360 = 1/12."""
        dc = DayCountConvention(Convention.ACT_360)
        d1 = date(2024, 3, 1)
        d2 = date(2024, 3, 31)
        assert dc.day_count(d1, d2) == 30
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 30 / 360)


# ------------------------------------------------------------------
# ACT/365
# ------------------------------------------------------------------

class TestAct365:
    def test_full_year(self) -> None:
        """365/365 = 1.0 for non-leap year."""
        dc = DayCountConvention(Convention.ACT_365)
        d1 = date(2023, 1, 1)
        d2 = date(2024, 1, 1)
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 1.0)

    def test_leap_year(self) -> None:
        """366/365 for leap year."""
        dc = DayCountConvention(Convention.ACT_365)
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 366 / 365)

    def test_half_year(self) -> None:
        """~182 days / 365."""
        dc = DayCountConvention(Convention.ACT_365)
        d1 = date(2023, 1, 1)
        d2 = date(2023, 7, 2)  # 182 days
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 182 / 365)


# ------------------------------------------------------------------
# ACT/ACT ISDA
# ------------------------------------------------------------------

class TestActAct:
    def test_full_non_leap_year(self) -> None:
        """Full non-leap year = 1.0."""
        dc = DayCountConvention(Convention.ACT_ACT)
        d1 = date(2023, 1, 1)
        d2 = date(2024, 1, 1)
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 1.0, atol=1e-10)

    def test_full_leap_year(self) -> None:
        """Full leap year = 1.0."""
        dc = DayCountConvention(Convention.ACT_ACT)
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        np.testing.assert_allclose(dc.year_fraction(d1, d2), 1.0, atol=1e-10)

    def test_cross_year_boundary(self) -> None:
        """Period spanning year boundary."""
        dc = DayCountConvention(Convention.ACT_ACT)
        d1 = date(2023, 7, 1)
        d2 = date(2024, 7, 1)
        yf = dc.year_fraction(d1, d2)
        # Should be close to 1.0 but account for leap day
        assert 0.99 < yf < 1.01

    def test_same_date(self) -> None:
        dc = DayCountConvention(Convention.ACT_ACT)
        d = date(2024, 3, 15)
        assert dc.year_fraction(d, d) == 0.0


# ------------------------------------------------------------------
# String construction
# ------------------------------------------------------------------

class TestStringConstruction:
    def test_from_string(self) -> None:
        """Should accept string convention names."""
        dc = DayCountConvention("30/360")
        assert dc.convention == Convention.THIRTY_360

    def test_act_360_string(self) -> None:
        dc = DayCountConvention("ACT/360")
        assert dc.convention == Convention.ACT_360
