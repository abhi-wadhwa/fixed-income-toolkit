"""Day count conventions for fixed income instruments.

Supported conventions:
    - 30/360 (Bond Basis / ISDA)
    - ACT/360
    - ACT/365 (Fixed)
    - ACT/ACT (ISDA)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Tuple


class Convention(Enum):
    """Enumeration of supported day count conventions."""

    THIRTY_360 = "30/360"
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"


class DayCountConvention:
    """Compute year fractions and day counts under various conventions.

    Examples
    --------
    >>> dc = DayCountConvention(Convention.THIRTY_360)
    >>> dc.year_fraction(date(2024, 1, 15), date(2024, 7, 15))
    0.5
    """

    def __init__(self, convention: Convention | str) -> None:
        if isinstance(convention, str):
            convention = Convention(convention)
        self.convention = convention

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def day_count(self, start: date, end: date) -> int:
        """Return the number of days between *start* and *end*."""
        if self.convention == Convention.THIRTY_360:
            return self._days_30_360(start, end)
        return (end - start).days

    def year_fraction(self, start: date, end: date) -> float:
        """Return the year fraction between *start* and *end*."""
        if self.convention == Convention.THIRTY_360:
            return self._days_30_360(start, end) / 360.0
        if self.convention == Convention.ACT_360:
            return (end - start).days / 360.0
        if self.convention == Convention.ACT_365:
            return (end - start).days / 365.0
        if self.convention == Convention.ACT_ACT:
            return self._act_act_isda(start, end)
        raise ValueError(f"Unknown convention: {self.convention}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_30_360(start: date, end: date) -> int:
        """ISDA 30/360 day count."""
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year

        # Adjust days per ISDA 30/360 rules
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    @staticmethod
    def _act_act_isda(start: date, end: date) -> float:
        """ACT/ACT ISDA year fraction."""
        if start == end:
            return 0.0

        total = 0.0
        current = start

        while current.year < end.year:
            year_end = date(current.year + 1, 1, 1)
            days_in_year = 366 if calendar.isleap(current.year) else 365
            total += (year_end - current).days / days_in_year
            current = year_end

        # Remaining fraction in the final year
        days_in_year = 366 if calendar.isleap(current.year) else 365
        total += (end - current).days / days_in_year
        return total
