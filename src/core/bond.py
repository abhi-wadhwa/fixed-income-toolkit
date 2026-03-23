"""Bond pricing from a fitted yield curve.

Supports fixed-coupon bonds with configurable day count conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from src.core.day_count import Convention, DayCountConvention


@dataclass
class CashFlow:
    """A single cash flow."""

    time: float  # years from settlement
    amount: float


class Bond:
    """Abstract base for bond instruments."""

    def cash_flows(self) -> List[CashFlow]:
        raise NotImplementedError

    def price(self, discount_fn) -> float:
        """Price bond given a discount function D(t)."""
        return sum(cf.amount * discount_fn(cf.time) for cf in self.cash_flows())


class FixedCouponBond(Bond):
    """Fixed-rate coupon bond.

    Parameters
    ----------
    face : float
        Face (par) value.
    coupon_rate : float
        Annual coupon rate (e.g. 0.05 for 5 %).
    maturity : float
        Time to maturity in years.
    frequency : int
        Coupon payments per year (1 = annual, 2 = semi-annual, 4 = quarterly).
    day_count : Convention or str
        Day count convention.
    settlement : float
        Settlement time offset in years (default 0).
    """

    def __init__(
        self,
        face: float = 100.0,
        coupon_rate: float = 0.05,
        maturity: float = 5.0,
        frequency: int = 2,
        day_count: Convention | str = Convention.THIRTY_360,
        settlement: float = 0.0,
    ) -> None:
        self.face = face
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.frequency = frequency
        self.settlement = settlement

        if isinstance(day_count, str):
            day_count = Convention(day_count)
        self.day_count = day_count
        self._dc = DayCountConvention(day_count)

    # ------------------------------------------------------------------
    # Cash flows
    # ------------------------------------------------------------------

    def cash_flows(self) -> List[CashFlow]:
        """Generate coupon and principal cash flows."""
        coupon = self.face * self.coupon_rate / self.frequency
        n_periods = int(round(self.maturity * self.frequency))
        period_length = 1.0 / self.frequency

        flows: List[CashFlow] = []
        for i in range(1, n_periods + 1):
            t = i * period_length
            if t <= self.settlement:
                continue
            amount = coupon
            if i == n_periods:
                amount += self.face  # Principal at maturity
            flows.append(CashFlow(time=t, amount=amount))
        return flows

    def payment_times(self) -> np.ndarray:
        """Return array of payment times."""
        return np.array([cf.time for cf in self.cash_flows()])

    def payment_amounts(self) -> np.ndarray:
        """Return array of payment amounts."""
        return np.array([cf.amount for cf in self.cash_flows()])

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, discount_fn) -> float:
        """Price the bond using a discount function D(t).

        Parameters
        ----------
        discount_fn : callable
            Maps time (years) to discount factor.

        Returns
        -------
        float
            Clean price.
        """
        return sum(cf.amount * discount_fn(cf.time) for cf in self.cash_flows())

    def price_from_yield(self, ytm: float) -> float:
        """Price the bond from a flat yield to maturity.

        Parameters
        ----------
        ytm : float
            Yield to maturity (annualised, compounding = frequency).

        Returns
        -------
        float
            Clean price.
        """

        def disc(t: float) -> float:
            return (1.0 + ytm / self.frequency) ** (-t * self.frequency)

        return self.price(disc)

    def yield_to_maturity(self, market_price: float) -> float:
        """Solve for yield to maturity given a market price.

        Uses Brent's method on the interval [0, 1].
        """

        def objective(y: float) -> float:
            return self.price_from_yield(y) - market_price

        return brentq(objective, -0.05, 2.0, xtol=1e-12)

    def accrued_interest(self, settlement_fraction: float = 0.0) -> float:
        """Compute accrued interest.

        Parameters
        ----------
        settlement_fraction : float
            Fraction of the current coupon period elapsed.

        Returns
        -------
        float
            Accrued interest.
        """
        coupon = self.face * self.coupon_rate / self.frequency
        return coupon * settlement_fraction

    def dirty_price(self, clean_price: float, settlement_fraction: float = 0.0) -> float:
        """Return dirty (invoice) price."""
        return clean_price + self.accrued_interest(settlement_fraction)
