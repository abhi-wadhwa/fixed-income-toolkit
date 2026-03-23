"""Risk metrics for fixed-income instruments.

Computes:
    - Macaulay duration
    - Modified duration
    - Effective (option-adjusted) duration
    - Convexity
    - DV01 (dollar value of a basis point)
    - Key rate durations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from src.core.bond import Bond, CashFlow, FixedCouponBond


@dataclass
class RiskReport:
    """Container for all risk metrics of a bond."""

    price: float
    macaulay_duration: float
    modified_duration: float
    effective_duration: float
    convexity: float
    dv01: float
    key_rate_durations: Optional[Dict[float, float]] = None


class RiskMetrics:
    """Compute risk metrics for a bond.

    Parameters
    ----------
    bond : Bond
        The bond instrument.
    """

    def __init__(self, bond: Bond) -> None:
        self.bond = bond

    # ------------------------------------------------------------------
    # Duration measures
    # ------------------------------------------------------------------

    def macaulay_duration(self, discount_fn: Callable[[float], float]) -> float:
        """Macaulay duration: D = (1/P) * sum(t_i * CF_i * D(t_i)).

        Parameters
        ----------
        discount_fn : callable
            Discount function D(t).

        Returns
        -------
        float
            Macaulay duration in years.
        """
        flows = self.bond.cash_flows()
        price = sum(cf.amount * discount_fn(cf.time) for cf in flows)
        weighted = sum(cf.time * cf.amount * discount_fn(cf.time) for cf in flows)
        return weighted / price

    def modified_duration(
        self, discount_fn: Callable[[float], float], ytm: float, frequency: int = 2
    ) -> float:
        """Modified duration: D_mod = D_mac / (1 + y/n).

        Parameters
        ----------
        discount_fn : callable
            Discount function D(t).
        ytm : float
            Yield to maturity.
        frequency : int
            Compounding frequency.

        Returns
        -------
        float
            Modified duration.
        """
        d_mac = self.macaulay_duration(discount_fn)
        return d_mac / (1.0 + ytm / frequency)

    def modified_duration_from_yield(self, ytm: float) -> float:
        """Modified duration using a flat yield curve.

        Parameters
        ----------
        ytm : float
            Yield to maturity.

        Returns
        -------
        float
            Modified duration.
        """
        if not isinstance(self.bond, FixedCouponBond):
            raise TypeError("modified_duration_from_yield requires FixedCouponBond")

        freq = self.bond.frequency

        def disc(t: float) -> float:
            return (1.0 + ytm / freq) ** (-t * freq)

        d_mac = self.macaulay_duration(disc)
        return d_mac / (1.0 + ytm / freq)

    def effective_duration(
        self,
        discount_fn_up: Callable[[float], float],
        discount_fn_down: Callable[[float], float],
        shift_bps: float = 1.0,
    ) -> float:
        """Effective duration via central difference.

        D_eff = (P_down - P_up) / (2 * dy * P_0)

        Parameters
        ----------
        discount_fn_up : callable
            Discount function after upward rate shift.
        discount_fn_down : callable
            Discount function after downward rate shift.
        shift_bps : float
            Size of the shift in basis points.

        Returns
        -------
        float
            Effective duration.
        """
        dy = shift_bps / 10_000.0
        p_up = self.bond.price(discount_fn_up)
        p_down = self.bond.price(discount_fn_down)
        # Use midpoint price as base
        p_0 = (p_up + p_down) / 2.0
        return (p_down - p_up) / (2.0 * dy * p_0)

    def effective_duration_from_yield(self, ytm: float, shift_bps: float = 1.0) -> float:
        """Effective duration using yield-based pricing."""
        if not isinstance(self.bond, FixedCouponBond):
            raise TypeError("effective_duration_from_yield requires FixedCouponBond")

        freq = self.bond.frequency
        dy = shift_bps / 10_000.0

        p_up = self.bond.price_from_yield(ytm + dy)
        p_down = self.bond.price_from_yield(ytm - dy)
        p_0 = self.bond.price_from_yield(ytm)

        return (p_down - p_up) / (2.0 * dy * p_0)

    # ------------------------------------------------------------------
    # Convexity
    # ------------------------------------------------------------------

    def convexity(self, discount_fn: Callable[[float], float]) -> float:
        """Convexity: C = (1/P) * sum(t_i^2 * CF_i * D(t_i)).

        Note: This is a simplified first-order convexity measure appropriate
        for non-callable bonds.

        Parameters
        ----------
        discount_fn : callable
            Discount function.

        Returns
        -------
        float
            Convexity.
        """
        flows = self.bond.cash_flows()
        price = sum(cf.amount * discount_fn(cf.time) for cf in flows)
        weighted = sum(cf.time**2 * cf.amount * discount_fn(cf.time) for cf in flows)
        return weighted / price

    def convexity_from_yield(self, ytm: float) -> float:
        """Convexity from a flat yield to maturity using finite differences.

        C = (P_up + P_down - 2*P_0) / (P_0 * dy^2)
        """
        if not isinstance(self.bond, FixedCouponBond):
            raise TypeError("convexity_from_yield requires FixedCouponBond")

        dy = 0.0001  # 1 bp
        p_0 = self.bond.price_from_yield(ytm)
        p_up = self.bond.price_from_yield(ytm + dy)
        p_down = self.bond.price_from_yield(ytm - dy)

        return (p_up + p_down - 2.0 * p_0) / (p_0 * dy**2)

    # ------------------------------------------------------------------
    # DV01
    # ------------------------------------------------------------------

    def dv01(self, discount_fn: Callable[[float], float], ytm: float, frequency: int = 2) -> float:
        """Dollar value of a basis point: DV01 = D_mod * P * 0.0001.

        Parameters
        ----------
        discount_fn : callable
            Discount function.
        ytm : float
            Yield to maturity.
        frequency : int
            Compounding frequency.

        Returns
        -------
        float
            DV01 (absolute price change per 1 bp yield change).
        """
        price = self.bond.price(discount_fn)
        d_mod = self.modified_duration(discount_fn, ytm, frequency)
        return d_mod * price * 0.0001

    def dv01_from_yield(self, ytm: float) -> float:
        """DV01 using finite differences on yield-based pricing."""
        if not isinstance(self.bond, FixedCouponBond):
            raise TypeError("dv01_from_yield requires FixedCouponBond")

        dy = 0.0001
        p_up = self.bond.price_from_yield(ytm + dy)
        p_down = self.bond.price_from_yield(ytm - dy)
        return -(p_up - p_down) / 2.0

    # ------------------------------------------------------------------
    # Key rate durations
    # ------------------------------------------------------------------

    def key_rate_durations(
        self,
        curve_zero_rate_fn: Callable[[float], float],
        key_rates: List[float],
        shift_bps: float = 1.0,
    ) -> Dict[float, float]:
        """Key rate durations: sensitivity to shifts at specific maturities.

        Parameters
        ----------
        curve_zero_rate_fn : callable
            Function returning zero rate for maturity t.
        key_rates : list of float
            Key rate maturities.
        shift_bps : float
            Shift size in basis points.

        Returns
        -------
        dict
            {maturity: key_rate_duration}
        """
        dy = shift_bps / 10_000.0

        def make_discount_fn(zero_rate_fn):
            def df(t):
                return np.exp(-zero_rate_fn(t) * t)
            return df

        p_0 = self.bond.price(make_discount_fn(curve_zero_rate_fn))
        krd = {}

        for kr in key_rates:
            def shifted_zr(t, _kr=kr, _dy=dy):
                base = curve_zero_rate_fn(t)
                # Triangular bump: affects +-1y around the key rate
                width = 1.0
                dist = abs(t - _kr)
                if dist < width:
                    bump = _dy * (1.0 - dist / width)
                    return base + bump
                return base

            p_shifted = self.bond.price(make_discount_fn(shifted_zr))
            krd[kr] = -(p_shifted - p_0) / (dy * p_0)

        return krd

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        discount_fn: Callable[[float], float],
        ytm: float,
        frequency: int = 2,
        curve_zero_rate_fn: Optional[Callable[[float], float]] = None,
        key_rates: Optional[List[float]] = None,
        shift_bps: float = 1.0,
    ) -> RiskReport:
        """Compute all risk metrics.

        Parameters
        ----------
        discount_fn : callable
            Discount function.
        ytm : float
            Yield to maturity.
        frequency : int
            Compounding frequency.
        curve_zero_rate_fn : callable, optional
            Zero rate function for key rate durations.
        key_rates : list of float, optional
            Key rate maturities.
        shift_bps : float
            Shift for effective duration and key rate durations.

        Returns
        -------
        RiskReport
        """
        price = self.bond.price(discount_fn)
        d_mac = self.macaulay_duration(discount_fn)
        d_mod = self.modified_duration(discount_fn, ytm, frequency)

        # Effective duration via small parallel shift
        dy = shift_bps / 10_000.0

        def disc_up(t):
            z = -np.log(discount_fn(t)) / t if t > 0 else 0
            return np.exp(-(z + dy) * t)

        def disc_down(t):
            z = -np.log(discount_fn(t)) / t if t > 0 else 0
            return np.exp(-(z - dy) * t)

        d_eff = self.effective_duration(disc_up, disc_down, shift_bps)
        conv = self.convexity(discount_fn)
        dv = self.dv01(discount_fn, ytm, frequency)

        krd = None
        if curve_zero_rate_fn is not None and key_rates is not None:
            krd = self.key_rate_durations(curve_zero_rate_fn, key_rates, shift_bps)

        return RiskReport(
            price=price,
            macaulay_duration=d_mac,
            modified_duration=d_mod,
            effective_duration=d_eff,
            convexity=conv,
            dv01=dv,
            key_rate_durations=krd,
        )
