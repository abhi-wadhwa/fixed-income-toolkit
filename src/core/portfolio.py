"""Portfolio-level analytics for fixed-income instruments.

Supports:
    - Portfolio duration (weighted average)
    - Portfolio convexity
    - Portfolio DV01
    - Cash flow aggregation
    - Duration immunization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.bond import Bond, CashFlow, FixedCouponBond
from src.core.risk import RiskMetrics


@dataclass
class PortfolioPosition:
    """A position in a bond."""

    bond: Bond
    quantity: float  # number of bonds (can be fractional)
    label: str = ""


@dataclass
class PortfolioRiskReport:
    """Portfolio-level risk summary."""

    total_market_value: float
    portfolio_duration: float
    portfolio_modified_duration: float
    portfolio_convexity: float
    portfolio_dv01: float
    position_weights: Dict[str, float]
    position_durations: Dict[str, float]


class Portfolio:
    """Collection of bond positions with portfolio-level analytics.

    Parameters
    ----------
    positions : list of PortfolioPosition
        Bond positions in the portfolio.
    """

    def __init__(self, positions: Optional[List[PortfolioPosition]] = None) -> None:
        self.positions: List[PortfolioPosition] = positions or []

    def add_position(self, bond: Bond, quantity: float, label: str = "") -> None:
        """Add a bond position to the portfolio."""
        self.positions.append(PortfolioPosition(bond=bond, quantity=quantity, label=label))

    # ------------------------------------------------------------------
    # Market value
    # ------------------------------------------------------------------

    def total_market_value(self, discount_fn: Callable[[float], float]) -> float:
        """Total portfolio market value."""
        return sum(
            pos.quantity * pos.bond.price(discount_fn) for pos in self.positions
        )

    def position_values(
        self, discount_fn: Callable[[float], float]
    ) -> Dict[str, float]:
        """Market value per position."""
        return {
            pos.label or f"pos_{i}": pos.quantity * pos.bond.price(discount_fn)
            for i, pos in enumerate(self.positions)
        }

    # ------------------------------------------------------------------
    # Duration and risk
    # ------------------------------------------------------------------

    def portfolio_duration(
        self, discount_fn: Callable[[float], float]
    ) -> float:
        """Market-value-weighted Macaulay duration.

        D_portfolio = sum(w_i * D_i) where w_i = MV_i / MV_total.
        """
        total_mv = self.total_market_value(discount_fn)
        if total_mv == 0:
            return 0.0

        weighted_dur = 0.0
        for pos in self.positions:
            mv = pos.quantity * pos.bond.price(discount_fn)
            rm = RiskMetrics(pos.bond)
            dur = rm.macaulay_duration(discount_fn)
            weighted_dur += (mv / total_mv) * dur

        return weighted_dur

    def portfolio_modified_duration(
        self,
        discount_fn: Callable[[float], float],
        ytm_map: Dict[str, float],
    ) -> float:
        """Market-value-weighted modified duration.

        Parameters
        ----------
        discount_fn : callable
            Discount function.
        ytm_map : dict
            {position_label: yield_to_maturity} for each position.

        Returns
        -------
        float
            Portfolio modified duration.
        """
        total_mv = self.total_market_value(discount_fn)
        if total_mv == 0:
            return 0.0

        weighted_dur = 0.0
        for i, pos in enumerate(self.positions):
            label = pos.label or f"pos_{i}"
            mv = pos.quantity * pos.bond.price(discount_fn)
            ytm = ytm_map.get(label, 0.05)
            freq = getattr(pos.bond, "frequency", 2)
            rm = RiskMetrics(pos.bond)
            d_mod = rm.modified_duration(discount_fn, ytm, freq)
            weighted_dur += (mv / total_mv) * d_mod

        return weighted_dur

    def portfolio_convexity(
        self, discount_fn: Callable[[float], float]
    ) -> float:
        """Market-value-weighted convexity."""
        total_mv = self.total_market_value(discount_fn)
        if total_mv == 0:
            return 0.0

        weighted_conv = 0.0
        for pos in self.positions:
            mv = pos.quantity * pos.bond.price(discount_fn)
            rm = RiskMetrics(pos.bond)
            conv = rm.convexity(discount_fn)
            weighted_conv += (mv / total_mv) * conv

        return weighted_conv

    def portfolio_dv01(
        self,
        discount_fn: Callable[[float], float],
        ytm_map: Dict[str, float],
    ) -> float:
        """Sum of position-level DV01s.

        Parameters
        ----------
        discount_fn : callable
            Discount function.
        ytm_map : dict
            {position_label: yield_to_maturity}.

        Returns
        -------
        float
            Total portfolio DV01.
        """
        total_dv01 = 0.0
        for i, pos in enumerate(self.positions):
            label = pos.label or f"pos_{i}"
            ytm = ytm_map.get(label, 0.05)
            freq = getattr(pos.bond, "frequency", 2)
            rm = RiskMetrics(pos.bond)
            dv = rm.dv01(discount_fn, ytm, freq)
            total_dv01 += pos.quantity * dv
        return total_dv01

    # ------------------------------------------------------------------
    # Cash flow aggregation
    # ------------------------------------------------------------------

    def aggregate_cash_flows(self) -> List[CashFlow]:
        """Aggregate cash flows from all positions.

        Returns
        -------
        list of CashFlow
            Sorted by time, with amounts scaled by position quantity.
        """
        all_flows: Dict[float, float] = {}
        for pos in self.positions:
            for cf in pos.bond.cash_flows():
                t = round(cf.time, 6)
                all_flows[t] = all_flows.get(t, 0.0) + cf.amount * pos.quantity

        times = sorted(all_flows.keys())
        return [CashFlow(time=t, amount=all_flows[t]) for t in times]

    # ------------------------------------------------------------------
    # Immunization
    # ------------------------------------------------------------------

    def immunization_target(
        self,
        target_duration: float,
        discount_fn: Callable[[float], float],
    ) -> Dict[str, float]:
        """Compute position weights to match a target duration (2-bond case).

        Works when the portfolio has exactly 2 positions.  Solves for
        weights w such that:
            w1 * D1 + w2 * D2 = D_target
            w1 + w2 = 1

        Parameters
        ----------
        target_duration : float
            Desired portfolio Macaulay duration.
        discount_fn : callable
            Discount function.

        Returns
        -------
        dict
            {label: weight} for each position.

        Raises
        ------
        ValueError
            If there are not exactly 2 positions.
        """
        if len(self.positions) != 2:
            raise ValueError(
                "Immunization target requires exactly 2 positions. "
                f"Got {len(self.positions)}."
            )

        durations = []
        labels = []
        for i, pos in enumerate(self.positions):
            rm = RiskMetrics(pos.bond)
            d = rm.macaulay_duration(discount_fn)
            durations.append(d)
            labels.append(pos.label or f"pos_{i}")

        d1, d2 = durations
        if abs(d1 - d2) < 1e-10:
            raise ValueError("Cannot immunize: both bonds have the same duration.")

        # w1 * D1 + (1-w1) * D2 = D_target  =>  w1 = (D_target - D2) / (D1 - D2)
        w1 = (target_duration - d2) / (d1 - d2)
        w2 = 1.0 - w1

        return {labels[0]: w1, labels[1]: w2}

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def risk_report(
        self,
        discount_fn: Callable[[float], float],
        ytm_map: Optional[Dict[str, float]] = None,
    ) -> PortfolioRiskReport:
        """Compute portfolio-level risk report.

        Parameters
        ----------
        discount_fn : callable
            Discount function.
        ytm_map : dict, optional
            {label: ytm} per position.  Defaults to 5 % for all.

        Returns
        -------
        PortfolioRiskReport
        """
        if ytm_map is None:
            ytm_map = {}

        total_mv = self.total_market_value(discount_fn)
        p_dur = self.portfolio_duration(discount_fn)
        p_mod_dur = self.portfolio_modified_duration(discount_fn, ytm_map)
        p_conv = self.portfolio_convexity(discount_fn)
        p_dv01 = self.portfolio_dv01(discount_fn, ytm_map)

        weights = {}
        durations = {}
        for i, pos in enumerate(self.positions):
            label = pos.label or f"pos_{i}"
            mv = pos.quantity * pos.bond.price(discount_fn)
            weights[label] = mv / total_mv if total_mv != 0 else 0.0
            rm = RiskMetrics(pos.bond)
            durations[label] = rm.macaulay_duration(discount_fn)

        return PortfolioRiskReport(
            total_market_value=total_mv,
            portfolio_duration=p_dur,
            portfolio_modified_duration=p_mod_dur,
            portfolio_convexity=p_conv,
            portfolio_dv01=p_dv01,
            position_weights=weights,
            position_durations=durations,
        )
