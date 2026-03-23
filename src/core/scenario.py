"""Scenario analysis for fixed-income portfolios.

Supported scenarios:
    - Parallel shift: uniform shift across all maturities
    - Twist (steepening/flattening): short end and long end move differently
    - Butterfly: belly moves differently from wings
    - Custom: arbitrary shift vector
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.bond import Bond


class ShiftType(Enum):
    PARALLEL = "parallel"
    TWIST = "twist"
    BUTTERFLY = "butterfly"
    CUSTOM = "custom"


@dataclass
class ScenarioResult:
    """Result of a scenario analysis for a single bond or portfolio."""

    scenario_name: str
    shift_type: ShiftType
    shift_bps: float
    base_price: float
    scenario_price: float
    pnl: float
    pnl_pct: float


class ScenarioAnalyzer:
    """Analyze bond/portfolio PnL under rate scenarios.

    Parameters
    ----------
    base_zero_rate_fn : callable
        Base zero-rate curve function z(t).
    """

    def __init__(self, base_zero_rate_fn: Callable[[float], float]) -> None:
        self.base_zero_rate_fn = base_zero_rate_fn

    def _make_discount_fn(self, zero_rate_fn: Callable[[float], float]):
        """Convert a zero-rate function to a discount function."""

        def df(t: float) -> float:
            if t <= 0:
                return 1.0
            return np.exp(-zero_rate_fn(t) * t)

        return df

    # ------------------------------------------------------------------
    # Shift generators
    # ------------------------------------------------------------------

    def parallel_shift(self, bps: float) -> Callable[[float], float]:
        """Return shifted zero-rate function: z(t) + shift."""
        shift = bps / 10_000.0

        def shifted(t: float) -> float:
            return self.base_zero_rate_fn(t) + shift

        return shifted

    def twist_shift(
        self, short_bps: float, long_bps: float, pivot_maturity: float = 5.0
    ) -> Callable[[float], float]:
        """Twist: linear interpolation from short shift to long shift.

        Short end shifts by *short_bps*, long end by *long_bps*, with linear
        interpolation between them.  The pivot is the crossover point.
        """
        short_shift = short_bps / 10_000.0
        long_shift = long_bps / 10_000.0

        def shifted(t: float) -> float:
            if t <= 0:
                return self.base_zero_rate_fn(t) + short_shift
            # Linear interpolation of shift magnitude
            alpha = min(t / pivot_maturity, 1.0) if pivot_maturity > 0 else 1.0
            bump = short_shift + alpha * (long_shift - short_shift)
            return self.base_zero_rate_fn(t) + bump

        return shifted

    def butterfly_shift(
        self,
        wing_bps: float,
        belly_bps: float,
        belly_maturity: float = 5.0,
        wing_width: float = 5.0,
    ) -> Callable[[float], float]:
        """Butterfly: belly moves differently from wings.

        Wings (t < belly_maturity - wing_width and t > belly_maturity + wing_width)
        shift by *wing_bps*.  Belly region shifts by *belly_bps*.  Smooth
        Gaussian-like blending.
        """
        wing_shift = wing_bps / 10_000.0
        belly_shift = belly_bps / 10_000.0

        def shifted(t: float) -> float:
            # Gaussian weight centred on belly
            w = np.exp(-0.5 * ((t - belly_maturity) / wing_width) ** 2)
            bump = wing_shift + w * (belly_shift - wing_shift)
            return self.base_zero_rate_fn(t) + bump

        return shifted

    def custom_shift(
        self, maturities: np.ndarray, shift_bps: np.ndarray
    ) -> Callable[[float], float]:
        """Apply arbitrary shift vector via linear interpolation."""
        shifts = np.asarray(shift_bps) / 10_000.0
        mats = np.asarray(maturities)

        def shifted(t: float) -> float:
            bump = float(np.interp(t, mats, shifts))
            return self.base_zero_rate_fn(t) + bump

        return shifted

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_bond(
        self,
        bond: Bond,
        shift_type: ShiftType,
        shift_bps: float = 0.0,
        scenario_name: str = "",
        **kwargs,
    ) -> ScenarioResult:
        """Compute PnL for a single bond under a given scenario.

        Parameters
        ----------
        bond : Bond
            Bond instrument.
        shift_type : ShiftType
            Type of rate shift.
        shift_bps : float
            Magnitude (used for parallel; for twist/butterfly see kwargs).
        scenario_name : str
            Human-readable label.
        **kwargs
            Additional parameters for twist/butterfly/custom shifts.

        Returns
        -------
        ScenarioResult
        """
        base_df = self._make_discount_fn(self.base_zero_rate_fn)
        base_price = bond.price(base_df)

        if shift_type == ShiftType.PARALLEL:
            shifted_zr = self.parallel_shift(shift_bps)
        elif shift_type == ShiftType.TWIST:
            short_bps = kwargs.get("short_bps", shift_bps)
            long_bps = kwargs.get("long_bps", -shift_bps)
            pivot = kwargs.get("pivot_maturity", 5.0)
            shifted_zr = self.twist_shift(short_bps, long_bps, pivot)
        elif shift_type == ShiftType.BUTTERFLY:
            wing_bps = kwargs.get("wing_bps", shift_bps)
            belly_bps = kwargs.get("belly_bps", -shift_bps)
            belly_mat = kwargs.get("belly_maturity", 5.0)
            wing_width = kwargs.get("wing_width", 5.0)
            shifted_zr = self.butterfly_shift(wing_bps, belly_bps, belly_mat, wing_width)
        elif shift_type == ShiftType.CUSTOM:
            mats = kwargs["maturities"]
            shifts = kwargs["shifts_bps"]
            shifted_zr = self.custom_shift(mats, shifts)
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")

        scenario_df = self._make_discount_fn(shifted_zr)
        scenario_price = bond.price(scenario_df)

        pnl = scenario_price - base_price
        pnl_pct = pnl / base_price * 100.0

        return ScenarioResult(
            scenario_name=scenario_name or f"{shift_type.value}_{shift_bps}bps",
            shift_type=shift_type,
            shift_bps=shift_bps,
            base_price=base_price,
            scenario_price=scenario_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

    def parallel_grid(
        self,
        bond: Bond,
        shifts_bps: List[float],
    ) -> List[ScenarioResult]:
        """Run parallel shift scenarios over a grid of shifts.

        Parameters
        ----------
        bond : Bond
            Bond instrument.
        shifts_bps : list of float
            List of shift sizes in basis points.

        Returns
        -------
        list of ScenarioResult
        """
        results = []
        for s in shifts_bps:
            r = self.analyze_bond(bond, ShiftType.PARALLEL, s, f"parallel_{s}bps")
            results.append(r)
        return results

    def heatmap_grid(
        self,
        bonds: List[Bond],
        shifts_bps: List[float],
    ) -> np.ndarray:
        """Generate PnL heatmap: bonds x shifts.

        Returns
        -------
        np.ndarray
            Shape (n_bonds, n_shifts) with PnL values.
        """
        n_bonds = len(bonds)
        n_shifts = len(shifts_bps)
        grid = np.zeros((n_bonds, n_shifts))

        for i, bond in enumerate(bonds):
            for j, s in enumerate(shifts_bps):
                result = self.analyze_bond(bond, ShiftType.PARALLEL, s)
                grid[i, j] = result.pnl

        return grid
