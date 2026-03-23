"""Yield curve bootstrapping from market instruments.

Supports bootstrapping from:
    - Cash deposits (short end)
    - Forward Rate Agreements (FRAs) (medium term)
    - Interest rate swaps (long end)

Interpolation modes:
    - Cubic spline on zero rates
    - Nelson-Siegel-Svensson parametric fit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from src.core.nss import NelsonSiegelSvensson, NSSParams


class InstrumentType(Enum):
    DEPOSIT = "deposit"
    FRA = "fra"
    SWAP = "swap"


@dataclass
class MarketInstrument:
    """A market instrument used for curve bootstrapping.

    Parameters
    ----------
    instrument_type : InstrumentType
        Type of instrument.
    maturity : float
        Maturity in years.
    rate : float
        Quoted rate (annualised, continuously compounded equivalent will be
        derived internally for deposits; swap rate for swaps).
    start : float
        Start time in years (relevant for FRAs). Defaults to 0.
    """

    instrument_type: InstrumentType
    maturity: float
    rate: float
    start: float = 0.0


class InterpolationMethod(Enum):
    CUBIC_SPLINE = "cubic_spline"
    NSS = "nss"


class YieldCurve:
    """Bootstrapped yield curve with interpolation.

    The curve is built from a sorted list of market instruments.  Discount
    factors are solved sequentially so that each instrument reprices to par.

    Parameters
    ----------
    instruments : list of MarketInstrument
        Market instruments sorted by maturity.
    interpolation : InterpolationMethod
        Interpolation method for the zero-rate curve.
    """

    def __init__(
        self,
        instruments: List[MarketInstrument],
        interpolation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE,
    ) -> None:
        self.instruments = sorted(instruments, key=lambda i: i.maturity)
        self.interpolation = interpolation

        # Bootstrapped pillars: {maturity: discount_factor}
        self._pillars: Dict[float, float] = {0.0: 1.0}
        self._zero_rates: Dict[float, float] = {}
        self._spline: Optional[CubicSpline] = None
        self._nss: Optional[NelsonSiegelSvensson] = None

        self._bootstrap()
        self._build_interpolator()

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Sequentially strip discount factors from market instruments."""
        for inst in self.instruments:
            if inst.instrument_type == InstrumentType.DEPOSIT:
                self._strip_deposit(inst)
            elif inst.instrument_type == InstrumentType.FRA:
                self._strip_fra(inst)
            elif inst.instrument_type == InstrumentType.SWAP:
                self._strip_swap(inst)

    def _strip_deposit(self, inst: MarketInstrument) -> None:
        """Deposit: D(T) = 1 / (1 + r * T)."""
        t = inst.maturity
        df = 1.0 / (1.0 + inst.rate * t)
        self._pillars[t] = df
        self._zero_rates[t] = -np.log(df) / t

    def _strip_fra(self, inst: MarketInstrument) -> None:
        """FRA: D(T2) = D(T1) / (1 + r * (T2 - T1))."""
        t1 = inst.start
        t2 = inst.maturity
        dt = t2 - t1
        df_start = self._get_pillar_df(t1)
        df_end = df_start / (1.0 + inst.rate * dt)
        self._pillars[t2] = df_end
        self._zero_rates[t2] = -np.log(df_end) / t2

    def _strip_swap(self, inst: MarketInstrument) -> None:
        """Swap: solve for D(T) such that fixed leg PV = floating leg PV = 1.

        Fixed leg pays annual coupons at the swap rate.
        PV_fixed = sum_{i=1}^{n} c * D(t_i) + D(T) = 1
        => D(T) = (1 - c * sum D(t_i)) / (1 + c)
        where c = swap_rate and t_i are annual payment dates.

        Intermediate coupon dates that are not already pillars are
        interpolated and stored as pillars so the spline passes through
        them consistently.
        """
        c = inst.rate
        t = inst.maturity
        n = int(round(t))

        # Sum discount factors for intermediate coupon dates
        pv_coupons = 0.0
        for i in range(1, n):
            ti = float(i)
            df_i = self._get_pillar_df(ti)
            # Store interpolated intermediate DFs as pillars for consistency
            if ti not in self._pillars:
                self._pillars[ti] = df_i
                self._zero_rates[ti] = -np.log(df_i) / ti
            pv_coupons += c * df_i

        # Solve for D(T): 1 = c * sum_D + (1+c) * D(T)
        df = (1.0 - pv_coupons) / (1.0 + c)

        # Ensure positive discount factor
        if df <= 0:
            raise ValueError(
                f"Negative discount factor at T={t}. Check input instruments."
            )

        self._pillars[t] = df
        self._zero_rates[t] = -np.log(df) / t

    def _get_pillar_df(self, t: float) -> float:
        """Look up or interpolate discount factor at pillar *t*.

        During bootstrapping, intermediate maturities may not yet have
        pillars.  We interpolate between known pillars using linear
        interpolation on zero rates, and flat-extrapolate beyond the
        last known pillar.
        """
        if t in self._pillars:
            return self._pillars[t]

        # Sorted known maturities (always includes 0.0)
        ts = sorted(self._pillars.keys())

        # If t is beyond the last pillar, flat-extrapolate the last zero rate
        if t > ts[-1]:
            t_last = ts[-1]
            if t_last > 0:
                z_last = -np.log(self._pillars[t_last]) / t_last
            else:
                # Only t=0 is known; use a small default rate
                z_last = 0.0
            return np.exp(-z_last * t)

        # Interpolate between bracketing pillars
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                t0, t1 = ts[i], ts[i + 1]
                z0 = -np.log(self._pillars[t0]) / t0 if t0 > 0 else 0.0
                z1 = -np.log(self._pillars[t1]) / t1
                z = z0 + (z1 - z0) * (t - t0) / (t1 - t0)
                return np.exp(-z * t)

        raise ValueError(f"Cannot interpolate discount factor at t={t}")

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _build_interpolator(self) -> None:
        """Build spline or NSS interpolator from bootstrapped pillars."""
        # Exclude t=0 for zero rate interpolation
        ts = sorted(t for t in self._zero_rates.keys() if t > 0)
        zrs = np.array([self._zero_rates[t] for t in ts])
        ts_arr = np.array(ts)

        if self.interpolation == InterpolationMethod.CUBIC_SPLINE:
            self._spline = CubicSpline(ts_arr, zrs, bc_type="natural")
        elif self.interpolation == InterpolationMethod.NSS:
            self._nss = NelsonSiegelSvensson()
            self._nss.fit(ts_arr, zrs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Continuously compounded zero rate at maturity *t* (years).

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Zero rate(s).
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))

        if self.interpolation == InterpolationMethod.CUBIC_SPLINE:
            if self._spline is None:
                raise RuntimeError("Spline not initialised.")
            result = self._spline(t_arr)
        else:
            if self._nss is None:
                raise RuntimeError("NSS model not initialised.")
            result = self._nss.zero_rate(t_arr)

        if np.ndim(t) == 0:
            return float(result[0])
        return result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Discount factor D(t) = exp(-z(t) * t).

        Parameters
        ----------
        t : float or array-like
            Maturity in years.

        Returns
        -------
        float or ndarray
            Discount factor(s).
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        z = np.atleast_1d(self.zero_rate(t_arr))
        df = np.exp(-z * t_arr)
        if np.ndim(t) == 0:
            return float(df[0])
        return df

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between *t1* and *t2*.

        f(t1, t2) = (z(t2)*t2 - z(t1)*t1) / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        z1 = self.zero_rate(t1) if t1 > 0 else 0.0
        z2 = self.zero_rate(t2)
        return (z2 * t2 - z1 * t1) / (t2 - t1)

    def pillar_maturities(self) -> np.ndarray:
        """Return sorted array of bootstrapped pillar maturities (excluding 0)."""
        return np.array(sorted(t for t in self._pillars if t > 0))

    def pillar_discount_factors(self) -> np.ndarray:
        """Return discount factors at pillar maturities."""
        ts = self.pillar_maturities()
        return np.array([self._pillars[t] for t in ts])

    def pillar_zero_rates(self) -> np.ndarray:
        """Return zero rates at pillar maturities."""
        ts = self.pillar_maturities()
        return np.array([self._zero_rates[t] for t in ts])

    def shift(self, bps: float) -> "YieldCurve":
        """Return a new curve with a parallel shift of *bps* basis points.

        This creates shifted instruments and re-bootstraps.
        """
        shift_rate = bps / 10_000.0
        shifted = []
        for inst in self.instruments:
            shifted.append(
                MarketInstrument(
                    instrument_type=inst.instrument_type,
                    maturity=inst.maturity,
                    rate=inst.rate + shift_rate,
                    start=inst.start,
                )
            )
        return YieldCurve(shifted, interpolation=self.interpolation)
