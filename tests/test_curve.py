"""Tests for yield curve bootstrapping.

Key invariant: the bootstrapped curve must reprice every input instrument
exactly (to numerical tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.curve import InstrumentType, InterpolationMethod, MarketInstrument, YieldCurve


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sample_instruments() -> list[MarketInstrument]:
    return [
        MarketInstrument(InstrumentType.DEPOSIT, maturity=0.25, rate=0.040),
        MarketInstrument(InstrumentType.DEPOSIT, maturity=0.50, rate=0.042),
        MarketInstrument(InstrumentType.DEPOSIT, maturity=1.00, rate=0.045),
        MarketInstrument(InstrumentType.SWAP, maturity=2.0, rate=0.046),
        MarketInstrument(InstrumentType.SWAP, maturity=3.0, rate=0.047),
        MarketInstrument(InstrumentType.SWAP, maturity=5.0, rate=0.048),
        MarketInstrument(InstrumentType.SWAP, maturity=7.0, rate=0.049),
        MarketInstrument(InstrumentType.SWAP, maturity=10.0, rate=0.050),
    ]


@pytest.fixture
def curve(sample_instruments) -> YieldCurve:
    return YieldCurve(sample_instruments)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestBootstrap:
    """Bootstrapped curve reprices input instruments."""

    def test_deposit_repricing(self, curve: YieldCurve) -> None:
        """Deposit: D(T) should equal 1 / (1 + r * T)."""
        for inst in curve.instruments:
            if inst.instrument_type != InstrumentType.DEPOSIT:
                continue
            expected_df = 1.0 / (1.0 + inst.rate * inst.maturity)
            actual_df = curve.discount_factor(inst.maturity)
            np.testing.assert_allclose(actual_df, expected_df, rtol=1e-6)

    def test_swap_repricing(self, curve: YieldCurve) -> None:
        """Swap: PV(fixed leg) should equal 1 (par).

        Uses raw pillar discount factors (not interpolated) for exact
        repricing at pillar maturities.  For intermediate coupon dates
        that are not pillars, we use the interpolated DF, so we allow a
        slightly larger tolerance.
        """
        for inst in curve.instruments:
            if inst.instrument_type != InstrumentType.SWAP:
                continue
            c = inst.rate
            n = int(round(inst.maturity))

            def _df(t: float) -> float:
                # Use raw pillar DF if available, else interpolated
                if t in curve._pillars:
                    return curve._pillars[t]
                return curve.discount_factor(t)

            pv_fixed = sum(c * _df(float(i)) for i in range(1, n + 1))
            pv_fixed += _df(inst.maturity)
            np.testing.assert_allclose(pv_fixed, 1.0, atol=1e-10)

    def test_discount_factor_monotonic(self, curve: YieldCurve) -> None:
        """Discount factors should be monotonically decreasing."""
        ts = np.linspace(0.25, 10.0, 50)
        dfs = np.array([curve.discount_factor(t) for t in ts])
        assert np.all(np.diff(dfs) < 0), "Discount factors not monotonically decreasing"

    def test_zero_rate_positive(self, curve: YieldCurve) -> None:
        """Zero rates should be positive for a normal curve."""
        ts = np.linspace(0.25, 10.0, 50)
        zrs = np.array([curve.zero_rate(t) for t in ts])
        assert np.all(zrs > 0), "Negative zero rates detected"


class TestInterpolation:
    """Test interpolation methods."""

    def test_cubic_spline_smooth(self, sample_instruments) -> None:
        """Cubic spline zero rates should be smooth (small second derivative)."""
        curve = YieldCurve(sample_instruments, InterpolationMethod.CUBIC_SPLINE)
        ts = np.linspace(0.5, 9.5, 100)
        zrs = np.array([curve.zero_rate(t) for t in ts])
        # Check smoothness via finite-difference second derivative
        d2 = np.diff(zrs, n=2)
        assert np.max(np.abs(d2)) < 0.01, "Spline not smooth"

    def test_nss_interpolation(self, sample_instruments) -> None:
        """NSS curve should produce positive zero rates."""
        curve = YieldCurve(sample_instruments, InterpolationMethod.NSS)
        ts = np.linspace(0.5, 10.0, 50)
        zrs = np.array([curve.zero_rate(t) for t in ts])
        assert np.all(zrs > 0), "NSS produced negative zero rates"


class TestForwardRate:
    """Test forward rate computations."""

    def test_forward_rate_positive(self, curve: YieldCurve) -> None:
        """Forward rates should be positive for an upward-sloping curve."""
        for t in [0.5, 1.0, 2.0, 5.0]:
            fwd = curve.forward_rate(t, t + 0.5)
            assert fwd > 0, f"Negative forward rate at t={t}"

    def test_forward_rate_consistency(self, curve: YieldCurve) -> None:
        """Forward rates should be consistent with zero rates.

        D(t2) = D(t1) * exp(-f * (t2 - t1))
        """
        t1, t2 = 2.0, 5.0
        f = curve.forward_rate(t1, t2)
        d1 = curve.discount_factor(t1)
        d2 = curve.discount_factor(t2)
        d2_check = d1 * np.exp(-f * (t2 - t1))
        np.testing.assert_allclose(d2, d2_check, rtol=1e-6)


class TestShift:
    """Test parallel curve shifts."""

    def test_shift_increases_rates(self, curve: YieldCurve) -> None:
        """Positive shift should increase zero rates."""
        shifted = curve.shift(50)  # +50 bps
        for t in [1.0, 3.0, 5.0, 10.0]:
            assert shifted.zero_rate(t) > curve.zero_rate(t)

    def test_shift_decreases_df(self, curve: YieldCurve) -> None:
        """Positive shift should decrease discount factors."""
        shifted = curve.shift(50)
        for t in [1.0, 5.0, 10.0]:
            assert shifted.discount_factor(t) < curve.discount_factor(t)
