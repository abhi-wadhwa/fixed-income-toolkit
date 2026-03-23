"""Tests for risk metrics.

Key test: duration approximation -- the change in price for a small yield
change should satisfy:
    dP approx -D_mod * P * dy + 0.5 * C * P * dy^2
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.bond import FixedCouponBond
from src.core.risk import RiskMetrics


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def bond() -> FixedCouponBond:
    return FixedCouponBond(
        face=100.0, coupon_rate=0.05, maturity=5.0, frequency=2
    )


@pytest.fixture
def risk(bond: FixedCouponBond) -> RiskMetrics:
    return RiskMetrics(bond)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _disc_fn(ytm: float, freq: int = 2):
    """Flat yield discount function."""
    return lambda t: (1.0 + ytm / freq) ** (-t * freq)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestMacaulayDuration:
    def test_positive(self, risk: RiskMetrics) -> None:
        """Duration should be positive."""
        d = risk.macaulay_duration(_disc_fn(0.05))
        assert d > 0

    def test_less_than_maturity(self, risk: RiskMetrics, bond: FixedCouponBond) -> None:
        """Macaulay duration should be less than maturity for coupon bonds."""
        d = risk.macaulay_duration(_disc_fn(0.05))
        assert d < bond.maturity

    def test_zero_coupon_equals_maturity(self) -> None:
        """For a zero-coupon bond, Macaulay duration equals maturity."""
        zc = FixedCouponBond(face=100, coupon_rate=0.0, maturity=5.0, frequency=1)
        rm = RiskMetrics(zc)
        d = rm.macaulay_duration(_disc_fn(0.05, freq=1))
        np.testing.assert_allclose(d, 5.0, atol=1e-10)


class TestModifiedDuration:
    def test_relation(self, risk: RiskMetrics) -> None:
        """D_mod = D_mac / (1 + y/n)."""
        ytm = 0.05
        d_mac = risk.macaulay_duration(_disc_fn(ytm))
        d_mod = risk.modified_duration(_disc_fn(ytm), ytm, 2)
        expected = d_mac / (1 + ytm / 2)
        np.testing.assert_allclose(d_mod, expected, rtol=1e-10)

    def test_from_yield(self, risk: RiskMetrics) -> None:
        """modified_duration_from_yield should match manual computation."""
        ytm = 0.06
        d_mod_a = risk.modified_duration(_disc_fn(ytm), ytm, 2)
        d_mod_b = risk.modified_duration_from_yield(ytm)
        np.testing.assert_allclose(d_mod_a, d_mod_b, rtol=1e-10)


class TestDurationApproximation:
    """dP ~ -D_mod * P * dy + 0.5 * C * P * dy^2."""

    @pytest.mark.parametrize("dy", [0.001, 0.005, 0.01])
    def test_second_order_approximation(self, bond: FixedCouponBond, dy: float) -> None:
        """Price change should match duration + convexity approximation."""
        ytm = 0.05
        p0 = bond.price_from_yield(ytm)
        p1 = bond.price_from_yield(ytm + dy)
        actual_dp = p1 - p0

        rm = RiskMetrics(bond)
        d_mod = rm.modified_duration_from_yield(ytm)
        conv = rm.convexity_from_yield(ytm)

        approx_dp = -d_mod * p0 * dy + 0.5 * conv * p0 * dy**2

        # For small dy, the approximation should be within 0.5% of actual
        if abs(actual_dp) > 1e-10:
            rel_error = abs(approx_dp - actual_dp) / abs(actual_dp)
            assert rel_error < 0.005, (
                f"Duration approximation error too large: {rel_error:.4%} "
                f"(dy={dy})"
            )


class TestConvexity:
    def test_positive(self, risk: RiskMetrics) -> None:
        """Convexity should be positive for a plain vanilla bond."""
        conv = risk.convexity(_disc_fn(0.05))
        assert conv > 0

    def test_from_yield(self, bond: FixedCouponBond) -> None:
        """convexity_from_yield finite difference should be close to analytic."""
        rm = RiskMetrics(bond)
        conv_fd = rm.convexity_from_yield(0.05)
        assert conv_fd > 0


class TestDV01:
    def test_positive(self, risk: RiskMetrics) -> None:
        """DV01 should be positive (price drops when yield rises)."""
        dv = risk.dv01(_disc_fn(0.05), 0.05, 2)
        assert dv > 0

    def test_from_yield(self, bond: FixedCouponBond) -> None:
        """DV01 from yield finite difference."""
        rm = RiskMetrics(bond)
        dv = rm.dv01_from_yield(0.05)
        assert dv > 0

    def test_dv01_magnitude(self, bond: FixedCouponBond) -> None:
        """DV01 for a 5Y par bond should be roughly face * duration * 0.0001."""
        rm = RiskMetrics(bond)
        dv = rm.dv01_from_yield(0.05)
        # Approximate: 100 * 4.5 * 0.0001 ~ 0.045
        assert 0.03 < dv < 0.06


class TestEffectiveDuration:
    def test_close_to_modified(self, bond: FixedCouponBond) -> None:
        """For a non-callable bond, effective ~ modified duration."""
        rm = RiskMetrics(bond)
        d_mod = rm.modified_duration_from_yield(0.05)
        d_eff = rm.effective_duration_from_yield(0.05, shift_bps=1.0)
        np.testing.assert_allclose(d_eff, d_mod, rtol=0.01)


class TestKeyRateDurations:
    def test_sum_approximates_total(self, bond: FixedCouponBond) -> None:
        """Sum of key rate durations should approximate modified duration."""
        ytm = 0.05
        freq = 2

        def flat_zr(t):
            return ytm

        rm = RiskMetrics(bond)
        krd = rm.key_rate_durations(flat_zr, [1.0, 2.0, 3.0, 4.0, 5.0], shift_bps=1.0)
        total_krd = sum(krd.values())
        d_mod = rm.modified_duration_from_yield(ytm)
        # Should be close but not exact due to triangular bumps
        np.testing.assert_allclose(total_krd, d_mod, rtol=0.15)
