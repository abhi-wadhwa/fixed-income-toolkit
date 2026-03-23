"""Tests for bond pricing."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.bond import FixedCouponBond


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def par_bond() -> FixedCouponBond:
    """5 % semi-annual 5-year par bond."""
    return FixedCouponBond(
        face=100.0,
        coupon_rate=0.05,
        maturity=5.0,
        frequency=2,
    )


@pytest.fixture
def annual_bond() -> FixedCouponBond:
    """6 % annual 3-year bond."""
    return FixedCouponBond(
        face=100.0,
        coupon_rate=0.06,
        maturity=3.0,
        frequency=1,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestCashFlows:
    def test_number_of_flows(self, par_bond: FixedCouponBond) -> None:
        """Semi-annual 5Y bond should have 10 cash flows."""
        assert len(par_bond.cash_flows()) == 10

    def test_final_flow_includes_principal(self, par_bond: FixedCouponBond) -> None:
        """Last cash flow should include face value."""
        last = par_bond.cash_flows()[-1]
        expected_coupon = 100.0 * 0.05 / 2
        np.testing.assert_allclose(last.amount, expected_coupon + 100.0)

    def test_coupon_amount(self, par_bond: FixedCouponBond) -> None:
        """Intermediate coupons should be face * rate / freq."""
        cf = par_bond.cash_flows()[0]
        np.testing.assert_allclose(cf.amount, 2.5)

    def test_annual_flows(self, annual_bond: FixedCouponBond) -> None:
        """Annual 3Y bond: 3 cash flows."""
        assert len(annual_bond.cash_flows()) == 3


class TestPricing:
    def test_par_bond_at_par_yield(self, par_bond: FixedCouponBond) -> None:
        """Bond priced at its coupon rate should trade at par."""
        price = par_bond.price_from_yield(0.05)
        np.testing.assert_allclose(price, 100.0, atol=1e-8)

    def test_premium_bond(self, par_bond: FixedCouponBond) -> None:
        """Lower yield => price above par."""
        price = par_bond.price_from_yield(0.03)
        assert price > 100.0

    def test_discount_bond(self, par_bond: FixedCouponBond) -> None:
        """Higher yield => price below par."""
        price = par_bond.price_from_yield(0.07)
        assert price < 100.0

    def test_annual_bond_pricing(self, annual_bond: FixedCouponBond) -> None:
        """Manual verification: 6 % annual, 3Y, ytm = 6 % => par."""
        price = annual_bond.price_from_yield(0.06)
        np.testing.assert_allclose(price, 100.0, atol=1e-8)


class TestYieldToMaturity:
    def test_ytm_round_trip(self, par_bond: FixedCouponBond) -> None:
        """Price -> YTM -> Price round-trip."""
        ytm_input = 0.06
        price = par_bond.price_from_yield(ytm_input)
        ytm_solved = par_bond.yield_to_maturity(price)
        np.testing.assert_allclose(ytm_solved, ytm_input, atol=1e-10)

    def test_ytm_at_par(self, par_bond: FixedCouponBond) -> None:
        """At par price, YTM should equal coupon rate."""
        ytm = par_bond.yield_to_maturity(100.0)
        np.testing.assert_allclose(ytm, 0.05, atol=1e-10)


class TestAccruedInterest:
    def test_no_accrued(self, par_bond: FixedCouponBond) -> None:
        """No accrual when settlement fraction = 0."""
        assert par_bond.accrued_interest(0.0) == 0.0

    def test_half_period_accrued(self, par_bond: FixedCouponBond) -> None:
        """Half-period accrued interest."""
        ai = par_bond.accrued_interest(0.5)
        expected = 100.0 * 0.05 / 2 * 0.5  # half of semi-annual coupon
        np.testing.assert_allclose(ai, expected)

    def test_dirty_price(self, par_bond: FixedCouponBond) -> None:
        """Dirty price = clean + accrued."""
        clean = 100.0
        dp = par_bond.dirty_price(clean, settlement_fraction=0.5)
        ai = par_bond.accrued_interest(0.5)
        np.testing.assert_allclose(dp, clean + ai)
