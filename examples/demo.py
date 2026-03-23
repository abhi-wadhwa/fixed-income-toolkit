"""End-to-end demo of the Fixed Income Analytics Toolkit.

Demonstrates:
    1. Yield curve bootstrapping from deposits and swaps
    2. Nelson-Siegel-Svensson fitting
    3. Bond pricing from the fitted curve
    4. Full risk metrics (duration, convexity, DV01)
    5. Scenario analysis (parallel shift, twist, butterfly)
    6. Portfolio analytics and immunization
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.bond import FixedCouponBond
from src.core.curve import InstrumentType, InterpolationMethod, MarketInstrument, YieldCurve
from src.core.nss import NelsonSiegelSvensson, NSSParams
from src.core.portfolio import Portfolio, PortfolioPosition
from src.core.risk import RiskMetrics
from src.core.scenario import ScenarioAnalyzer, ShiftType


def main() -> None:
    print("=" * 70)
    print("  Fixed Income Analytics Toolkit - Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Yield curve bootstrapping
    # ------------------------------------------------------------------
    print("\n--- 1. Yield Curve Bootstrapping ---\n")

    instruments = [
        MarketInstrument(InstrumentType.DEPOSIT, maturity=0.25, rate=0.040),
        MarketInstrument(InstrumentType.DEPOSIT, maturity=0.50, rate=0.042),
        MarketInstrument(InstrumentType.DEPOSIT, maturity=1.00, rate=0.045),
        MarketInstrument(InstrumentType.SWAP, maturity=2.0, rate=0.046),
        MarketInstrument(InstrumentType.SWAP, maturity=3.0, rate=0.047),
        MarketInstrument(InstrumentType.SWAP, maturity=5.0, rate=0.048),
        MarketInstrument(InstrumentType.SWAP, maturity=7.0, rate=0.049),
        MarketInstrument(InstrumentType.SWAP, maturity=10.0, rate=0.050),
    ]

    curve = YieldCurve(instruments, InterpolationMethod.CUBIC_SPLINE)

    print(f"{'Maturity':>10s}  {'Zero Rate':>12s}  {'Discount Factor':>16s}")
    print("-" * 42)
    for t in curve.pillar_maturities():
        zr = curve.zero_rate(t)
        df = curve.discount_factor(t)
        print(f"{t:10.2f}  {zr * 100:11.4f}%  {df:16.8f}")

    # ------------------------------------------------------------------
    # 2. Nelson-Siegel-Svensson
    # ------------------------------------------------------------------
    print("\n--- 2. Nelson-Siegel-Svensson Fit ---\n")

    maturities = curve.pillar_maturities()
    zero_rates = curve.pillar_zero_rates()

    nss = NelsonSiegelSvensson()
    params = nss.fit(maturities, zero_rates)
    print(f"beta0 = {params.beta0:.6f}")
    print(f"beta1 = {params.beta1:.6f}")
    print(f"beta2 = {params.beta2:.6f}")
    print(f"beta3 = {params.beta3:.6f}")
    print(f"tau1  = {params.tau1:.6f}")
    print(f"tau2  = {params.tau2:.6f}")

    print("\nNSS vs bootstrapped zero rates:")
    for t in maturities:
        z_boot = curve.zero_rate(t) * 100
        z_nss = nss.zero_rate(t) * 100
        print(f"  T={t:5.2f}  bootstrap={z_boot:.4f}%  NSS={z_nss:.4f}%")

    # ------------------------------------------------------------------
    # 3. Bond pricing
    # ------------------------------------------------------------------
    print("\n--- 3. Bond Pricing ---\n")

    bond = FixedCouponBond(
        face=100.0, coupon_rate=0.05, maturity=5.0, frequency=2
    )

    price_curve = bond.price(curve.discount_factor)
    price_flat = bond.price_from_yield(0.05)
    ytm = bond.yield_to_maturity(price_curve)

    print(f"5Y 5% semi-annual bond:")
    print(f"  Price (from curve): {price_curve:.6f}")
    print(f"  Price (flat 5%):    {price_flat:.6f}")
    print(f"  Implied YTM:        {ytm * 100:.4f}%")

    # ------------------------------------------------------------------
    # 4. Risk metrics
    # ------------------------------------------------------------------
    print("\n--- 4. Risk Metrics ---\n")

    rm = RiskMetrics(bond)
    report = rm.full_report(
        discount_fn=curve.discount_factor,
        ytm=ytm,
        frequency=2,
        curve_zero_rate_fn=curve.zero_rate,
        key_rates=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    )

    print(f"  Price:              {report.price:.6f}")
    print(f"  Macaulay Duration:  {report.macaulay_duration:.6f}")
    print(f"  Modified Duration:  {report.modified_duration:.6f}")
    print(f"  Effective Duration: {report.effective_duration:.6f}")
    print(f"  Convexity:          {report.convexity:.4f}")
    print(f"  DV01:               {report.dv01:.6f}")

    if report.key_rate_durations:
        print("\n  Key Rate Durations:")
        for kr, krd in sorted(report.key_rate_durations.items()):
            print(f"    {kr:5.1f}Y: {krd:.6f}")

    # ------------------------------------------------------------------
    # 5. Scenario analysis
    # ------------------------------------------------------------------
    print("\n--- 5. Scenario Analysis ---\n")

    analyzer = ScenarioAnalyzer(curve.zero_rate)

    # Parallel shifts
    print("Parallel shifts:")
    for bps in [-100, -50, -25, 0, 25, 50, 100]:
        result = analyzer.analyze_bond(bond, ShiftType.PARALLEL, bps)
        print(
            f"  {bps:+4d} bps: price={result.scenario_price:.4f}  "
            f"PnL={result.pnl:+.4f} ({result.pnl_pct:+.4f}%)"
        )

    # Twist
    print("\nTwist scenario (short +50, long -50):")
    result = analyzer.analyze_bond(
        bond, ShiftType.TWIST, scenario_name="twist",
        short_bps=50, long_bps=-50, pivot_maturity=5.0,
    )
    print(f"  Price: {result.scenario_price:.4f}  PnL: {result.pnl:+.4f}")

    # Butterfly
    print("\nButterfly scenario (wings +25, belly -25):")
    result = analyzer.analyze_bond(
        bond, ShiftType.BUTTERFLY, scenario_name="butterfly",
        wing_bps=25, belly_bps=-25, belly_maturity=5.0,
    )
    print(f"  Price: {result.scenario_price:.4f}  PnL: {result.pnl:+.4f}")

    # ------------------------------------------------------------------
    # 6. Portfolio analytics
    # ------------------------------------------------------------------
    print("\n--- 6. Portfolio Analytics ---\n")

    bond_2y = FixedCouponBond(face=100, coupon_rate=0.03, maturity=2.0, frequency=2)
    bond_10y = FixedCouponBond(face=100, coupon_rate=0.05, maturity=10.0, frequency=2)

    portfolio = Portfolio()
    portfolio.add_position(bond_2y, quantity=50, label="2Y Bond")
    portfolio.add_position(bond_10y, quantity=30, label="10Y Bond")

    ytm_map = {"2Y Bond": 0.045, "10Y Bond": 0.05}
    report = portfolio.risk_report(curve.discount_factor, ytm_map)

    print(f"  Total Market Value: {report.total_market_value:.2f}")
    print(f"  Portfolio Duration: {report.portfolio_duration:.4f}")
    print(f"  Modified Duration:  {report.portfolio_modified_duration:.4f}")
    print(f"  Portfolio Convexity:{report.portfolio_convexity:.4f}")
    print(f"  Portfolio DV01:     {report.portfolio_dv01:.6f}")

    print("\n  Position weights:")
    for label, w in report.position_weights.items():
        print(f"    {label}: {w:.4f}")

    # Immunization
    print("\n  Immunization (target duration = 5.0Y):")
    weights = portfolio.immunization_target(5.0, curve.discount_factor)
    for label, w in weights.items():
        print(f"    {label}: {w:.4f}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
