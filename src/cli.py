"""Command-line interface for the Fixed Income Analytics Toolkit.

Usage:
    python -m src.cli price --coupon 0.05 --maturity 5 --ytm 0.05
    python -m src.cli risk --coupon 0.05 --maturity 5 --ytm 0.05
    python -m src.cli curve
    python -m src.cli scenario --coupon 0.05 --maturity 5 --shift 50
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.core.bond import FixedCouponBond
from src.core.curve import InstrumentType, InterpolationMethod, MarketInstrument, YieldCurve
from src.core.risk import RiskMetrics
from src.core.scenario import ScenarioAnalyzer, ShiftType


def _default_curve() -> YieldCurve:
    """Build a sample curve from default instruments."""
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
    return YieldCurve(instruments)


def cmd_price(args: argparse.Namespace) -> None:
    """Price a bond."""
    bond = FixedCouponBond(
        face=args.face,
        coupon_rate=args.coupon,
        maturity=args.maturity,
        frequency=args.frequency,
    )
    price = bond.price_from_yield(args.ytm)
    print(f"Bond Price:    {price:.6f}")
    print(f"Coupon Rate:   {args.coupon * 100:.2f}%")
    print(f"Maturity:      {args.maturity:.1f}Y")
    print(f"YTM:           {args.ytm * 100:.2f}%")


def cmd_risk(args: argparse.Namespace) -> None:
    """Compute risk metrics."""
    bond = FixedCouponBond(
        face=args.face,
        coupon_rate=args.coupon,
        maturity=args.maturity,
        frequency=args.frequency,
    )
    ytm = args.ytm
    freq = args.frequency
    price = bond.price_from_yield(ytm)

    rm = RiskMetrics(bond)
    d_mac = rm.macaulay_duration(
        lambda t: (1 + ytm / freq) ** (-t * freq)
    )
    d_mod = rm.modified_duration_from_yield(ytm)
    conv = rm.convexity_from_yield(ytm)
    dv01 = rm.dv01_from_yield(ytm)

    print(f"Price:              {price:.6f}")
    print(f"Macaulay Duration:  {d_mac:.6f}")
    print(f"Modified Duration:  {d_mod:.6f}")
    print(f"Convexity:          {conv:.4f}")
    print(f"DV01:               {dv01:.6f}")


def cmd_curve(args: argparse.Namespace) -> None:
    """Display bootstrapped yield curve."""
    curve = _default_curve()
    ts = curve.pillar_maturities()
    print(f"{'Maturity':>10s}  {'Zero Rate':>12s}  {'Discount Factor':>16s}")
    print("-" * 42)
    for t in ts:
        zr = curve.zero_rate(t)
        df = curve.discount_factor(t)
        print(f"{t:10.2f}  {zr * 100:11.4f}%  {df:16.8f}")


def cmd_scenario(args: argparse.Namespace) -> None:
    """Run parallel shift scenario."""
    bond = FixedCouponBond(
        face=args.face,
        coupon_rate=args.coupon,
        maturity=args.maturity,
        frequency=args.frequency,
    )

    base_zr = lambda t: args.ytm
    analyzer = ScenarioAnalyzer(base_zr)

    shifts = list(range(-args.shift, args.shift + 1, max(args.shift // 5, 1)))
    results = analyzer.parallel_grid(bond, shifts)

    print(f"{'Shift (bps)':>12s}  {'Price':>10s}  {'PnL':>10s}  {'PnL (%)':>10s}")
    print("-" * 46)
    for r in results:
        print(
            f"{r.shift_bps:12.0f}  {r.scenario_price:10.4f}  "
            f"{r.pnl:10.4f}  {r.pnl_pct:9.4f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fixed-income-toolkit",
        description="Fixed Income Analytics Toolkit CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # Common bond arguments
    def add_bond_args(p):
        p.add_argument("--face", type=float, default=100.0, help="Face value")
        p.add_argument("--coupon", type=float, default=0.05, help="Coupon rate")
        p.add_argument("--maturity", type=float, default=5.0, help="Maturity (years)")
        p.add_argument("--frequency", type=int, default=2, help="Coupon frequency")
        p.add_argument("--ytm", type=float, default=0.05, help="Yield to maturity")

    p_price = sub.add_parser("price", help="Price a bond")
    add_bond_args(p_price)

    p_risk = sub.add_parser("risk", help="Compute risk metrics")
    add_bond_args(p_risk)

    p_curve = sub.add_parser("curve", help="Display bootstrapped curve")

    p_scenario = sub.add_parser("scenario", help="Parallel shift scenario")
    add_bond_args(p_scenario)
    p_scenario.add_argument("--shift", type=int, default=50, help="Max shift (bps)")

    args = parser.parse_args()
    if args.command == "price":
        cmd_price(args)
    elif args.command == "risk":
        cmd_risk(args)
    elif args.command == "curve":
        cmd_curve(args)
    elif args.command == "scenario":
        cmd_scenario(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
