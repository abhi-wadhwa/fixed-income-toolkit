"""Streamlit interactive dashboard for fixed income analytics.

Run with:
    streamlit run src/viz/app.py

Pages:
    1. Yield Curve Plotter
    2. Bond Calculator
    3. Scenario Heatmap
    4. Cash Flow Waterfall
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.bond import FixedCouponBond
from src.core.curve import InstrumentType, InterpolationMethod, MarketInstrument, YieldCurve
from src.core.nss import NelsonSiegelSvensson, NSSParams
from src.core.portfolio import Portfolio, PortfolioPosition
from src.core.risk import RiskMetrics
from src.core.scenario import ScenarioAnalyzer, ShiftType

# ------------------------------------------------------------------
# App configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Fixed Income Analytics Toolkit",
    page_icon="",
    layout="wide",
)

st.title("Fixed Income Analytics Toolkit")

page = st.sidebar.selectbox(
    "Page",
    [
        "Yield Curve Plotter",
        "Bond Calculator",
        "Scenario Heatmap",
        "Cash Flow Waterfall",
    ],
)


# ------------------------------------------------------------------
# Helper: build default curve
# ------------------------------------------------------------------


def _build_default_curve(
    deposit_rates: dict[float, float],
    swap_rates: dict[float, float],
    method: str,
) -> YieldCurve:
    """Build a yield curve from user-supplied deposit and swap rates."""
    instruments = []
    for mat, rate in deposit_rates.items():
        instruments.append(
            MarketInstrument(InstrumentType.DEPOSIT, maturity=mat, rate=rate)
        )
    for mat, rate in swap_rates.items():
        instruments.append(
            MarketInstrument(InstrumentType.SWAP, maturity=mat, rate=rate)
        )
    interp = (
        InterpolationMethod.CUBIC_SPLINE
        if method == "Cubic Spline"
        else InterpolationMethod.NSS
    )
    return YieldCurve(instruments, interpolation=interp)


# ==================================================================
# PAGE 1: Yield Curve Plotter
# ==================================================================

if page == "Yield Curve Plotter":
    st.header("Yield Curve Plotter")
    st.markdown(
        "Build a yield curve from deposit and swap rates, then visualise "
        "zero rates, forward rates, and discount factors."
    )

    col_inputs, col_chart = st.columns([1, 2])

    with col_inputs:
        st.subheader("Deposit Rates")
        dep_0_25 = st.number_input("0.25Y deposit rate", value=0.040, step=0.001, format="%.4f")
        dep_0_50 = st.number_input("0.50Y deposit rate", value=0.042, step=0.001, format="%.4f")
        dep_1_00 = st.number_input("1.00Y deposit rate", value=0.045, step=0.001, format="%.4f")

        st.subheader("Swap Rates")
        sw_2 = st.number_input("2Y swap rate", value=0.046, step=0.001, format="%.4f")
        sw_3 = st.number_input("3Y swap rate", value=0.047, step=0.001, format="%.4f")
        sw_5 = st.number_input("5Y swap rate", value=0.048, step=0.001, format="%.4f")
        sw_7 = st.number_input("7Y swap rate", value=0.049, step=0.001, format="%.4f")
        sw_10 = st.number_input("10Y swap rate", value=0.050, step=0.001, format="%.4f")

        method = st.selectbox("Interpolation", ["Cubic Spline", "Nelson-Siegel-Svensson"])

    deposits = {0.25: dep_0_25, 0.50: dep_0_50, 1.0: dep_1_00}
    swaps = {2.0: sw_2, 3.0: sw_3, 5.0: sw_5, 7.0: sw_7, 10.0: sw_10}

    try:
        curve = _build_default_curve(deposits, swaps, method)
        ts = np.linspace(0.25, 10.0, 200)
        zr = np.array([curve.zero_rate(t) for t in ts]) * 100
        fwd = np.array(
            [curve.forward_rate(max(t - 0.01, 0.01), t + 0.01) for t in ts]
        ) * 100
        df_vals = np.array([curve.discount_factor(t) for t in ts])

        with col_chart:
            tab_zr, tab_fwd, tab_df = st.tabs(
                ["Zero Rates", "Forward Rates", "Discount Factors"]
            )

            with tab_zr:
                chart_data = pd.DataFrame({"Maturity (Y)": ts, "Zero Rate (%)": zr})
                st.line_chart(chart_data, x="Maturity (Y)", y="Zero Rate (%)")

            with tab_fwd:
                chart_data = pd.DataFrame(
                    {"Maturity (Y)": ts, "Forward Rate (%)": fwd}
                )
                st.line_chart(chart_data, x="Maturity (Y)", y="Forward Rate (%)")

            with tab_df:
                chart_data = pd.DataFrame(
                    {"Maturity (Y)": ts, "Discount Factor": df_vals}
                )
                st.line_chart(chart_data, x="Maturity (Y)", y="Discount Factor")

            # Pillar data table
            st.subheader("Bootstrapped Pillars")
            pillar_ts = curve.pillar_maturities()
            pillar_zr = curve.pillar_zero_rates() * 100
            pillar_df = curve.pillar_discount_factors()
            df_table = pd.DataFrame(
                {
                    "Maturity": pillar_ts,
                    "Zero Rate (%)": np.round(pillar_zr, 4),
                    "Discount Factor": np.round(pillar_df, 6),
                }
            )
            st.dataframe(df_table, use_container_width=True)
    except Exception as e:
        st.error(f"Error building curve: {e}")

# ==================================================================
# PAGE 2: Bond Calculator
# ==================================================================

elif page == "Bond Calculator":
    st.header("Bond Calculator")

    col_params, col_results = st.columns([1, 2])

    with col_params:
        face = st.number_input("Face Value", value=100.0, step=1.0)
        coupon = st.number_input(
            "Coupon Rate (%)", value=5.0, step=0.1, format="%.2f"
        )
        maturity = st.number_input("Maturity (years)", value=5.0, step=0.5)
        freq = st.selectbox("Coupon Frequency", [1, 2, 4], index=1)
        ytm_input = st.number_input(
            "Yield to Maturity (%)", value=5.0, step=0.1, format="%.2f"
        )
        day_count = st.selectbox(
            "Day Count Convention", ["30/360", "ACT/360", "ACT/365"]
        )

    bond = FixedCouponBond(
        face=face,
        coupon_rate=coupon / 100.0,
        maturity=maturity,
        frequency=freq,
        day_count=day_count,
    )
    ytm = ytm_input / 100.0
    price = bond.price_from_yield(ytm)

    rm = RiskMetrics(bond)

    def disc_fn(t):
        return (1.0 + ytm / freq) ** (-t * freq)

    d_mac = rm.macaulay_duration(disc_fn)
    d_mod = rm.modified_duration(disc_fn, ytm, freq)
    d_eff = rm.effective_duration_from_yield(ytm)
    conv = rm.convexity_from_yield(ytm)
    dv01 = rm.dv01_from_yield(ytm)

    with col_results:
        st.subheader("Pricing")
        met_cols = st.columns(3)
        met_cols[0].metric("Clean Price", f"{price:.4f}")
        met_cols[1].metric("Yield (%)", f"{ytm * 100:.4f}")
        met_cols[2].metric("Face Value", f"{face:.2f}")

        st.subheader("Risk Metrics")
        met_cols2 = st.columns(3)
        met_cols2[0].metric("Macaulay Duration", f"{d_mac:.4f}")
        met_cols2[1].metric("Modified Duration", f"{d_mod:.4f}")
        met_cols2[2].metric("Effective Duration", f"{d_eff:.4f}")

        met_cols3 = st.columns(3)
        met_cols3[0].metric("Convexity", f"{conv:.4f}")
        met_cols3[1].metric("DV01", f"{dv01:.6f}")
        met_cols3[2].metric("Annual Coupon", f"{face * coupon / 100:.2f}")

        # Cash flows table
        st.subheader("Cash Flows")
        cfs = bond.cash_flows()
        cf_data = pd.DataFrame(
            {"Period (Y)": [cf.time for cf in cfs], "Amount": [cf.amount for cf in cfs]}
        )
        st.dataframe(cf_data, use_container_width=True)

# ==================================================================
# PAGE 3: Scenario Heatmap
# ==================================================================

elif page == "Scenario Heatmap":
    st.header("Scenario Analysis Heatmap")
    st.markdown(
        "Visualise PnL impact of parallel rate shifts on a portfolio of bonds."
    )

    col_setup, col_heat = st.columns([1, 2])

    with col_setup:
        st.subheader("Portfolio")
        n_bonds = st.number_input("Number of bonds", value=3, min_value=1, max_value=10)
        bonds = []
        for i in range(int(n_bonds)):
            with st.expander(f"Bond {i + 1}", expanded=(i == 0)):
                c = st.number_input(
                    f"Coupon {i + 1} (%)", value=4.0 + i, step=0.5, key=f"sc_{i}"
                )
                m = st.number_input(
                    f"Maturity {i + 1} (Y)", value=2.0 + i * 3, step=1.0, key=f"sm_{i}"
                )
                bonds.append(
                    FixedCouponBond(face=100, coupon_rate=c / 100, maturity=m, frequency=2)
                )

        shift_range = st.slider(
            "Shift range (bps)", min_value=10, max_value=300, value=100, step=10
        )

    shifts = list(range(-shift_range, shift_range + 1, max(shift_range // 10, 1)))

    # Use a flat 5 % zero rate as base curve for scenario analysis
    base_zr = lambda t: 0.05

    analyzer = ScenarioAnalyzer(base_zr)
    grid = analyzer.heatmap_grid(bonds, shifts)

    with col_heat:
        labels_bonds = [f"Bond {i + 1}" for i in range(len(bonds))]
        df_heat = pd.DataFrame(grid, columns=[f"{s}bp" for s in shifts], index=labels_bonds)
        st.subheader("PnL Heatmap (rate shift in bps)")
        st.dataframe(df_heat.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

        # Line chart of PnL per bond
        st.subheader("PnL by Shift")
        chart_df = pd.DataFrame(
            {f"Bond {i + 1}": grid[i] for i in range(len(bonds))},
            index=shifts,
        )
        chart_df.index.name = "Shift (bps)"
        st.line_chart(chart_df)

# ==================================================================
# PAGE 4: Cash Flow Waterfall
# ==================================================================

elif page == "Cash Flow Waterfall":
    st.header("Cash Flow Waterfall")

    col_cfg, col_chart = st.columns([1, 2])

    with col_cfg:
        face = st.number_input("Face Value", value=100.0, step=1.0, key="wf_face")
        coupon = st.number_input(
            "Coupon Rate (%)", value=5.0, step=0.1, format="%.2f", key="wf_coupon"
        )
        maturity = st.number_input(
            "Maturity (years)", value=5.0, step=0.5, key="wf_mat"
        )
        freq = st.selectbox("Coupon Frequency", [1, 2, 4], index=1, key="wf_freq")

    bond = FixedCouponBond(
        face=face,
        coupon_rate=coupon / 100.0,
        maturity=maturity,
        frequency=freq,
    )
    cfs = bond.cash_flows()

    with col_chart:
        cf_df = pd.DataFrame(
            {
                "Period (Y)": [cf.time for cf in cfs],
                "Cash Flow": [cf.amount for cf in cfs],
            }
        )
        st.bar_chart(cf_df, x="Period (Y)", y="Cash Flow")

        st.subheader("Cash Flow Schedule")
        cf_df["Cumulative"] = cf_df["Cash Flow"].cumsum()
        st.dataframe(cf_df, use_container_width=True)

        st.metric("Total Cash Flows", f"{sum(cf.amount for cf in cfs):.2f}")
