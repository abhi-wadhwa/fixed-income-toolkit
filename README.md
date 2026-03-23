# Fixed Income Analytics Toolkit

A comprehensive Python toolkit for fixed income analysis: yield curve bootstrapping, bond pricing, duration/convexity risk metrics, scenario analysis, and Nelson-Siegel-Svensson curve fitting.

[![CI](https://github.com/abhi-wadhwa/fixed-income-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/abhi-wadhwa/fixed-income-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Yield Curve Bootstrapping** -- Build discount curves from deposits, FRAs, and interest rate swaps
- **Nelson-Siegel-Svensson Model** -- Parametric curve fitting with 6-parameter NSS model
- **Bond Pricing** -- Fixed-coupon bond pricing with multiple day count conventions (30/360, ACT/360, ACT/365, ACT/ACT)
- **Risk Metrics** -- Macaulay duration, modified duration, effective duration, convexity, DV01, key rate durations
- **Scenario Analysis** -- Parallel shift, twist, butterfly, and custom rate scenarios with PnL impact
- **Portfolio Analytics** -- Portfolio-level duration, immunization, cash flow aggregation
- **Interactive Dashboard** -- Streamlit UI with yield curve plotter, bond calculator, scenario heatmap, and cash flow waterfall

---

## Theory

### Yield Curve Bootstrapping

The yield curve is the fundamental building block of fixed income pricing. We bootstrap discount factors $D(t)$ sequentially from liquid market instruments.

**Deposits** (short end): The discount factor is derived from the simple interest rate:

$$D(T) = \frac{1}{1 + r \cdot T}$$

**Forward Rate Agreements**: Using the no-arbitrage relationship:

$$D(T_2) = \frac{D(T_1)}{1 + r_{FRA} \cdot (T_2 - T_1)}$$

**Interest Rate Swaps**: A par swap has $PV_{fixed} = PV_{float} = 1$. For annual fixed payments:

$$1 = \sum_{i=1}^{n} c \cdot D(t_i) + D(T)$$

$$\Rightarrow D(T) = \frac{1 - c \cdot \sum_{i=1}^{n-1} D(t_i)}{1 + c}$$

where $c$ is the swap rate and $D(t_i)$ are previously bootstrapped discount factors.

### Interpolation

**Continuously compounded zero rate**: Given discount factors, we extract the zero rate:

$$z(t) = -\frac{\ln D(t)}{t}, \qquad D(t) = e^{-z(t) \cdot t}$$

We then interpolate $z(t)$ using either:
- **Cubic spline** with natural boundary conditions
- **Nelson-Siegel-Svensson** parametric model

### Nelson-Siegel-Svensson Model

The NSS model parameterises the zero rate curve with 6 parameters $(\beta_0, \beta_1, \beta_2, \beta_3, \tau_1, \tau_2)$:

$$z(t) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-t/\tau_1}}{t/\tau_1} + \beta_2 \cdot \left(\frac{1 - e^{-t/\tau_1}}{t/\tau_1} - e^{-t/\tau_1}\right) + \beta_3 \cdot \left(\frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2}\right)$$

The instantaneous forward rate is:

$$f(t) = \beta_0 + \beta_1 \cdot e^{-t/\tau_1} + \beta_2 \cdot \frac{t}{\tau_1} \cdot e^{-t/\tau_1} + \beta_3 \cdot \frac{t}{\tau_2} \cdot e^{-t/\tau_2}$$

**Parameter interpretation:**
| Parameter | Interpretation |
|-----------|---------------|
| $\beta_0$ | Long-term rate level (asymptote as $t \to \infty$) |
| $\beta_1$ | Short-term component ($\beta_0 + \beta_1$ is the instantaneous short rate) |
| $\beta_2$ | Medium-term hump/trough (curvature at $\tau_1$) |
| $\beta_3$ | Second hump (curvature at $\tau_2$) |
| $\tau_1$ | Decay factor for short/medium-term components |
| $\tau_2$ | Decay factor for the second hump |

Parameters are fitted via least squares minimisation:

$$\min_{\theta} \sum_{i} w_i \cdot \left(z_{model}(t_i; \theta) - z_{market}(t_i)\right)^2$$

### Bond Pricing

A fixed-coupon bond with face value $F$, coupon rate $c$, and $n$ coupon payments per year is priced as:

$$P = \sum_{i=1}^{N} \frac{c \cdot F}{n} \cdot D(t_i) + F \cdot D(T)$$

where $D(t_i)$ is the discount factor at each payment date.

**Yield to maturity** $y$ is the single rate that equates the price to the discounted cash flows:

$$P = \sum_{i=1}^{N} \frac{C}{(1 + y/n)^{n \cdot t_i}} + \frac{F}{(1 + y/n)^{n \cdot T}}$$

### Duration and Convexity

**Macaulay Duration** -- The weighted-average time to receive the bond's cash flows:

$$D_{mac} = \frac{1}{P} \sum_{i=1}^{N} t_i \cdot CF_i \cdot D(t_i)$$

**Modified Duration** -- Sensitivity of price to yield changes:

$$D_{mod} = \frac{D_{mac}}{1 + y/n}$$

**Effective Duration** -- Numerical (model-free) sensitivity via central differences:

$$D_{eff} = \frac{P_{-\Delta y} - P_{+\Delta y}}{2 \cdot \Delta y \cdot P_0}$$

**Convexity** -- Second-order sensitivity:

$$C = \frac{1}{P} \sum_{i=1}^{N} t_i^2 \cdot CF_i \cdot D(t_i)$$

**Price approximation** using Taylor expansion:

$$\Delta P \approx -D_{mod} \cdot P \cdot \Delta y + \frac{1}{2} \cdot C \cdot P \cdot (\Delta y)^2$$

**DV01** (Dollar Value of a Basis Point):

$$DV01 = D_{mod} \cdot P \cdot 0.0001$$

**Key Rate Duration** -- Sensitivity to shifts at specific maturities using triangular bump functions.

### Scenario Analysis

Scenarios model how the curve could change and compute the resulting PnL:

| Scenario | Description |
|----------|-------------|
| **Parallel** | Uniform shift: $z'(t) = z(t) + \Delta$ |
| **Twist** | Linear: $z'(t) = z(t) + \Delta_s + \frac{t}{T_p}(\Delta_l - \Delta_s)$ |
| **Butterfly** | Bell-shaped: belly moves differently from wings |
| **Custom** | Arbitrary shift vector interpolated across maturities |

---

## Project Structure

```
fixed-income-toolkit/
├── README.md
├── Makefile
├── pyproject.toml
├── Dockerfile
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── curve.py           # Yield curve bootstrapping
│   │   ├── nss.py             # Nelson-Siegel-Svensson model
│   │   ├── bond.py            # Bond pricing
│   │   ├── risk.py            # Duration, convexity, DV01
│   │   ├── scenario.py        # Scenario analysis
│   │   ├── day_count.py       # Day count conventions
│   │   └── portfolio.py       # Portfolio analytics
│   ├── viz/
│   │   ├── __init__.py
│   │   └── app.py             # Streamlit dashboard
│   └── cli.py                 # Command-line interface
├── tests/
│   ├── test_curve.py
│   ├── test_bond.py
│   ├── test_risk.py
│   ├── test_nss.py
│   └── test_day_count.py
├── examples/
│   └── demo.py
└── LICENSE
```

---

## Installation

```bash
# Clone
git clone https://github.com/abhi-wadhwa/fixed-income-toolkit.git
cd fixed-income-toolkit

# Install
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Command-Line Interface

```bash
# Price a bond
python -m src.cli price --coupon 0.05 --maturity 5 --ytm 0.05

# Compute risk metrics
python -m src.cli risk --coupon 0.05 --maturity 5 --ytm 0.06

# Display bootstrapped yield curve
python -m src.cli curve

# Run parallel shift scenario analysis
python -m src.cli scenario --coupon 0.05 --maturity 5 --shift 100
```

### Interactive Dashboard

```bash
streamlit run src/viz/app.py
```

The dashboard provides four interactive pages:
1. **Yield Curve Plotter** -- Input deposit and swap rates, visualise zero rates, forward rates, and discount factors
2. **Bond Calculator** -- Enter bond parameters to get price, yield, and full risk metrics
3. **Scenario Heatmap** -- Build a portfolio and see PnL under a grid of parallel rate shifts
4. **Cash Flow Waterfall** -- Timeline of coupon and principal payments

### Python API

```python
from src.core.curve import InstrumentType, MarketInstrument, YieldCurve
from src.core.bond import FixedCouponBond
from src.core.risk import RiskMetrics

# Bootstrap a yield curve
instruments = [
    MarketInstrument(InstrumentType.DEPOSIT, maturity=0.25, rate=0.040),
    MarketInstrument(InstrumentType.DEPOSIT, maturity=0.50, rate=0.042),
    MarketInstrument(InstrumentType.DEPOSIT, maturity=1.00, rate=0.045),
    MarketInstrument(InstrumentType.SWAP, maturity=2.0, rate=0.046),
    MarketInstrument(InstrumentType.SWAP, maturity=5.0, rate=0.048),
    MarketInstrument(InstrumentType.SWAP, maturity=10.0, rate=0.050),
]
curve = YieldCurve(instruments)

# Price a bond from the curve
bond = FixedCouponBond(face=100, coupon_rate=0.05, maturity=5, frequency=2)
price = bond.price(curve.discount_factor)

# Compute risk metrics
rm = RiskMetrics(bond)
ytm = bond.yield_to_maturity(price)
report = rm.full_report(curve.discount_factor, ytm, frequency=2)
print(f"Duration: {report.macaulay_duration:.4f}")
print(f"DV01:     {report.dv01:.6f}")
```

### Run Demo

```bash
python examples/demo.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Docker

```bash
# Build
docker build -t fixed-income-toolkit .

# Run dashboard
docker run -p 8501:8501 fixed-income-toolkit
```

---

## License

[MIT](LICENSE)
