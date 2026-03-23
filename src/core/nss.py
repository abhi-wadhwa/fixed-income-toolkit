"""Nelson-Siegel-Svensson yield curve model.

The NSS model parameterises the instantaneous forward rate curve as:

    f(t) = beta0
         + beta1 * exp(-t/tau1)
         + beta2 * (t/tau1) * exp(-t/tau1)
         + beta3 * (t/tau2) * exp(-t/tau2)

Integrating gives the zero rate:

    z(t) = beta0
         + beta1 * (1 - exp(-t/tau1)) / (t/tau1)
         + beta2 * ((1 - exp(-t/tau1)) / (t/tau1) - exp(-t/tau1))
         + beta3 * ((1 - exp(-t/tau2)) / (t/tau2) - exp(-t/tau2))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class NSSParams:
    """Nelson-Siegel-Svensson parameters."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float


class NelsonSiegelSvensson:
    """Fit and evaluate the Nelson-Siegel-Svensson model.

    Parameters
    ----------
    params : NSSParams, optional
        If provided the model is ready for evaluation without fitting.
    """

    def __init__(self, params: Optional[NSSParams] = None) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_rate_vector(t: np.ndarray, p: NSSParams) -> np.ndarray:
        """Vectorised zero rate evaluation."""
        t = np.asarray(t, dtype=float)
        # Guard against t == 0 (limit as t -> 0 is beta0 + beta1)
        safe_t = np.where(t == 0, 1e-10, t)

        x1 = safe_t / p.tau1
        x2 = safe_t / p.tau2

        factor1 = (1.0 - np.exp(-x1)) / x1
        factor2 = (1.0 - np.exp(-x2)) / x2

        z = (
            p.beta0
            + p.beta1 * factor1
            + p.beta2 * (factor1 - np.exp(-x1))
            + p.beta3 * (factor2 - np.exp(-x2))
        )
        return z

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return continuously compounded zero rate(s) at maturity *t* (years)."""
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        result = self._zero_rate_vector(np.atleast_1d(t), self.params)
        if np.ndim(t) == 0:
            return float(result[0])
        return result

    def discount_factor(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return discount factor(s) D(t) = exp(-z(t)*t)."""
        t_arr = np.atleast_1d(t)
        z = self.zero_rate(t_arr)
        df = np.exp(-z * t_arr)
        if np.ndim(t) == 0:
            return float(df[0])
        return df

    def forward_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return instantaneous forward rate(s) f(t)."""
        if self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        p = self.params
        t_arr = np.atleast_1d(t).astype(float)
        safe_t = np.where(t_arr == 0, 1e-10, t_arr)

        x1 = safe_t / p.tau1
        x2 = safe_t / p.tau2

        f = (
            p.beta0
            + p.beta1 * np.exp(-x1)
            + p.beta2 * x1 * np.exp(-x1)
            + p.beta3 * x2 * np.exp(-x2)
        )
        if np.ndim(t) == 0:
            return float(f[0])
        return f

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        maturities: np.ndarray,
        zero_rates: np.ndarray,
        weights: Optional[np.ndarray] = None,
        initial_guess: Optional[Dict[str, float]] = None,
    ) -> NSSParams:
        """Fit NSS parameters to observed zero rates.

        Parameters
        ----------
        maturities : array-like
            Observed maturities in years (> 0).
        zero_rates : array-like
            Observed continuously compounded zero rates.
        weights : array-like, optional
            Weights for weighted least squares.
        initial_guess : dict, optional
            Starting values for the optimiser.

        Returns
        -------
        NSSParams
            Fitted parameters (also stored in ``self.params``).
        """
        maturities = np.asarray(maturities, dtype=float)
        zero_rates = np.asarray(zero_rates, dtype=float)
        if weights is None:
            weights = np.ones_like(maturities)
        else:
            weights = np.asarray(weights, dtype=float)

        # Default initial guess
        if initial_guess is None:
            initial_guess = {}
        x0 = np.array(
            [
                initial_guess.get("beta0", zero_rates[-1]),
                initial_guess.get("beta1", zero_rates[0] - zero_rates[-1]),
                initial_guess.get("beta2", 0.0),
                initial_guess.get("beta3", 0.0),
                initial_guess.get("tau1", 1.0),
                initial_guess.get("tau2", 5.0),
            ]
        )

        def objective(x: np.ndarray) -> float:
            p = NSSParams(
                beta0=x[0],
                beta1=x[1],
                beta2=x[2],
                beta3=x[3],
                tau1=max(x[4], 0.01),
                tau2=max(x[5], 0.01),
            )
            model_rates = self._zero_rate_vector(maturities, p)
            residuals = weights * (model_rates - zero_rates) ** 2
            return float(np.sum(residuals))

        bounds = [
            (None, None),  # beta0
            (None, None),  # beta1
            (None, None),  # beta2
            (None, None),  # beta3
            (0.01, 30.0),  # tau1
            (0.01, 30.0),  # tau2
        ]

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-15},
        )

        self.params = NSSParams(
            beta0=result.x[0],
            beta1=result.x[1],
            beta2=result.x[2],
            beta3=result.x[3],
            tau1=max(result.x[4], 0.01),
            tau2=max(result.x[5], 0.01),
        )
        return self.params
