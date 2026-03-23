"""Tests for Nelson-Siegel-Svensson model."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.nss import NelsonSiegelSvensson, NSSParams


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def known_params() -> NSSParams:
    """Known NSS parameters (typical EUR curve shape)."""
    return NSSParams(
        beta0=0.05,
        beta1=-0.02,
        beta2=0.01,
        beta3=0.005,
        tau1=1.5,
        tau2=5.0,
    )


@pytest.fixture
def nss_model(known_params: NSSParams) -> NelsonSiegelSvensson:
    return NelsonSiegelSvensson(params=known_params)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestEvaluation:
    def test_long_end_converges_to_beta0(self, nss_model: NelsonSiegelSvensson) -> None:
        """As t -> infinity, z(t) -> beta0."""
        z_100 = nss_model.zero_rate(100.0)
        np.testing.assert_allclose(z_100, 0.05, atol=1e-4)

    def test_short_end(self, nss_model: NelsonSiegelSvensson, known_params: NSSParams) -> None:
        """As t -> 0, z(t) -> beta0 + beta1."""
        z_near_zero = nss_model.zero_rate(0.001)
        expected = known_params.beta0 + known_params.beta1
        np.testing.assert_allclose(z_near_zero, expected, atol=1e-2)

    def test_vectorised(self, nss_model: NelsonSiegelSvensson) -> None:
        """Vectorised call should return array."""
        ts = np.array([1.0, 2.0, 5.0, 10.0])
        zr = nss_model.zero_rate(ts)
        assert isinstance(zr, np.ndarray)
        assert len(zr) == 4

    def test_discount_factor_at_zero(self, nss_model: NelsonSiegelSvensson) -> None:
        """D(0) should be approximately 1."""
        df = nss_model.discount_factor(0.001)
        np.testing.assert_allclose(df, 1.0, atol=0.01)

    def test_discount_factor_decreasing(self, nss_model: NelsonSiegelSvensson) -> None:
        """Discount factors should be decreasing."""
        ts = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        dfs = nss_model.discount_factor(ts)
        assert np.all(np.diff(dfs) < 0)


class TestForwardRate:
    def test_forward_positive(self, nss_model: NelsonSiegelSvensson) -> None:
        """Forward rates should be positive for typical parameters."""
        ts = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        fwd = nss_model.forward_rate(ts)
        assert np.all(fwd > 0)

    def test_forward_long_end(self, nss_model: NelsonSiegelSvensson) -> None:
        """f(t) -> beta0 as t -> infinity."""
        f_100 = nss_model.forward_rate(100.0)
        np.testing.assert_allclose(f_100, 0.05, atol=1e-4)


class TestFitting:
    def test_exact_fit(self, known_params: NSSParams) -> None:
        """Fit should recover parameters from noise-free data."""
        true_model = NelsonSiegelSvensson(params=known_params)
        ts = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
        true_rates = true_model.zero_rate(ts)

        fitted = NelsonSiegelSvensson()
        fitted.fit(ts, true_rates)

        # Fitted rates should match true rates closely
        fitted_rates = fitted.zero_rate(ts)
        np.testing.assert_allclose(fitted_rates, true_rates, atol=5e-4)

    def test_fit_with_noise(self, known_params: NSSParams) -> None:
        """Fit should be robust to small noise."""
        true_model = NelsonSiegelSvensson(params=known_params)
        ts = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        rng = np.random.default_rng(42)
        noisy_rates = true_model.zero_rate(ts) + rng.normal(0, 0.0005, len(ts))

        fitted = NelsonSiegelSvensson()
        fitted.fit(ts, noisy_rates)

        fitted_rates = fitted.zero_rate(ts)
        residuals = np.abs(fitted_rates - noisy_rates)
        assert np.max(residuals) < 0.005, f"Max residual too large: {np.max(residuals)}"

    def test_unfitted_raises(self) -> None:
        """Evaluating an unfitted model should raise RuntimeError."""
        model = NelsonSiegelSvensson()
        with pytest.raises(RuntimeError):
            model.zero_rate(1.0)
