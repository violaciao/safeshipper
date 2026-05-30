"""
Tests for src/calibration.py — classifier calibration and error propagation.

Covers: ECE computation, Platt scaling, isotonic regression, reliability
diagram, and calibration-error-to-prevalence CI widening.
"""

import math

import numpy as np
import pytest

from src.calibration import CalibrationResult, ClassifierCalibrator


@pytest.fixture
def calibrator() -> ClassifierCalibrator:
    return ClassifierCalibrator(n_bins=10)


@pytest.fixture
def perfect_calibration_data():
    """Synthetic data where confidence ≈ accuracy (near-perfect calibration)."""
    rng = np.random.default_rng(42)
    n = 2000
    y_prob = rng.uniform(0, 1, size=n)
    y_true = rng.binomial(1, y_prob)
    return y_true, y_prob


@pytest.fixture
def miscalibrated_data():
    """Data where model is systematically overconfident (scores too high)."""
    rng = np.random.default_rng(99)
    n = 2000
    y_prob = rng.beta(5, 2, size=n)  # Skewed toward 1
    true_rate = 0.30
    y_true = rng.binomial(1, true_rate, size=n)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_low_ece(self, calibrator, perfect_calibration_data):
        y_true, y_prob = perfect_calibration_data
        ece = calibrator.expected_calibration_error(y_true, y_prob)
        assert ece < 0.05  # Near-zero ECE for perfectly calibrated data

    def test_miscalibrated_higher_ece(
        self, calibrator, perfect_calibration_data, miscalibrated_data
    ):
        _, y_prob_good = perfect_calibration_data
        y_true_bad, y_prob_bad = miscalibrated_data
        ece_good = calibrator.expected_calibration_error(
            perfect_calibration_data[0], y_prob_good
        )
        ece_bad = calibrator.expected_calibration_error(y_true_bad, y_prob_bad)
        assert ece_bad > ece_good

    def test_ece_bounded(self, calibrator):
        rng = np.random.default_rng(7)
        y_true = rng.binomial(1, 0.3, size=500)
        y_prob = rng.uniform(0, 1, size=500)
        ece = calibrator.expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_all_confident_wrong_high_ece(self, calibrator):
        """All predictions at 0.99 confidence, all wrong → high ECE."""
        n = 500
        y_true = np.zeros(n)  # All negative
        y_prob = np.full(n, 0.99)  # All predicted positive
        ece = calibrator.expected_calibration_error(y_true, y_prob)
        assert ece > 0.5


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


class TestReliabilityDiagram:
    def test_returns_calibration_result(self, calibrator, perfect_calibration_data):
        y_true, y_prob = perfect_calibration_data
        result = calibrator.reliability_diagram(y_true, y_prob)
        assert isinstance(result, CalibrationResult)

    def test_bin_lengths_consistent(self, calibrator, perfect_calibration_data):
        y_true, y_prob = perfect_calibration_data
        result = calibrator.reliability_diagram(y_true, y_prob)
        assert len(result.bin_accuracies) == len(result.bin_confidences)

    def test_mce_gte_ece(self, calibrator, miscalibrated_data):
        y_true, y_prob = miscalibrated_data
        result = calibrator.reliability_diagram(y_true, y_prob)
        assert result.mce >= result.ece - 1e-9  # MCE is worst-case bin

    def test_method_label(self, calibrator, perfect_calibration_data):
        y_true, y_prob = perfect_calibration_data
        result = calibrator.reliability_diagram(y_true, y_prob)
        assert result.method == "uncalibrated"

    def test_summary_string(self, calibrator, perfect_calibration_data):
        y_true, y_prob = perfect_calibration_data
        result = calibrator.reliability_diagram(y_true, y_prob)
        summary = result.summary()
        assert "ECE" in summary and "MCE" in summary


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------


class TestPlattScaling:
    def test_reduces_ece_on_miscalibrated(self, calibrator, miscalibrated_data):
        y_true, y_prob = miscalibrated_data
        ece_before = calibrator.expected_calibration_error(y_true, y_prob)
        calibrated = calibrator.platt_scale(y_true, y_prob)
        ece_after = calibrator.expected_calibration_error(y_true, calibrated)
        assert ece_after < ece_before

    def test_calibrated_scores_in_range(self, calibrator, miscalibrated_data):
        y_true, y_prob = miscalibrated_data
        calibrated = calibrator.platt_scale(y_true, y_prob)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_transform_requires_fit(self, calibrator):
        new_calibrator = ClassifierCalibrator()
        with pytest.raises(RuntimeError, match="platt_scale"):
            new_calibrator.platt_transform(np.array([0.5, 0.6]))

    def test_transform_after_fit(self, calibrator, miscalibrated_data):
        y_true, y_prob = miscalibrated_data
        calibrator.platt_scale(y_true, y_prob)
        new_scores = np.array([0.1, 0.5, 0.9])
        transformed = calibrator.platt_transform(new_scores)
        assert len(transformed) == 3
        assert all(0 <= s <= 1 for s in transformed)


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------


class TestIsotonicRegression:
    def test_reduces_ece(self, calibrator, miscalibrated_data):
        y_true, y_prob = miscalibrated_data
        ece_before = calibrator.expected_calibration_error(y_true, y_prob)
        calibrated = calibrator.isotonic_regression(y_true, y_prob)
        ece_after = calibrator.expected_calibration_error(y_true, calibrated)
        assert ece_after <= ece_before

    def test_output_is_monotone(self, calibrator, miscalibrated_data):
        """Isotonic regression must produce a monotone mapping."""
        y_true, y_prob = miscalibrated_data
        sorted_idx = np.argsort(y_prob)
        calibrated = calibrator.isotonic_regression(y_true, y_prob)
        calibrated_sorted = calibrated[sorted_idx]
        diffs = np.diff(calibrated_sorted)
        assert (diffs >= -1e-9).all()  # Non-decreasing (allow float tolerance)

    def test_transform_requires_fit(self, calibrator):
        new_calibrator = ClassifierCalibrator()
        with pytest.raises(RuntimeError, match="isotonic_regression"):
            new_calibrator.isotonic_transform(np.array([0.5]))


# ---------------------------------------------------------------------------
# Calibration error propagation to prevalence
# ---------------------------------------------------------------------------


class TestPropagateCalibrationError:
    def test_zero_ece_no_amplification(self, calibrator):
        result = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.0,
            prevalence_estimate=0.02,
            tpr=0.85,
            fpr=0.05,
            n_sample=5000,
        )
        assert math.isclose(result["amplification_factor"], 1.0, abs_tol=1e-6)

    def test_nonzero_ece_amplifies_ci(self, calibrator):
        result = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.10,
            prevalence_estimate=0.01,
            tpr=0.85,
            fpr=0.05,
            n_sample=2000,
        )
        assert result["amplification_factor"] > 1.0
        assert result["ci_width_calibration_adjusted"] > result["ci_width_naive"]

    def test_low_prevalence_high_amplification(self, calibrator):
        """At low base rates, calibration error is amplified dramatically."""
        result_low = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.05,
            prevalence_estimate=0.001,  # 0.1% prevalence
            tpr=0.85,
            fpr=0.05,
            n_sample=5000,
        )
        result_high = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.05,
            prevalence_estimate=0.20,  # 20% prevalence
            tpr=0.85,
            fpr=0.05,
            n_sample=5000,
        )
        assert result_low["amplification_factor"] > result_high["amplification_factor"]

    def test_ci_bounds_in_unit_interval(self, calibrator):
        result = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.05,
            prevalence_estimate=0.01,
            tpr=0.80,
            fpr=0.02,
            n_sample=3000,
        )
        assert 0.0 <= result["ci_lower"] <= 1.0
        assert 0.0 <= result["ci_upper"] <= 1.0

    def test_tpr_equals_fpr_raises(self, calibrator):
        with pytest.raises(ValueError, match="uninformative"):
            calibrator.propagate_calibration_error_to_prevalence(
                calibration_error=0.05,
                prevalence_estimate=0.05,
                tpr=0.50,
                fpr=0.50,
                n_sample=1000,
            )

    def test_result_keys(self, calibrator):
        result = calibrator.propagate_calibration_error_to_prevalence(
            calibration_error=0.03,
            prevalence_estimate=0.05,
            tpr=0.85,
            fpr=0.03,
            n_sample=2000,
        )
        expected_keys = {
            "prevalence_estimate",
            "ci_lower",
            "ci_upper",
            "ci_width_naive",
            "ci_width_calibration_adjusted",
            "amplification_factor",
            "calibration_error_ece",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestCalibrationSensitivityAnalysis:
    def test_returns_dataframe(self, calibrator):
        import pandas as pd

        df = calibrator.calibration_sensitivity_analysis(
            prevalence=0.02, tpr=0.85, fpr=0.05, n_sample=3000
        )
        assert isinstance(df, pd.DataFrame)
        assert "ece" in df.columns
        assert "amplification_factor" in df.columns

    def test_amplification_monotone_in_ece(self, calibrator):
        df = calibrator.calibration_sensitivity_analysis(
            prevalence=0.01, tpr=0.85, fpr=0.05, n_sample=3000
        )
        # Amplification should be non-decreasing as ECE increases
        amp = df["amplification_factor"].values
        diffs = np.diff(amp)
        assert (diffs >= -1e-9).all()

    def test_custom_ece_values(self, calibrator):
        import numpy as np
        import pandas as pd

        custom = np.array([0.01, 0.05, 0.10])
        df = calibrator.calibration_sensitivity_analysis(
            prevalence=0.05, tpr=0.80, fpr=0.03, n_sample=2000, ece_values=custom
        )
        assert len(df) == 3
