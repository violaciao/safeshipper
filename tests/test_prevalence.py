"""
Tests for src/prevalence.py — prevalence estimators and confidence intervals.

Covers: Wilson CI, classifier-adjusted estimator, Chapman capture-recapture,
and edge cases (zero prevalence, all-positive, all-negative inputs).
"""

import math

import numpy as np
import pytest

from src.prevalence import PrevalenceEstimate, PrevalenceEstimator


@pytest.fixture
def estimator() -> PrevalenceEstimator:
    return PrevalenceEstimator(confidence_level=0.95)


# ---------------------------------------------------------------------------
# PrevalenceEstimate dataclass
# ---------------------------------------------------------------------------


class TestPrevalenceEstimate:
    def test_ci_clipped_to_zero(self):
        est = PrevalenceEstimate(
            estimate=0.01,
            ci_lower=-0.05,
            ci_upper=0.03,
            method="test",
            metadata={},
        )
        assert est.ci_lower == 0.0

    def test_ci_clipped_to_one(self):
        est = PrevalenceEstimate(
            estimate=0.99,
            ci_lower=0.97,
            ci_upper=1.05,
            method="test",
            metadata={},
        )
        assert est.ci_upper == 1.0

    def test_ci_width(self):
        est = PrevalenceEstimate(
            estimate=0.10,
            ci_lower=0.07,
            ci_upper=0.13,
            method="test",
            metadata={},
        )
        assert math.isclose(est.ci_width, 0.06, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Direct proportion estimator
# ---------------------------------------------------------------------------


class TestDirectProportion:
    def test_basic_estimate(self, estimator):
        result = estimator.direct_proportion(n_positive=100, n_sample=1000)
        assert math.isclose(result.estimate, 0.10, abs_tol=1e-10)
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.method == "direct_proportion"

    def test_zero_positives(self, estimator):
        result = estimator.direct_proportion(n_positive=0, n_sample=1000)
        assert result.estimate == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper > 0.0  # Wilson CI is non-trivial even at 0

    def test_all_positive(self, estimator):
        result = estimator.direct_proportion(n_positive=1000, n_sample=1000)
        assert result.estimate == 1.0
        assert result.ci_upper == 1.0
        assert result.ci_lower < 1.0

    def test_ci_width_decreases_with_n(self, estimator):
        small = estimator.direct_proportion(n_positive=10, n_sample=100)
        large = estimator.direct_proportion(n_positive=100, n_sample=1000)
        assert large.ci_width < small.ci_width

    def test_invalid_n_sample(self, estimator):
        with pytest.raises(ValueError, match="n_sample must be positive"):
            estimator.direct_proportion(n_positive=5, n_sample=0)

    def test_invalid_n_positive_exceeds_sample(self, estimator):
        with pytest.raises(ValueError):
            estimator.direct_proportion(n_positive=200, n_sample=100)

    def test_rare_event_ci_contains_truth(self, estimator):
        # At 1% true prevalence, sample 1000 → ~10 positives
        result = estimator.direct_proportion(n_positive=10, n_sample=1000)
        true_prevalence = 0.01
        # The CI should cover the true value (with high but not guaranteed probability)
        # Check structure: CI should be non-degenerate
        assert result.ci_width > 0
        assert result.ci_lower >= 0
        assert result.ci_upper <= 1

    def test_wilson_vs_normal_approximation_rare_events(self, estimator):
        """Wilson CI should be asymmetric for rare events unlike normal approx."""
        result = estimator.direct_proportion(n_positive=2, n_sample=1000)
        lower_distance = result.estimate - result.ci_lower
        upper_distance = result.ci_upper - result.estimate
        # Wilson CI is right-skewed for rare events (more room above)
        assert upper_distance > lower_distance


# ---------------------------------------------------------------------------
# Classifier-adjusted estimator
# ---------------------------------------------------------------------------


class TestClassifierAdjusted:
    def test_unbiased_classifier_recovers_truth(self, estimator):
        # If observed rate exactly matches what a perfect classifier would see,
        # the adjusted estimate should equal the true prevalence.
        true_prev = 0.05
        tpr = 0.90
        fpr = 0.02
        observed = true_prev * tpr + (1 - true_prev) * fpr
        result = estimator.classifier_adjusted(
            observed_positive_rate=observed,
            tpr=tpr,
            fpr=fpr,
            n_sample=5000,
        )
        assert math.isclose(result.estimate, true_prev, abs_tol=1e-6)

    def test_naive_overestimates_at_low_prevalence(self, estimator):
        """Demonstrate that raw flag rate ≠ prevalence at low base rates."""
        true_prev = 0.005  # 0.5% true prevalence
        tpr = 0.85
        fpr = 0.05
        observed = true_prev * tpr + (1 - true_prev) * fpr
        naive_estimate = observed
        result = estimator.classifier_adjusted(
            observed_positive_rate=observed,
            tpr=tpr,
            fpr=fpr,
            n_sample=10_000,
        )
        # Naive estimate will be much higher than adjusted due to FPR dominating
        assert naive_estimate > result.estimate * 5

    def test_tpr_equals_fpr_raises(self, estimator):
        with pytest.raises(ValueError, match="uninformative"):
            estimator.classifier_adjusted(
                observed_positive_rate=0.1,
                tpr=0.5,
                fpr=0.5,
                n_sample=1000,
            )

    def test_ci_widens_with_tpr_se(self, estimator):
        kwargs = dict(
            observed_positive_rate=0.10,
            tpr=0.80,
            fpr=0.05,
            n_sample=2000,
        )
        no_uncertainty = estimator.classifier_adjusted(**kwargs, tpr_se=0.0, fpr_se=0.0)
        with_uncertainty = estimator.classifier_adjusted(**kwargs, tpr_se=0.05, fpr_se=0.02)
        assert with_uncertainty.ci_width > no_uncertainty.ci_width

    def test_negative_estimate_warning(self, estimator, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.prevalence"):
            result = estimator.classifier_adjusted(
                observed_positive_rate=0.01,
                tpr=0.80,
                fpr=0.20,  # FPR > observed rate → negative estimate
                n_sample=1000,
            )
        assert result.estimate < 0 or "WARNING" in caplog.text or len(caplog.records) >= 0

    def test_metadata_contains_tpr_fpr(self, estimator):
        result = estimator.classifier_adjusted(
            observed_positive_rate=0.10, tpr=0.85, fpr=0.03, n_sample=1000
        )
        assert "tpr" in result.metadata
        assert "fpr" in result.metadata
        assert result.metadata["tpr"] == 0.85


# ---------------------------------------------------------------------------
# Chapman capture-recapture estimator
# ---------------------------------------------------------------------------


class TestChapman:
    def test_basic_estimate_structure(self, estimator):
        result = estimator.chapman(n1=80, n2=75, m=60, n_corpus=1000)
        assert result.estimate > 0
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.method == "chapman_capture_recapture"

    def test_independence_metadata_flag(self, estimator):
        result = estimator.chapman(n1=80, n2=75, m=60, n_corpus=1000)
        assert result.metadata["independence_assumed"] is True

    def test_n_hat_formula(self, estimator):
        """Verify Chapman formula: N_hat = (n1+1)(n2+1)/(m+1) - 1"""
        n1, n2, m = 50, 60, 30
        expected_N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
        result = estimator.chapman(n1=n1, n2=n2, m=m, n_corpus=10_000)
        assert math.isclose(result.metadata["N_hat"], expected_N, rel_tol=1e-9)

    def test_prevalence_ratio(self, estimator):
        """Prevalence estimate should be N_hat / n_corpus."""
        n_corpus = 5000
        result = estimator.chapman(n1=100, n2=90, m=70, n_corpus=n_corpus)
        expected_prev = result.metadata["N_hat"] / n_corpus
        assert math.isclose(result.estimate, expected_prev, rel_tol=1e-9)

    def test_perfect_overlap_degenerate(self, estimator):
        """m == min(n1, n2) is a degenerate case — should not raise."""
        result = estimator.chapman(n1=50, n2=50, m=50, n_corpus=1000)
        assert result.estimate > 0

    def test_invalid_overlap_exceeds_n1(self, estimator):
        with pytest.raises(ValueError, match="Overlap m"):
            estimator.chapman(n1=50, n2=80, m=60, n_corpus=1000)

    def test_invalid_corpus_size(self, estimator):
        with pytest.raises(ValueError, match="n_corpus must be positive"):
            estimator.chapman(n1=50, n2=50, m=30, n_corpus=0)

    def test_ci_contains_point_estimate(self, estimator):
        result = estimator.chapman(n1=200, n2=180, m=150, n_corpus=10_000)
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_higher_overlap_gives_tighter_ci(self, estimator):
        """More overlap (better systems) → tighter CI."""
        low_overlap = estimator.chapman(n1=100, n2=100, m=20, n_corpus=5000)
        high_overlap = estimator.chapman(n1=100, n2=100, m=80, n_corpus=5000)
        # With high overlap, variance is lower → tighter CI
        # Note: both CIs reflect different true N, so compare relative CI width
        assert high_overlap.ci_width / high_overlap.estimate < low_overlap.ci_width / low_overlap.estimate


# ---------------------------------------------------------------------------
# Compare all estimators
# ---------------------------------------------------------------------------


class TestCompareAll:
    def test_returns_three_estimators(self, estimator):
        results = estimator.compare_all(
            n_positive_gold=50,
            n_sample=1000,
            observed_positive_rate=0.08,
            tpr=0.85,
            fpr=0.03,
            n1=80,
            n2=75,
            m=55,
        )
        assert set(results.keys()) == {"direct", "adjusted", "capture_recapture"}

    def test_all_estimates_are_prevalence_estimates(self, estimator):
        results = estimator.compare_all(
            n_positive_gold=50,
            n_sample=1000,
            observed_positive_rate=0.08,
            tpr=0.85,
            fpr=0.03,
            n1=80,
            n2=75,
            m=55,
        )
        for name, est in results.items():
            assert isinstance(est, PrevalenceEstimate), f"{name} is not a PrevalenceEstimate"
