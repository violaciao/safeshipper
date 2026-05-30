"""
Harm prevalence estimators with confidence intervals.

Implements three estimators of increasing sophistication:

1. Direct proportion (baseline) with Wilson score CI.
2. Classifier-adjusted estimator with delta method variance propagation.
3. Chapman capture-recapture estimator with log-transformed CI.

The classifier-adjusted and capture-recapture estimators address the core
production problem: raw detection counts are not prevalence estimates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PrevalenceEstimate:
    """
    A prevalence estimate with confidence interval.

    Attributes
    ----------
    estimate : float
        Point estimate of prevalence (items per unit).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    ci_width : float
        Full width of the CI (upper - lower).
    method : str
        Estimation method used.
    metadata : dict
        Method-specific metadata (sample sizes, TPR/FPR, etc.).
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    method: str
    metadata: dict

    def __post_init__(self) -> None:
        self.ci_lower = max(0.0, self.ci_lower)
        self.ci_upper = min(1.0, self.ci_upper)

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def __repr__(self) -> str:
        return (
            f"PrevalenceEstimate(estimate={self.estimate:.4%}, "
            f"CI=[{self.ci_lower:.4%}, {self.ci_upper:.4%}], method={self.method!r})"
        )


class PrevalenceEstimator:
    """
    Unified interface for harm prevalence estimation.

    Implements direct proportion, classifier-adjusted, and capture-recapture
    estimators. All methods return :class:`PrevalenceEstimate` objects.

    Parameters
    ----------
    confidence_level : float
        Coverage probability for confidence intervals (default 0.95).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level
        self._alpha = 1 - confidence_level
        self._z = stats.norm.ppf(1 - self._alpha / 2)

    def direct_proportion(
        self,
        n_positive: int,
        n_sample: int,
    ) -> PrevalenceEstimate:
        """
        Direct proportion estimator with Wilson score confidence interval.

        The Wilson interval is preferred over the normal approximation for small
        samples and extreme proportions (which are common in harm detection).

        Parameters
        ----------
        n_positive : int
            Number of items labeled as harmful in the sample (from gold standard
            review or oracle labels).
        n_sample : int
            Total sample size.

        Returns
        -------
        PrevalenceEstimate
            Point estimate = n_positive / n_sample, with Wilson CI.

        Notes
        -----
        Wilson score interval formula:

        .. math::

            \\hat{p} = \\frac{k + z^2/2}{n + z^2}

            CI = \\hat{p} \\pm \\frac{z \\sqrt{n \\cdot p(1-p) + z^2/4}}{n + z^2}

        References
        ----------
        Wilson, E. B. (1927). Probable inference, the law of succession,
        and statistical inference. *JASA*, 22(158), 209–212.
        """
        if n_sample <= 0:
            raise ValueError("n_sample must be positive")
        if not (0 <= n_positive <= n_sample):
            raise ValueError(f"n_positive={n_positive} out of range [0, {n_sample}]")

        k = n_positive
        n = n_sample
        z = self._z
        z2 = z**2

        # Wilson point estimate (center of Wilson interval)
        p_tilde = (k + z2 / 2) / (n + z2)
        half_width = z * math.sqrt(n * (k / n) * (1 - k / n) + z2 / 4) / (n + z2)

        estimate = k / n
        ci_lower = p_tilde - half_width
        ci_upper = p_tilde + half_width

        logger.debug(
            "Direct estimator: n_positive=%d, n_sample=%d → %.4f%% [%.4f%%, %.4f%%]",
            k, n, 100 * estimate, 100 * ci_lower, 100 * ci_upper,
        )

        return PrevalenceEstimate(
            estimate=estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="direct_proportion",
            metadata={"n_positive": k, "n_sample": n},
        )

    def classifier_adjusted(
        self,
        observed_positive_rate: float,
        tpr: float,
        fpr: float,
        n_sample: int,
        tpr_se: float = 0.0,
        fpr_se: float = 0.0,
    ) -> PrevalenceEstimate:
        """
        Classifier-adjusted prevalence estimator with delta method CI.

        Corrects for imperfect detection by accounting for TPR and FPR:

        .. math::

            \\hat{\\pi} = \\frac{\\hat{q} - \\text{FPR}}{\\text{TPR} - \\text{FPR}}

        where :math:`\\hat{q}` is the observed positive rate from the detection system.

        This is the critical production insight: a naive count of flags from an
        imperfect detection system is a biased prevalence estimate. Even a
        well-performing system (95% TPR, 1% FPR) can produce dramatically
        wrong prevalence estimates at low base rates.

        Parameters
        ----------
        observed_positive_rate : float
            Fraction of items flagged by the detection system (q̂ = detections / n).
        tpr : float
            True positive rate of the detection system (estimated from gold standard).
        fpr : float
            False positive rate of the detection system.
        n_sample : int
            Number of items scored by the detection system.
        tpr_se : float
            Standard error of the TPR estimate (0 if assumed known).
        fpr_se : float
            Standard error of the FPR estimate (0 if assumed known).

        Returns
        -------
        PrevalenceEstimate
            Adjusted point estimate and delta-method CI.

        Raises
        ------
        ValueError
            If TPR == FPR (detection system is no better than random).

        Notes
        -----
        Delta method variance:

        .. math::

            \\text{Var}(\\hat{\\pi}) \\approx
                \\left(\\frac{\\partial \\pi}{\\partial q}\\right)^2 \\text{Var}(q) +
                \\left(\\frac{\\partial \\pi}{\\partial \\text{TPR}}\\right)^2 \\text{Var}(\\text{TPR}) +
                \\left(\\frac{\\partial \\pi}{\\partial \\text{FPR}}\\right)^2 \\text{Var}(\\text{FPR})
        """
        if abs(tpr - fpr) < 1e-8:
            raise ValueError(
                f"TPR ({tpr:.4f}) ≈ FPR ({fpr:.4f}): detection system is uninformative."
            )

        q = observed_positive_rate
        denom = tpr - fpr
        pi_hat = (q - fpr) / denom

        # Delta method variance components
        var_q = q * (1 - q) / n_sample  # Bernoulli sampling variance
        d_pi_d_q = 1.0 / denom
        d_pi_d_tpr = -(q - fpr) / denom**2
        d_pi_d_fpr = (tpr - q) / denom**2  # corrected sign

        var_pi = (
            d_pi_d_q**2 * var_q
            + d_pi_d_tpr**2 * tpr_se**2
            + d_pi_d_fpr**2 * fpr_se**2
        )
        se_pi = math.sqrt(max(var_pi, 0.0))

        ci_lower = pi_hat - self._z * se_pi
        ci_upper = pi_hat + self._z * se_pi

        if pi_hat < 0 or pi_hat > 1:
            logger.warning(
                "Adjusted estimate %.4f is outside [0, 1]. This may indicate "
                "model misspecification or TPR/FPR estimates from a different "
                "population than the target corpus.",
                pi_hat,
            )

        logger.debug(
            "Adjusted estimator: q=%.4f, TPR=%.3f, FPR=%.3f → π̂=%.4f%% [%.4f%%, %.4f%%]",
            q, tpr, fpr, 100 * pi_hat, 100 * ci_lower, 100 * ci_upper,
        )

        return PrevalenceEstimate(
            estimate=pi_hat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="classifier_adjusted",
            metadata={
                "observed_positive_rate": q,
                "tpr": tpr,
                "fpr": fpr,
                "n_sample": n_sample,
                "se_pi": se_pi,
            },
        )

    def chapman(
        self,
        n1: int,
        n2: int,
        m: int,
        n_corpus: int,
    ) -> PrevalenceEstimate:
        """
        Chapman capture-recapture estimator with log-transformed confidence interval.

        Estimates the total harmful item count using two independent detection
        systems as "capture occasions." Does not require ground-truth labels.

        .. math::

            \\hat{N} = \\frac{(n_1 + 1)(n_2 + 1)}{m + 1} - 1

        where:
        - :math:`n_1` = detections by system A
        - :math:`n_2` = detections by system B
        - :math:`m` = items detected by both (overlap)

        Prevalence estimate: :math:`\\hat{\\pi} = \\hat{N} / N_{\\text{corpus}}`

        Parameters
        ----------
        n1 : int
            Number of detections by system A.
        n2 : int
            Number of detections by system B.
        m : int
            Number of items detected by both systems.
        n_corpus : int
            Total corpus size (denominator for prevalence).

        Returns
        -------
        PrevalenceEstimate
            Chapman estimate with log-transformed CI.

        Notes
        -----
        **Independence assumption:** The Chapman estimator assumes the two
        detection systems are independent. In practice, two LLM-based systems
        using the same base model will be positively correlated, causing
        underestimation of N. This limitation should be acknowledged in reporting.

        Log-transformed CI (standard in ecology / epidemiology):

        .. math::

            CI_{\\log} = \\hat{N} \\cdot \\exp\\left(\\pm z \\cdot \\text{SE}_{\\log N}\\right)

        References
        ----------
        Chapman, D. G. (1951). Some properties of the hypergeometric distribution
        with applications to zoological censuses. *UC Publications in Statistics*.

        Seber, G. A. F. (1982). *The Estimation of Animal Abundance*.
        """
        if m > min(n1, n2):
            raise ValueError(
                f"Overlap m={m} cannot exceed min(n1, n2)={min(n1, n2)}"
            )
        if n_corpus <= 0:
            raise ValueError("n_corpus must be positive")

        # Chapman estimator
        N_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1

        # Log-transformed variance (Seber, 1982, eq. 3.15)
        # Var(log N_hat) ≈ (n1 - m)(n2 - m) / ((n1+1)(n2+1)(m+1))
        numerator_var = (n1 - m) * (n2 - m)
        denominator_var = (n1 + 1) * (n2 + 1) * (m + 1)

        if denominator_var == 0 or numerator_var < 0:
            # Degenerate case: all detections overlap
            logger.warning("Degenerate capture-recapture: m=%d, n1=%d, n2=%d", m, n1, n2)
            se_log_N = 0.0
        else:
            var_log_N = numerator_var / denominator_var
            se_log_N = math.sqrt(var_log_N)

        log_ci_factor = math.exp(self._z * se_log_N)
        N_lower = N_hat / log_ci_factor
        N_upper = N_hat * log_ci_factor

        pi_hat = N_hat / n_corpus
        pi_lower = N_lower / n_corpus
        pi_upper = N_upper / n_corpus

        logger.debug(
            "Chapman estimator: n1=%d, n2=%d, m=%d, N̂=%.1f → π̂=%.4f%% [%.4f%%, %.4f%%]",
            n1, n2, m, N_hat, 100 * pi_hat, 100 * pi_lower, 100 * pi_upper,
        )

        return PrevalenceEstimate(
            estimate=pi_hat,
            ci_lower=pi_lower,
            ci_upper=pi_upper,
            method="chapman_capture_recapture",
            metadata={
                "N_hat": N_hat,
                "N_ci_lower": N_lower,
                "N_ci_upper": N_upper,
                "n1": n1,
                "n2": n2,
                "m": m,
                "n_corpus": n_corpus,
                "independence_assumed": True,
            },
        )

    def compare_all(
        self,
        n_positive_gold: int,
        n_sample: int,
        observed_positive_rate: float,
        tpr: float,
        fpr: float,
        n1: int,
        n2: int,
        m: int,
        tpr_se: float = 0.0,
        fpr_se: float = 0.0,
    ) -> dict[str, PrevalenceEstimate]:
        """
        Run all three estimators and return a comparison dict.

        Parameters
        ----------
        n_positive_gold : int
            Gold-standard positive count (for direct estimator).
        n_sample : int
            Sample size.
        observed_positive_rate : float
            Detection system positive rate.
        tpr : float
            System A true positive rate.
        fpr : float
            System A false positive rate.
        n1 : int
            System A detections.
        n2 : int
            System B detections.
        m : int
            Overlap between systems.
        tpr_se : float
            Standard error of TPR.
        fpr_se : float
            Standard error of FPR.

        Returns
        -------
        dict[str, PrevalenceEstimate]
            Keyed by estimator name.
        """
        return {
            "direct": self.direct_proportion(n_positive_gold, n_sample),
            "adjusted": self.classifier_adjusted(
                observed_positive_rate, tpr, fpr, n_sample, tpr_se, fpr_se
            ),
            "capture_recapture": self.chapman(n1, n2, m, n_sample),
        }
