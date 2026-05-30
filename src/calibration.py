"""
Classifier calibration analysis and error propagation.

Calibration measures the agreement between a classifier's stated confidence
scores and its empirical accuracy. A well-calibrated system where the model
says "70% confidence" should be right ~70% of the time.

Key insight: a 5% miscalibration in confidence → potentially 50%+ error
in prevalence estimates at low base rates. This module quantifies and
corrects that error.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Output of a calibration analysis.

    Attributes
    ----------
    ece : float
        Expected Calibration Error (probability-weighted mean |accuracy - confidence|).
    mce : float
        Maximum Calibration Error (worst-case bin).
    bin_accuracies : np.ndarray
        Empirical accuracy per confidence bin.
    bin_confidences : np.ndarray
        Mean predicted confidence per bin.
    bin_counts : np.ndarray
        Number of items per confidence bin.
    method : str
        Calibration method label.
    """

    ece: float
    mce: float
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    method: str = "uncalibrated"

    def summary(self) -> str:
        return (
            f"[{self.method}] ECE={self.ece:.4f}, MCE={self.mce:.4f}"
        )


class ClassifierCalibrator:
    """
    Calibration analysis and recalibration for harm detection systems.

    Supports:
    - Reliability diagram computation
    - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    - Platt scaling (logistic recalibration)
    - Isotonic regression recalibration
    - Error propagation from calibration error to prevalence CI

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins for reliability diagram (default 10).
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._platt_model: LogisticRegression | None = None
        self._isotonic_model: IsotonicRegression | None = None

    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> CalibrationResult:
        """
        Compute reliability diagram data and calibration error metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels (0/1).
        y_prob : np.ndarray
            Predicted probabilities in [0, 1].

        Returns
        -------
        CalibrationResult
            Calibration metrics and per-bin statistics.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        bin_accuracies, bin_confidences = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy="uniform"
        )

        # Recompute bin counts manually (sklearn doesn't expose them directly)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1])
        bin_counts = np.bincount(bin_ids, minlength=self.n_bins)

        # Ensure lengths match after sklearn may drop empty bins
        n_bins_used = len(bin_accuracies)
        bin_counts_used = bin_counts[:n_bins_used]

        ece = self.expected_calibration_error(y_true, y_prob)
        mce = float(np.max(np.abs(bin_accuracies - bin_confidences)))

        return CalibrationResult(
            ece=ece,
            mce=mce,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts_used,
            method="uncalibrated",
        )

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        .. math::

            \\text{ECE} = \\sum_{b=1}^{B} \\frac{|B_b|}{n}
                          \\left| \\text{acc}(B_b) - \\text{conf}(B_b) \\right|

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_prob : np.ndarray
            Predicted probabilities.

        Returns
        -------
        float
            ECE in [0, 1]. Lower is better. Values above 0.05 indicate
            meaningful miscalibration that will distort prevalence estimates.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / n
            ece += bin_weight * abs(bin_acc - bin_conf)

        return float(ece)

    def platt_scale(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """
        Fit Platt scaling (logistic recalibration) and return calibrated scores.

        Fits a logistic regression on the log-odds of y_prob to map raw scores
        to calibrated probabilities. Should be fit on a held-out validation set
        separate from the training data.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the calibration set.
        y_prob : np.ndarray
            Raw classifier probabilities to calibrate.

        Returns
        -------
        np.ndarray
            Calibrated probability scores.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        # Clip to avoid log(0)
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)

        self._platt_model = LogisticRegression(C=1e10)  # no regularization
        self._platt_model.fit(log_odds, y_true)
        calibrated = self._platt_model.predict_proba(log_odds)[:, 1]

        ece_before = self.expected_calibration_error(y_true, y_prob)
        ece_after = self.expected_calibration_error(y_true, calibrated)
        logger.info("Platt scaling: ECE %.4f → %.4f", ece_before, ece_after)

        return calibrated

    def platt_transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply a fitted Platt scaler to new scores.

        Parameters
        ----------
        y_prob : np.ndarray
            Raw classifier probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.

        Raises
        ------
        RuntimeError
            If :meth:`platt_scale` has not been called yet.
        """
        if self._platt_model is None:
            raise RuntimeError("Call platt_scale() first to fit the model.")
        y_prob_clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        return self._platt_model.predict_proba(log_odds)[:, 1]

    def isotonic_regression(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """
        Fit isotonic regression calibration and return calibrated scores.

        More flexible than Platt scaling — models arbitrary monotone mappings
        from raw scores to calibrated probabilities. Requires larger calibration
        sets (≥ 1000 samples) to avoid overfitting.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the calibration set.
        y_prob : np.ndarray
            Raw classifier probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probability scores.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        calibrated = self._isotonic_model.fit_transform(y_prob, y_true)

        ece_before = self.expected_calibration_error(y_true, y_prob)
        ece_after = self.expected_calibration_error(y_true, calibrated)
        logger.info("Isotonic regression: ECE %.4f → %.4f", ece_before, ece_after)

        return calibrated

    def isotonic_transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply a fitted isotonic regression model to new scores.

        Raises
        ------
        RuntimeError
            If :meth:`isotonic_regression` has not been called yet.
        """
        if self._isotonic_model is None:
            raise RuntimeError("Call isotonic_regression() first to fit the model.")
        return self._isotonic_model.predict(np.asarray(y_prob, dtype=float))

    def propagate_calibration_error_to_prevalence(
        self,
        calibration_error: float,
        prevalence_estimate: float,
        tpr: float,
        fpr: float,
        n_sample: int,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Propagate calibration uncertainty into prevalence CI widening.

        A classifier with ECE = ε introduces additional uncertainty into the
        adjusted prevalence estimate. This method quantifies how much wider
        the CI must be to account for calibration error.

        The key insight: at low base rates, calibration error is amplified
        dramatically. For π ≈ 0.01 and (TPR - FPR) ≈ 0.8, a 5% ECE causes
        the effective FPR uncertainty to expand the CI by a factor of 3–10×.

        Parameters
        ----------
        calibration_error : float
            ECE of the detection system (in [0, 1]).
        prevalence_estimate : float
            Point estimate of true prevalence from the adjusted estimator.
        tpr : float
            Detection system true positive rate.
        fpr : float
            Detection system false positive rate.
        n_sample : int
            Sample size.
        confidence_level : float
            CI coverage (default 0.95).

        Returns
        -------
        dict
            Keys: ``"ci_lower"``, ``"ci_upper"``, ``"ci_width_naive"``,
            ``"ci_width_calibration_adjusted"``, ``"amplification_factor"``.
        """
        from scipy import stats as scipy_stats

        z = scipy_stats.norm.ppf(1 - (1 - confidence_level) / 2)

        q = prevalence_estimate * (tpr - fpr) + fpr  # Expected observed rate
        denom = tpr - fpr

        if abs(denom) < 1e-8:
            raise ValueError("TPR - FPR is near zero: detection system is uninformative.")

        # Naive variance (sampling only)
        var_q_naive = q * (1 - q) / n_sample
        var_pi_naive = var_q_naive / denom**2
        ci_half_naive = z * math.sqrt(var_pi_naive)

        # Calibration-induced uncertainty: ECE translates to uncertainty in
        # both TPR and FPR. Model as:
        #   σ_tpr ≈ ECE / 2, σ_fpr ≈ ECE / 2
        sigma_tpr = calibration_error / 2
        sigma_fpr = calibration_error / 2

        d_pi_d_tpr = -(q - fpr) / denom**2
        d_pi_d_fpr = (tpr - q) / denom**2
        var_calib = d_pi_d_tpr**2 * sigma_tpr**2 + d_pi_d_fpr**2 * sigma_fpr**2

        var_pi_total = var_pi_naive + var_calib
        ci_half_total = z * math.sqrt(var_pi_total)

        amplification = ci_half_total / ci_half_naive if ci_half_naive > 0 else float("inf")

        result = {
            "prevalence_estimate": prevalence_estimate,
            "ci_lower": max(0.0, prevalence_estimate - ci_half_total),
            "ci_upper": min(1.0, prevalence_estimate + ci_half_total),
            "ci_width_naive": 2 * ci_half_naive,
            "ci_width_calibration_adjusted": 2 * ci_half_total,
            "amplification_factor": amplification,
            "calibration_error_ece": calibration_error,
        }

        logger.info(
            "Calibration error propagation: ECE=%.3f → CI amplification ×%.2f "
            "(%.4f%% → %.4f%% width)",
            calibration_error,
            amplification,
            100 * 2 * ci_half_naive,
            100 * 2 * ci_half_total,
        )

        return result

    def calibration_sensitivity_analysis(
        self,
        prevalence: float,
        tpr: float,
        fpr: float,
        n_sample: int,
        ece_values: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Sensitivity analysis: CI amplification as a function of ECE.

        Parameters
        ----------
        prevalence : float
            True prevalence point estimate.
        tpr : float
            Detection system TPR.
        fpr : float
            Detection system FPR.
        n_sample : int
            Sample size.
        ece_values : np.ndarray, optional
            ECE values to sweep. Defaults to [0, 0.02, 0.05, 0.10, 0.15, 0.20].

        Returns
        -------
        pd.DataFrame
            One row per ECE value with CI metrics.
        """
        if ece_values is None:
            ece_values = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20])

        records = []
        for ece in ece_values:
            result = self.propagate_calibration_error_to_prevalence(
                calibration_error=ece,
                prevalence_estimate=prevalence,
                tpr=tpr,
                fpr=fpr,
                n_sample=n_sample,
            )
            records.append(
                {
                    "ece": ece,
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "ci_width": result["ci_width_calibration_adjusted"],
                    "amplification_factor": result["amplification_factor"],
                }
            )

        return pd.DataFrame(records)
