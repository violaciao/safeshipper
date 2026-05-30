"""
Threshold-level classification metrics for harm detection system evaluation.

Computes standard binary classification metrics at a given decision threshold,
and sweeps thresholds to generate ROC / precision-recall curves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """
    Classification metrics at a single decision threshold.

    Attributes
    ----------
    threshold : float
        Decision threshold applied to classifier confidence scores.
    tp : int
        True positives (correct harmful detections).
    fp : int
        False positives (benign items flagged as harmful).
    tn : int
        True negatives (correct benign classifications).
    fn : int
        False negatives (harmful items missed).
    precision : float
        TP / (TP + FP). Fraction of flags that are truly harmful.
    recall : float
        TP / (TP + FN). Fraction of harmful items detected (TPR).
    fpr : float
        FP / (FP + TN). False positive rate.
    fnr : float
        FN / (FN + TP). False negative rate (1 - recall).
    f1 : float
        Harmonic mean of precision and recall.
    specificity : float
        TN / (TN + FP). True negative rate.
    """

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    fpr: float
    fnr: float
    f1: float
    specificity: float

    @classmethod
    def from_confusion(
        cls, threshold: float, tp: int, fp: int, tn: int, fn: int
    ) -> "ThresholdMetrics":
        """Construct metrics from raw confusion matrix counts."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return cls(
            threshold=threshold,
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=precision,
            recall=recall,
            fpr=fpr,
            fnr=fnr,
            f1=f1,
            specificity=specificity,
        )


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ThresholdMetrics:
    """
    Compute binary classification metrics at a given threshold.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        Ground-truth binary labels (0/1 or bool).
    y_score : np.ndarray of shape (n,)
        Classifier confidence scores in [0, 1].
    threshold : float
        Decision threshold. Items with score >= threshold are flagged.

    Returns
    -------
    ThresholdMetrics
        Metrics at the specified threshold.

    Raises
    ------
    ValueError
        If y_true and y_score have different lengths, or if y_true has no
        positive or negative examples.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(f"y_true length {len(y_true)} != y_score length {len(y_score)}")
    if y_true.sum() == 0:
        raise ValueError("y_true contains no positive examples — cannot compute recall.")
    if (~y_true).sum() == 0:
        raise ValueError("y_true contains no negative examples — cannot compute FPR.")

    y_pred = y_score >= threshold
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())

    return ThresholdMetrics.from_confusion(threshold, tp, fp, tn, fn)


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """
    Sweep decision thresholds to produce a full metrics profile.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_score : np.ndarray
        Classifier confidence scores.
    thresholds : np.ndarray, optional
        Explicit threshold values to evaluate. If None, uses ``n_thresholds``
        evenly spaced values from the score range.
    n_thresholds : int
        Number of thresholds to evaluate when ``thresholds`` is None.

    Returns
    -------
    pd.DataFrame
        One row per threshold with all ThresholdMetrics fields plus AUC-ROC
        (constant column for reference).
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)

    records = []
    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_score, threshold=t)
        records.append(
            {
                "threshold": m.threshold,
                "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
                "precision": m.precision,
                "recall": m.recall,
                "fpr": m.fpr,
                "fnr": m.fnr,
                "f1": m.f1,
                "specificity": m.specificity,
            }
        )

    df = pd.DataFrame(records)

    try:
        auc = roc_auc_score(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        df["auc_roc"] = auc
        df["avg_precision"] = avg_precision
    except Exception as exc:
        logger.warning("Could not compute AUC: %s", exc)
        df["auc_roc"] = float("nan")
        df["avg_precision"] = float("nan")

    return df


def best_threshold_by_f1(sweep_df: pd.DataFrame) -> ThresholdMetrics:
    """
    Select the threshold that maximizes F1 score from a sweep DataFrame.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of :func:`sweep_thresholds`.

    Returns
    -------
    ThresholdMetrics
        Metrics at the F1-optimal threshold.
    """
    best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    return ThresholdMetrics.from_confusion(
        threshold=best_row["threshold"],
        tp=int(best_row["tp"]),
        fp=int(best_row["fp"]),
        tn=int(best_row["tn"]),
        fn=int(best_row["fn"]),
    )


def compute_cost_sensitive_threshold(
    sweep_df: pd.DataFrame,
    cost_fn: float = 1.0,
    cost_fp: float = 0.1,
) -> float:
    """
    Select threshold that minimizes expected cost under asymmetric FN/FP costs.

    In integrity contexts, missing a harmful item (FN) typically costs more than
    a false flag (FP) that requires human review. Use this to calibrate the
    operating point explicitly.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of :func:`sweep_thresholds`.
    cost_fn : float
        Cost per false negative (missed harmful item).
    cost_fp : float
        Cost per false positive (incorrectly flagged benign item).

    Returns
    -------
    float
        Optimal decision threshold.
    """
    df = sweep_df.copy()
    df["cost"] = cost_fn * df["fn"] + cost_fp * df["fp"]
    optimal_threshold = float(df.loc[df["cost"].idxmin(), "threshold"])
    logger.info(
        "Cost-sensitive threshold: %.3f (cost_fn=%.2f, cost_fp=%.2f)",
        optimal_threshold, cost_fn, cost_fp
    )
    return optimal_threshold
