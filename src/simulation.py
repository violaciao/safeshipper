"""
Synthetic corpus generator for methodology validation.

Generates platform corpora with known true prevalence, simulates two independent
detection systems with configurable TPR/FPR, and validates estimator coverage
across repeated trials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """
    Configuration for a synthetic platform corpus simulation.

    Parameters
    ----------
    n_corpus : int
        Total number of items in the platform corpus.
    true_prevalence : float
        True proportion of harmful items in the corpus (0, 1).
    tpr_a : float
        True positive rate of detection system A (recall).
    fpr_a : float
        False positive rate of detection system A.
    tpr_b : float
        True positive rate of detection system B.
    fpr_b : float
        False positive rate of detection system B.
    random_seed : int
        Seed for reproducibility.
    harm_category : str
        Label for the harm vertical being simulated.
    """

    n_corpus: int = 100_000
    true_prevalence: float = 0.01
    tpr_a: float = 0.80
    fpr_a: float = 0.05
    tpr_b: float = 0.75
    fpr_b: float = 0.03
    random_seed: int = 42
    harm_category: str = "generic"

    def __post_init__(self) -> None:
        if not 0 < self.true_prevalence < 1:
            raise ValueError("true_prevalence must be in (0, 1)")
        if not 0 <= self.fpr_a < self.tpr_a <= 1:
            raise ValueError("Require 0 <= fpr_a < tpr_a <= 1 for system A to be informative")
        if not 0 <= self.fpr_b < self.tpr_b <= 1:
            raise ValueError("Require 0 <= fpr_b < tpr_b <= 1 for system B to be informative")


@dataclass
class CorpusSimulation:
    """
    Result of a synthetic corpus simulation.

    Attributes
    ----------
    corpus : pd.DataFrame
        Full corpus with true labels and detection outcomes.
    config : SimulationConfig
        The configuration used to generate this simulation.
    n_true_positives : int
        Ground-truth count of harmful items in the corpus.
    """

    corpus: pd.DataFrame
    config: SimulationConfig
    n_true_positives: int


def generate_corpus(config: SimulationConfig) -> CorpusSimulation:
    """
    Generate a synthetic platform corpus with known ground truth.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters.

    Returns
    -------
    CorpusSimulation
        Contains the full corpus DataFrame and metadata.

    Notes
    -----
    Each item in the corpus receives:
    - ``true_label``: Bernoulli draw with p = config.true_prevalence
    - ``score_a``, ``score_b``: Simulated classifier confidence scores
      drawn from Beta distributions conditioned on true label.
    - ``detected_a``, ``detected_b``: Binary detection outcomes from
      applying TPR/FPR to the true label.
    """
    rng = np.random.default_rng(config.random_seed)

    n = config.n_corpus
    true_labels = rng.binomial(1, config.true_prevalence, size=n).astype(bool)
    n_positives = int(true_labels.sum())

    logger.info(
        "Generated corpus: n=%d, true_positives=%d (%.4f%%)",
        n,
        n_positives,
        100 * n_positives / n,
    )

    # Simulate continuous classifier scores using Beta distributions.
    # Positives: high-scoring Beta(8, 2); negatives: low-scoring Beta(2, 8).
    # Scale TPR/FPR to modulate the separation.
    def _simulate_scores(tpr: float, fpr: float) -> np.ndarray:
        # The score distribution is a mixture:
        # positives ~ Beta(alpha_pos, beta_pos), negatives ~ Beta(alpha_neg, beta_neg)
        # We parameterize so that at threshold=0.5: detection rate ≈ tpr for positives
        # and false alarm rate ≈ fpr for negatives.
        alpha_pos = 6 * tpr + 1
        beta_pos = 6 * (1 - tpr) + 1
        alpha_neg = 6 * fpr + 1
        beta_neg = 6 * (1 - fpr) + 1

        scores = np.where(
            true_labels,
            rng.beta(alpha_pos, beta_pos, size=n),
            rng.beta(alpha_neg, beta_neg, size=n),
        )
        return scores

    scores_a = _simulate_scores(config.tpr_a, config.fpr_a)
    scores_b = _simulate_scores(config.tpr_b, config.fpr_b)

    # Apply Bernoulli draws for detection outcomes consistent with TPR/FPR
    detected_a = np.where(
        true_labels,
        rng.binomial(1, config.tpr_a, size=n).astype(bool),
        rng.binomial(1, config.fpr_a, size=n).astype(bool),
    )
    detected_b = np.where(
        true_labels,
        rng.binomial(1, config.tpr_b, size=n).astype(bool),
        rng.binomial(1, config.fpr_b, size=n).astype(bool),
    )

    corpus = pd.DataFrame(
        {
            "item_id": np.arange(n),
            "true_label": true_labels,
            "score_a": scores_a,
            "score_b": scores_b,
            "detected_a": detected_a.astype(bool),
            "detected_b": detected_b.astype(bool),
            "harm_category": config.harm_category,
        }
    )

    return CorpusSimulation(
        corpus=corpus,
        config=config,
        n_true_positives=n_positives,
    )


def simulate_detection_systems(
    sim: CorpusSimulation,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Extract detection outcomes from a simulated corpus for prevalence estimation.

    Parameters
    ----------
    sim : CorpusSimulation
        The simulated corpus.
    sample_size : int, optional
        If provided, draw a random sample of this size from the corpus.
        Otherwise returns the full corpus.

    Returns
    -------
    sample : pd.DataFrame
        The (possibly subsampled) detection data.
    true_params : dict[str, float]
        Ground-truth parameters: ``prevalence``, ``tpr_a``, ``fpr_a``,
        ``tpr_b``, ``fpr_b``.
    """
    corpus = sim.corpus

    if sample_size is not None:
        rng = np.random.default_rng(sim.config.random_seed + 1)
        idx = rng.choice(len(corpus), size=min(sample_size, len(corpus)), replace=False)
        corpus = corpus.iloc[idx].reset_index(drop=True)

    true_params = {
        "prevalence": sim.config.true_prevalence,
        "tpr_a": sim.config.tpr_a,
        "fpr_a": sim.config.fpr_a,
        "tpr_b": sim.config.tpr_b,
        "fpr_b": sim.config.fpr_b,
        "n_true_positives": sim.n_true_positives,
    }

    return corpus, true_params


def run_coverage_simulation(
    config: SimulationConfig,
    n_trials: int = 500,
    sample_size: int = 2000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Validate estimator coverage via repeated simulation.

    Runs ``n_trials`` independent simulations with the given config and checks
    whether each estimator's confidence interval contains the true prevalence.

    Parameters
    ----------
    config : SimulationConfig
        Base configuration. Random seed is incremented per trial.
    n_trials : int
        Number of Monte Carlo trials.
    sample_size : int
        Items drawn from each corpus for estimation.
    confidence_level : float
        Nominal coverage for CIs.

    Returns
    -------
    pd.DataFrame
        Per-trial results including point estimates, CI bounds, and coverage
        indicators for each estimator.
    """
    from .prevalence import PrevalenceEstimator  # avoid circular import

    records = []
    for trial in range(n_trials):
        trial_config = SimulationConfig(
            n_corpus=config.n_corpus,
            true_prevalence=config.true_prevalence,
            tpr_a=config.tpr_a,
            fpr_a=config.fpr_a,
            tpr_b=config.tpr_b,
            fpr_b=config.fpr_b,
            random_seed=config.random_seed + trial,
            harm_category=config.harm_category,
        )
        sim = generate_corpus(trial_config)
        sample, _ = simulate_detection_systems(sim, sample_size=sample_size)

        estimator = PrevalenceEstimator(confidence_level=confidence_level)

        # Direct estimator
        direct = estimator.direct_proportion(
            n_positive=int(sample["true_label"].sum()),
            n_sample=len(sample),
        )

        # Classifier-adjusted (using system A)
        adjusted = estimator.classifier_adjusted(
            observed_positive_rate=sample["detected_a"].mean(),
            tpr=config.tpr_a,
            fpr=config.fpr_a,
            n_sample=len(sample),
            tpr_se=0.02,
            fpr_se=0.01,
        )

        # Capture-recapture
        n1 = int(sample["detected_a"].sum())
        n2 = int(sample["detected_b"].sum())
        m = int((sample["detected_a"] & sample["detected_b"]).sum())
        cr = estimator.chapman(n1=n1, n2=n2, m=m, n_corpus=len(sample))

        π = config.true_prevalence
        records.append(
            {
                "trial": trial,
                "true_prevalence": π,
                "direct_estimate": direct.estimate,
                "direct_ci_lower": direct.ci_lower,
                "direct_ci_upper": direct.ci_upper,
                "direct_covered": direct.ci_lower <= π <= direct.ci_upper,
                "adjusted_estimate": adjusted.estimate,
                "adjusted_ci_lower": adjusted.ci_lower,
                "adjusted_ci_upper": adjusted.ci_upper,
                "adjusted_covered": adjusted.ci_lower <= π <= adjusted.ci_upper,
                "cr_estimate": cr.estimate,
                "cr_ci_lower": cr.ci_lower,
                "cr_ci_upper": cr.ci_upper,
                "cr_covered": cr.ci_lower <= π <= cr.ci_upper,
            }
        )

    results = pd.DataFrame(records)
    for col in ["direct_covered", "adjusted_covered", "cr_covered"]:
        coverage = results[col].mean()
        logger.info("Estimator %s: empirical coverage = %.3f (nominal %.3f)", col, coverage, confidence_level)

    return results
