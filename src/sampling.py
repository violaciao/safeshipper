"""
Stratified sampling design for rare event estimation.

Implements proportional, Neyman optimal, and risk-score-stratified allocation
strategies. Demonstrates why proportional sampling fails at low prevalence and
how oversampling high-risk strata improves estimation efficiency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Stratum:
    """
    Definition of a single stratum in a sampling design.

    Parameters
    ----------
    name : str
        Stratum label (e.g., ``"high_risk"``).
    N : int
        Stratum population size.
    estimated_prevalence : float
        Prior estimate of harm prevalence within this stratum.
    labeling_cost_per_item : float
        Cost in USD per item to obtain a label (e.g., LLM API cost).
    """

    name: str
    N: int
    estimated_prevalence: float
    labeling_cost_per_item: float = 0.001


@dataclass
class SampleDesign:
    """
    Output of a sampling design calculation.

    Attributes
    ----------
    strata : list[Stratum]
        The strata used in the design.
    allocation : dict[str, int]
        Mapping from stratum name to allocated sample size.
    n_total : int
        Total sample size across all strata.
    expected_ci_half_width : float
        Expected half-width of the 95% CI on the overall prevalence estimate.
    design_effect : float
        DEFF relative to simple random sampling. Values >1 indicate efficiency loss.
    expected_cost_usd : float
        Estimated total labeling cost in USD.
    strategy : str
        Allocation strategy used (``"proportional"``, ``"neyman"``, ``"risk_stratified"``).
    notes : list[str]
        Human-readable notes on the design.
    """

    strata: list[Stratum]
    allocation: dict[str, int]
    n_total: int
    expected_ci_half_width: float
    design_effect: float
    expected_cost_usd: float
    strategy: str
    notes: list[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """Return a formatted summary DataFrame of the sampling design."""
        rows = []
        for s in self.strata:
            n_h = self.allocation[s.name]
            rows.append(
                {
                    "stratum": s.name,
                    "N_h": s.N,
                    "est_prevalence": s.estimated_prevalence,
                    "n_h_allocated": n_h,
                    "sampling_fraction": n_h / s.N,
                    "cost_h": n_h * s.labeling_cost_per_item,
                }
            )
        return pd.DataFrame(rows)


class StratifiedHarmSampler:
    """
    Designs sampling plans for rare event prevalence estimation.

    Supports proportional, Neyman optimal, and risk-score-stratified allocation.
    Calculates confidence interval expectations and design effects relative to
    simple random sampling.

    Parameters
    ----------
    confidence_level : float
        Desired coverage probability for CI calculations (default 0.95).
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level
        self._z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    def design_sample(
        self,
        corpus_size: int,
        estimated_prevalence: float,
        target_ci_width: float,
        strata_weights: dict[str, float] | None = None,
        labeling_cost_per_item: float = 0.001,
        strategy: str = "neyman",
    ) -> SampleDesign:
        """
        Design a stratified sample for a platform corpus.

        Parameters
        ----------
        corpus_size : int
            Total number of items in the platform corpus.
        estimated_prevalence : float
            Prior estimate of overall harm prevalence.
        target_ci_width : float
            Target half-width of the 95% CI on prevalence (e.g., 0.002 for ±0.2pp).
        strata_weights : dict[str, float], optional
            Mapping of stratum name to proportion of corpus. If None, uses a
            default two-stratum design: ``{"high_risk": 0.1, "low_risk": 0.9}``.
            Strata are assumed to have prevalence rates of 10× and 1× the
            overall estimate respectively.
        labeling_cost_per_item : float
            Labeling cost per item in USD (applies to all strata equally).
        strategy : str
            One of ``"proportional"``, ``"neyman"``, or ``"risk_stratified"``.

        Returns
        -------
        SampleDesign
            Full sampling design specification.
        """
        if strata_weights is None:
            strata_weights = {"high_risk": 0.1, "low_risk": 0.9}

        # Assign per-stratum prevalence estimates based on risk tier.
        # High-risk stratum is assumed to concentrate ~80% of all harmful items.
        total_harm_count = corpus_size * estimated_prevalence
        stratum_names = list(strata_weights.keys())
        strata_N = {name: int(corpus_size * w) for name, w in strata_weights.items()}

        # Compute per-stratum prevalence estimates.
        stratum_prevalences = self._estimate_stratum_prevalences(
            strata_N, estimated_prevalence, total_harm_count
        )

        strata = [
            Stratum(
                name=name,
                N=strata_N[name],
                estimated_prevalence=stratum_prevalences[name],
                labeling_cost_per_item=labeling_cost_per_item,
            )
            for name in stratum_names
        ]

        if strategy == "proportional":
            allocation = self._proportional_allocation(strata, target_ci_width)
        elif strategy == "neyman":
            allocation = self._neyman_allocation(strata, target_ci_width)
        elif strategy == "risk_stratified":
            allocation = self._risk_stratified_allocation(strata, target_ci_width)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Choose from 'proportional', 'neyman', 'risk_stratified'.")

        n_total = sum(allocation.values())
        ci_half_width = self._expected_ci_half_width(strata, allocation, corpus_size)
        deff = self._design_effect(strata, allocation, corpus_size, estimated_prevalence)
        cost = sum(allocation[s.name] * s.labeling_cost_per_item for s in strata)

        notes = self._generate_design_notes(
            corpus_size, estimated_prevalence, n_total, ci_half_width, strategy
        )

        logger.info(
            "Sample design (%s): n=%d, CI half-width=%.4f, DEFF=%.2f, cost=$%.2f",
            strategy,
            n_total,
            ci_half_width,
            deff,
            cost,
        )

        return SampleDesign(
            strata=strata,
            allocation=allocation,
            n_total=n_total,
            expected_ci_half_width=ci_half_width,
            design_effect=deff,
            expected_cost_usd=cost,
            strategy=strategy,
            notes=notes,
        )

    def minimum_sample_size_srs(
        self,
        estimated_prevalence: float,
        target_ci_width: float,
    ) -> int:
        """
        Compute the minimum sample size for a simple random sample.

        Uses the standard formula: n = z² * p * (1-p) / e²

        Parameters
        ----------
        estimated_prevalence : float
            Prior estimate of harm prevalence.
        target_ci_width : float
            Target half-width of CI (e.g., 0.005 for ±0.5pp).

        Returns
        -------
        int
            Minimum sample size.
        """
        p = estimated_prevalence
        e = target_ci_width
        n = (self._z**2 * p * (1 - p)) / (e**2)
        return math.ceil(n)

    def expected_positives(self, n_sample: int, prevalence: float) -> float:
        """Expected number of harmful items in a sample."""
        return n_sample * prevalence

    # ------------------------------------------------------------------
    # Internal allocation methods
    # ------------------------------------------------------------------

    def _estimate_stratum_prevalences(
        self,
        strata_N: dict[str, int],
        overall_prevalence: float,
        total_harm_count: float,
    ) -> dict[str, float]:
        """
        Assign per-stratum prevalences consistent with the overall prevalence.

        The first stratum (highest risk) is assumed to contain 80% of all harmful
        items; the remainder is distributed proportionally among lower-risk strata.
        """
        names = list(strata_N.keys())
        if len(names) == 1:
            return {names[0]: overall_prevalence}

        prevalences: dict[str, float] = {}
        harm_in_high = 0.80 * total_harm_count
        prevalences[names[0]] = min(harm_in_high / strata_N[names[0]], 0.999)

        remaining_harm = total_harm_count - harm_in_high
        remaining_n = sum(strata_N[n] for n in names[1:])
        base_prev = remaining_harm / remaining_n if remaining_n > 0 else 0

        for name in names[1:]:
            prevalences[name] = max(min(base_prev, 0.999), 1e-6)

        return prevalences

    def _proportional_allocation(
        self, strata: list[Stratum], target_ci_width: float
    ) -> dict[str, int]:
        """Allocate samples proportional to stratum size."""
        # First compute total n for SRS at the pooled level
        p_overall = sum(s.N * s.estimated_prevalence for s in strata) / sum(s.N for s in strata)
        n_srs = self.minimum_sample_size_srs(p_overall, target_ci_width)
        n_total = max(n_srs, 100)

        N_total = sum(s.N for s in strata)
        allocation = {}
        for s in strata:
            n_h = max(1, round(n_total * s.N / N_total))
            allocation[s.name] = n_h
        return allocation

    def _neyman_allocation(
        self, strata: list[Stratum], target_ci_width: float
    ) -> dict[str, int]:
        """
        Neyman optimal allocation — allocate proportional to N_h * σ_h.

        Minimizes variance for a fixed total sample size. Particularly effective
        when stratum variances differ substantially (as they do in rare event settings).
        """
        # Compute total n using overall variance formula
        N_total = sum(s.N for s in strata)
        p_overall = sum(s.N * s.estimated_prevalence for s in strata) / N_total

        # Use Neyman formula: n = (Σ N_h σ_h)² / (N²e²/z² + Σ N_h σ_h²)
        sigma_h = {s.name: math.sqrt(s.estimated_prevalence * (1 - s.estimated_prevalence)) for s in strata}
        sum_N_sigma = sum(s.N * sigma_h[s.name] for s in strata)
        sum_N_sigma2 = sum(s.N * sigma_h[s.name] ** 2 for s in strata)
        e = target_ci_width
        denom = (N_total * e / self._z) ** 2 + sum_N_sigma2
        n_total = max(math.ceil(sum_N_sigma**2 / denom), 100)

        # Allocate proportional to N_h * σ_h
        weights = {s.name: s.N * sigma_h[s.name] for s in strata}
        weight_sum = sum(weights.values())
        allocation = {s.name: max(1, round(n_total * weights[s.name] / weight_sum)) for s in strata}
        return allocation

    def _risk_stratified_allocation(
        self, strata: list[Stratum], target_ci_width: float
    ) -> dict[str, int]:
        """
        Risk-stratified allocation with aggressive oversampling of the high-risk stratum.

        Designed for the operational setting where:
        1. A cheap signal (keyword match or prior classifier score) has already
           partitioned the corpus into risk tiers.
        2. We want to concentrate expensive LLM labeling budget on high-risk items.

        Allocates 60% of budget to the first (highest-risk) stratum regardless of size.
        """
        n_neyman = self._neyman_allocation(strata, target_ci_width)
        n_total = sum(n_neyman.values())

        allocation = {}
        high_risk_n = max(1, round(0.60 * n_total))
        allocation[strata[0].name] = high_risk_n

        remaining = n_total - high_risk_n
        remaining_weight = sum(s.N for s in strata[1:])
        for s in strata[1:]:
            w = s.N / remaining_weight if remaining_weight > 0 else 1 / (len(strata) - 1)
            allocation[s.name] = max(1, round(remaining * w))

        return allocation

    def _expected_ci_half_width(
        self,
        strata: list[Stratum],
        allocation: dict[str, int],
        N_total: int,
    ) -> float:
        """Compute expected CI half-width for the stratified estimator."""
        variance = 0.0
        for s in strata:
            n_h = allocation[s.name]
            p_h = s.estimated_prevalence
            W_h = s.N / N_total
            var_h = p_h * (1 - p_h) / n_h * (1 - n_h / s.N)
            variance += W_h**2 * var_h
        return self._z * math.sqrt(variance)

    def _design_effect(
        self,
        strata: list[Stratum],
        allocation: dict[str, int],
        N_total: int,
        overall_prevalence: float,
    ) -> float:
        """
        Compute the design effect (DEFF) relative to SRS.

        DEFF = Var(stratified) / Var(SRS)

        Values < 1 indicate efficiency gains from stratification; > 1 indicate loss.
        """
        n_total = sum(allocation.values())
        var_srs = overall_prevalence * (1 - overall_prevalence) / n_total
        var_strat = self._expected_ci_half_width(strata, allocation, N_total) ** 2 / self._z**2

        if var_srs == 0:
            return 1.0
        return var_strat / var_srs

    def _generate_design_notes(
        self,
        corpus_size: int,
        prevalence: float,
        n_total: int,
        ci_half_width: float,
        strategy: str,
    ) -> list[str]:
        """Generate human-readable design rationale notes."""
        n_srs = self.minimum_sample_size_srs(prevalence, ci_half_width)
        expected_positives = n_total * prevalence

        notes = [
            f"Corpus size: {corpus_size:,}; true prevalence estimate: {prevalence:.4%}",
            f"Strategy: {strategy} — total sample size: {n_total:,}",
            f"Equivalent SRS size for same CI width: {n_srs:,}",
            f"Expected harmful items in sample: {expected_positives:.1f}",
        ]

        if expected_positives < 5:
            notes.append(
                "WARNING: Fewer than 5 expected positives in sample — CI will be unreliable. "
                "Consider increasing sample size or using risk-stratified design."
            )

        if prevalence < 0.001:
            notes.append(
                "Prevalence < 0.1% — proportional sampling is highly inefficient. "
                "Risk-stratified or Neyman design is strongly recommended."
            )

        return notes
