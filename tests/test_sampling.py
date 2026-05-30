"""
Tests for src/sampling.py — stratified sampling design.

Covers: Neyman allocation, proportional allocation, risk-stratified allocation,
minimum sample size calculations, and design effect computation.
"""

import math

import pytest

from src.sampling import SampleDesign, Stratum, StratifiedHarmSampler


@pytest.fixture
def sampler() -> StratifiedHarmSampler:
    return StratifiedHarmSampler(confidence_level=0.95)


@pytest.fixture
def two_strata() -> dict[str, float]:
    return {"high_risk": 0.10, "low_risk": 0.90}


# ---------------------------------------------------------------------------
# Stratum dataclass
# ---------------------------------------------------------------------------


class TestStratum:
    def test_basic_creation(self):
        s = Stratum(name="high_risk", N=10_000, estimated_prevalence=0.10)
        assert s.name == "high_risk"
        assert s.N == 10_000
        assert s.estimated_prevalence == 0.10

    def test_default_cost(self):
        s = Stratum(name="low_risk", N=90_000, estimated_prevalence=0.01)
        assert s.labeling_cost_per_item == 0.001


# ---------------------------------------------------------------------------
# Minimum sample size (SRS)
# ---------------------------------------------------------------------------


class TestMinimumSampleSize:
    def test_common_scenario(self, sampler):
        # p=0.05, e=0.005 → n = 1.96² * 0.05 * 0.95 / 0.005² ≈ 7299
        n = sampler.minimum_sample_size_srs(
            estimated_prevalence=0.05,
            target_ci_width=0.005,
        )
        assert 7000 <= n <= 8000

    def test_rare_event_large_n(self, sampler):
        # At 0.1% prevalence, need very large sample for tight CI
        n = sampler.minimum_sample_size_srs(
            estimated_prevalence=0.001,
            target_ci_width=0.001,
        )
        assert n > 5000

    def test_common_event_smaller_n(self, sampler):
        # At 50% prevalence, CI converges faster
        n_rare = sampler.minimum_sample_size_srs(0.001, 0.005)
        n_common = sampler.minimum_sample_size_srs(0.50, 0.005)
        assert n_common < n_rare

    def test_tighter_ci_requires_more_samples(self, sampler):
        n_wide = sampler.minimum_sample_size_srs(0.05, 0.01)
        n_tight = sampler.minimum_sample_size_srs(0.05, 0.005)
        assert n_tight > n_wide


# ---------------------------------------------------------------------------
# Proportional allocation
# ---------------------------------------------------------------------------


class TestProportionalAllocation:
    def test_basic_design(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.01,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="proportional",
        )
        assert design.strategy == "proportional"
        assert design.n_total > 0
        assert set(design.allocation.keys()) == {"high_risk", "low_risk"}

    def test_allocation_roughly_proportional_to_strata(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.05,
            target_ci_width=0.01,
            strata_weights=two_strata,
            strategy="proportional",
        )
        n_high = design.allocation["high_risk"]
        n_low = design.allocation["low_risk"]
        # high_risk is 10% of corpus → should get ~10% of samples
        ratio = n_high / (n_high + n_low)
        assert 0.05 <= ratio <= 0.20  # allow some rounding slack

    def test_all_strata_have_positive_allocation(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.02,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="proportional",
        )
        for stratum, n in design.allocation.items():
            assert n >= 1, f"Stratum {stratum!r} has zero allocation"


# ---------------------------------------------------------------------------
# Neyman optimal allocation
# ---------------------------------------------------------------------------


class TestNeymanAllocation:
    def test_basic_design(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.01,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="neyman",
        )
        assert design.strategy == "neyman"
        assert design.n_total > 0

    def test_neyman_oversamples_high_variance_stratum(self, sampler):
        # High-risk stratum has much higher prevalence → more oversampling with Neyman
        design_neyman = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.01,
            target_ci_width=0.003,
            strata_weights={"high_risk": 0.10, "low_risk": 0.90},
            strategy="neyman",
        )
        design_prop = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.01,
            target_ci_width=0.003,
            strata_weights={"high_risk": 0.10, "low_risk": 0.90},
            strategy="proportional",
        )
        n_high_neyman = design_neyman.allocation["high_risk"]
        n_high_prop = design_prop.allocation["high_risk"]
        # Neyman should put more weight on high-risk (higher σ)
        ratio_neyman = n_high_neyman / design_neyman.n_total
        ratio_prop = n_high_prop / design_prop.n_total
        assert ratio_neyman > ratio_prop

    def test_cost_is_nonnegative(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.02,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="neyman",
        )
        assert design.expected_cost_usd >= 0.0

    def test_design_effect_positive(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.02,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="neyman",
        )
        assert design.design_effect > 0.0


# ---------------------------------------------------------------------------
# Risk-stratified allocation
# ---------------------------------------------------------------------------


class TestRiskStratifiedAllocation:
    def test_high_risk_gets_majority(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=1_000_000,
            estimated_prevalence=0.005,
            target_ci_width=0.002,
            strata_weights=two_strata,
            strategy="risk_stratified",
        )
        n_high = design.allocation["high_risk"]
        assert n_high / design.n_total >= 0.55  # at least 55% to high-risk

    def test_strategy_label(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.01,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="risk_stratified",
        )
        assert design.strategy == "risk_stratified"


# ---------------------------------------------------------------------------
# Unknown strategy
# ---------------------------------------------------------------------------


class TestInvalidStrategy:
    def test_raises_on_unknown_strategy(self, sampler, two_strata):
        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.design_sample(
                corpus_size=100_000,
                estimated_prevalence=0.01,
                target_ci_width=0.005,
                strata_weights=two_strata,
                strategy="banana",
            )


# ---------------------------------------------------------------------------
# SampleDesign summary table
# ---------------------------------------------------------------------------


class TestSampleDesignSummaryTable:
    def test_summary_table_shape(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.02,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="neyman",
        )
        df = design.summary_table()
        assert len(df) == 2
        assert "stratum" in df.columns
        assert "n_h_allocated" in df.columns
        assert "sampling_fraction" in df.columns

    def test_sampling_fractions_in_range(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=500_000,
            estimated_prevalence=0.02,
            target_ci_width=0.005,
            strata_weights=two_strata,
            strategy="neyman",
        )
        df = design.summary_table()
        assert (df["sampling_fraction"] >= 0).all()
        assert (df["sampling_fraction"] <= 1).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_stratum(self, sampler):
        design = sampler.design_sample(
            corpus_size=100_000,
            estimated_prevalence=0.05,
            target_ci_width=0.01,
            strata_weights={"all": 1.0},
            strategy="neyman",
        )
        assert design.n_total > 0
        assert "all" in design.allocation

    def test_very_low_prevalence_generates_warning_note(self, sampler, two_strata):
        design = sampler.design_sample(
            corpus_size=10_000_000,
            estimated_prevalence=0.0001,  # 0.01%
            target_ci_width=0.0001,
            strata_weights=two_strata,
            strategy="neyman",
        )
        notes_text = " ".join(design.notes)
        assert "prevalence" in notes_text.lower() or len(design.notes) > 0

    def test_expected_positives_helper(self, sampler):
        expected = sampler.expected_positives(n_sample=1000, prevalence=0.05)
        assert math.isclose(expected, 50.0)
