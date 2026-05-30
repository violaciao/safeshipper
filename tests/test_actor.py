"""
Tests for src/actor.py — actor-level feature engineering and network detection.

Covers: ActorFeatureExtractor score computation, risk tier assignment,
coordination network detection, and simulation utility.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.actor import (
    ActorFeatureExtractor,
    ActorFeatures,
    ActorRiskScore,
    NetworkCluster,
    detect_coordination_networks,
    simulate_actor_corpus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> ActorFeatureExtractor:
    return ActorFeatureExtractor()


@pytest.fixture
def small_corpus() -> pd.DataFrame:
    """Minimal 5-actor DataFrame for deterministic unit tests."""
    return pd.DataFrame(
        {
            "actor_id": [f"a{i}" for i in range(5)],
            "account_age_days": [365, 730, 5, 1, 1000],
            "post_count_7d": [1, 2, 200, 500, 3],
            "post_count_30d": [4, 8, 800, 2000, 12],
            "post_count_total": [100, 500, 900, 2100, 3000],
            "unique_content_ratio": [0.95, 0.90, 0.10, 0.05, 0.80],
            "avg_content_length": [120, 200, 30, 25, 150],
            "api_usage_fraction": [0.02, 0.05, 0.80, 0.95, 0.10],
            "prior_enforcement_count": [0, 0, 2, 5, 0],
            "is_verified": [False, True, False, False, True],
        }
    )


# ---------------------------------------------------------------------------
# ActorFeatures dataclass
# ---------------------------------------------------------------------------


class TestActorFeaturesDataclass:
    def test_creation(self):
        f = ActorFeatures(
            actor_id="u1",
            account_age_days=30,
            post_count_7d=100,
            post_count_30d=400,
            post_count_total=1000,
            unique_content_ratio=0.05,
            avg_content_length=20,
            api_usage_fraction=0.9,
            prior_enforcement_count=3,
            is_verified=False,
        )
        assert f.actor_id == "u1"
        assert f.account_age_days == 30
        assert f.api_usage_fraction == 0.9


# ---------------------------------------------------------------------------
# ActorRiskScore dataclass
# ---------------------------------------------------------------------------


class TestActorRiskScore:
    def test_composite_in_range(self):
        score = ActorRiskScore(
            actor_id="u1",
            velocity_score=0.8,
            newness_score=0.9,
            repetition_score=0.7,
            automation_score=0.85,
            enforcement_score=0.5,
            coordination_score=0.0,
            composite_score=0.77,
            risk_tier="high",
        )
        assert 0.0 <= score.composite_score <= 1.0
        assert score.risk_tier in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# ActorFeatureExtractor.compute_risk_scores
# ---------------------------------------------------------------------------


class TestComputeRiskScores:
    def test_output_shape(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        assert len(scored) == len(small_corpus)

    def test_required_columns_present(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        for col in [
            "velocity_score",
            "newness_score",
            "repetition_score",
            "automation_score",
            "enforcement_score",
            "composite_score",
            "risk_tier",
        ]:
            assert col in scored.columns, f"Missing column: {col}"

    def test_composite_in_unit_interval(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        assert (scored["composite_score"] >= 0.0).all()
        assert (scored["composite_score"] <= 1.0).all()

    def test_risk_tiers_are_valid(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        assert set(scored["risk_tier"]).issubset({"high", "medium", "low"})

    def test_bot_like_actor_scores_higher_than_normal(self, extractor, small_corpus):
        """Bot-like actor (high velocity, new, repetitive, API-driven) should
        have higher composite score than a normal, established actor."""
        scored = extractor.compute_risk_scores(small_corpus)
        # actor a2 (index 2): 200 posts/7d, 5 days old, 10% unique, 80% API
        # actor a1 (index 1): 2 posts/7d, 730 days old, 90% unique, 5% API
        bot_score = scored.loc[scored["actor_id"] == "a2", "composite_score"].iloc[0]
        normal_score = scored.loc[scored["actor_id"] == "a1", "composite_score"].iloc[0]
        assert bot_score > normal_score

    def test_high_enforcement_history_raises_score(self, extractor):
        """An actor with prior enforcement should have higher enforcement_score."""
        df = pd.DataFrame(
            {
                "actor_id": ["clean", "prior_action"],
                "account_age_days": [365, 365],
                "post_count_7d": [5, 5],
                "post_count_30d": [20, 20],
                "post_count_total": [500, 500],
                "unique_content_ratio": [0.80, 0.80],
                "avg_content_length": [100, 100],
                "api_usage_fraction": [0.05, 0.05],
                "prior_enforcement_count": [0, 10],
                "is_verified": [False, False],
            }
        )
        scored = extractor.compute_risk_scores(df)
        clean = scored.loc[scored["actor_id"] == "clean", "enforcement_score"].iloc[0]
        prior = scored.loc[
            scored["actor_id"] == "prior_action", "enforcement_score"
        ].iloc[0]
        assert prior > clean

    def test_very_new_account_has_high_newness_score(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        # a3: account_age_days=1
        new_score = scored.loc[scored["actor_id"] == "a3", "newness_score"].iloc[0]
        assert new_score > 0.8

    def test_old_account_has_low_newness_score(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        # a4: account_age_days=1000
        old_score = scored.loc[scored["actor_id"] == "a4", "newness_score"].iloc[0]
        assert old_score < 0.2


# ---------------------------------------------------------------------------
# ActorFeatureExtractor.stratum_sampling_weights
# ---------------------------------------------------------------------------


class TestStratumSamplingWeights:
    def test_weights_sum_to_one(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        weights = extractor.stratum_sampling_weights(scored)
        assert math.isclose(weights.sum(), 1.0, rel_tol=1e-6)

    def test_weights_nonnegative(self, extractor, small_corpus):
        scored = extractor.compute_risk_scores(small_corpus)
        weights = extractor.stratum_sampling_weights(scored)
        assert (weights >= 0).all()

    def test_high_risk_actor_gets_higher_weight(self, extractor, small_corpus):
        """Higher-risk actors should receive proportionally more weight in
        risk-stratified sampling."""
        scored = extractor.compute_risk_scores(small_corpus)
        weights = extractor.stratum_sampling_weights(scored)
        # a2 or a3 are the high-velocity/new accounts → should dominate
        high_risk_mask = scored["risk_tier"] == "high"
        if high_risk_mask.any():
            high_risk_ids = scored.loc[high_risk_mask, "actor_id"]
            low_risk_ids = scored.loc[scored["risk_tier"] == "low", "actor_id"]
            if len(low_risk_ids) > 0:
                avg_high = weights[high_risk_ids.index].mean()
                avg_low = weights[low_risk_ids.index].mean()
                assert avg_high >= avg_low


# ---------------------------------------------------------------------------
# detect_coordination_networks
# ---------------------------------------------------------------------------


class TestDetectCoordinationNetworks:
    def test_returns_correct_types(self, extractor):
        corpus = simulate_actor_corpus(n_actors=200, random_seed=0)
        scored = extractor.compute_risk_scores(corpus)
        actor_df, clusters = detect_coordination_networks(scored)
        assert isinstance(actor_df, pd.DataFrame)
        assert isinstance(clusters, list)
        if clusters:
            assert isinstance(clusters[0], NetworkCluster)

    def test_cluster_column_added(self, extractor):
        corpus = simulate_actor_corpus(n_actors=200, random_seed=0)
        scored = extractor.compute_risk_scores(corpus)
        actor_df, _ = detect_coordination_networks(scored)
        assert "network_cluster_id" in actor_df.columns

    def test_network_actors_cluster_together(self, extractor):
        """Network bad actors (highly similar behavioral vectors) should
        form at least one cluster."""
        corpus = simulate_actor_corpus(
            n_actors=500,
            bad_actor_fraction=0.10,
            network_fraction=0.60,
            random_seed=42,
        )
        scored = extractor.compute_risk_scores(corpus)
        _, clusters = detect_coordination_networks(
            scored, similarity_threshold=0.75, min_cluster_size=3
        )
        assert len(clusters) >= 1, "Expected at least one coordination cluster"

    def test_cluster_sizes_match_actor_ids(self, extractor):
        corpus = simulate_actor_corpus(n_actors=300, random_seed=7)
        scored = extractor.compute_risk_scores(corpus)
        actor_df, clusters = detect_coordination_networks(scored)
        for cluster in clusters:
            assert cluster.size == len(cluster.actor_ids)
            assert cluster.size >= 3  # min_cluster_size default

    def test_cluster_similarity_in_range(self, extractor):
        corpus = simulate_actor_corpus(n_actors=300, random_seed=7)
        scored = extractor.compute_risk_scores(corpus)
        _, clusters = detect_coordination_networks(scored)
        for cluster in clusters:
            assert 0.0 <= cluster.avg_behavioral_similarity <= 1.0

    def test_threshold_zero_no_clusters(self, extractor, small_corpus):
        """With similarity_threshold=1.0, only identical vectors cluster."""
        scored = extractor.compute_risk_scores(small_corpus)
        _, clusters = detect_coordination_networks(
            scored, similarity_threshold=1.0, min_cluster_size=2
        )
        # Five diverse actors; none should be identical
        assert len(clusters) == 0

    def test_min_cluster_size_respected(self, extractor):
        corpus = simulate_actor_corpus(n_actors=400, random_seed=1)
        scored = extractor.compute_risk_scores(corpus)
        _, clusters = detect_coordination_networks(
            scored, similarity_threshold=0.70, min_cluster_size=5
        )
        for cluster in clusters:
            assert cluster.size >= 5


# ---------------------------------------------------------------------------
# simulate_actor_corpus
# ---------------------------------------------------------------------------


class TestSimulateActorCorpus:
    def test_output_shape(self):
        df = simulate_actor_corpus(n_actors=1000, random_seed=0)
        assert len(df) == 1000

    def test_required_columns(self):
        df = simulate_actor_corpus(n_actors=100, random_seed=0)
        for col in [
            "actor_id",
            "account_age_days",
            "post_count_7d",
            "post_count_30d",
            "post_count_total",
            "unique_content_ratio",
            "avg_content_length",
            "api_usage_fraction",
            "prior_enforcement_count",
            "is_verified",
            "true_label",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_bad_actor_fraction(self):
        df = simulate_actor_corpus(
            n_actors=1000, bad_actor_fraction=0.10, random_seed=0
        )
        bad_fraction = df["true_label"].mean()
        assert 0.05 <= bad_fraction <= 0.15

    def test_reproducibility(self):
        df1 = simulate_actor_corpus(n_actors=500, random_seed=42)
        df2 = simulate_actor_corpus(n_actors=500, random_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_actor_corpus(n_actors=500, random_seed=1)
        df2 = simulate_actor_corpus(n_actors=500, random_seed=2)
        assert not df1["actor_id"].equals(df2["actor_id"])

    def test_actor_ids_unique(self):
        df = simulate_actor_corpus(n_actors=500, random_seed=0)
        assert df["actor_id"].nunique() == 500

    def test_numeric_columns_nonnegative(self):
        df = simulate_actor_corpus(n_actors=200, random_seed=0)
        for col in ["account_age_days", "post_count_7d", "post_count_30d"]:
            assert (df[col] >= 0).all(), f"Negative values in {col}"

    def test_ratios_in_unit_interval(self):
        df = simulate_actor_corpus(n_actors=200, random_seed=0)
        for col in ["unique_content_ratio", "api_usage_fraction"]:
            assert (df[col] >= 0).all() and (df[col] <= 1).all(), \
                f"Out-of-range values in {col}"

    def test_network_fraction_produces_clusters(self):
        """network_fraction > 0 should produce groups with similar features."""
        df = simulate_actor_corpus(
            n_actors=500,
            bad_actor_fraction=0.10,
            network_fraction=0.50,
            random_seed=0,
        )
        bad = df[df["true_label"] == 1]
        # Network bad actors should have high API usage + low unique content
        network_bad = bad[bad["api_usage_fraction"] > 0.7]
        assert len(network_bad) > 0
