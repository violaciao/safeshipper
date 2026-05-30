"""
Actor-level feature engineering for integrity measurement.

Models posting patterns, session dynamics, and cross-account coordination 
as signals for prevalence measurement beyond content classifiers. 

Key insight
-----------
Content signals are lagging indicators. By the time a harmful post is detected and
reviewed, a bad actor may have posted thousands of items. Actor-level risk scores
enable proactive detection and improve sampling efficiency by weighting measurement
samples toward high-risk actors before expensive LLM review.

Architecture
------------
ActorFeatureExtractor  — computes per-actor behavioral risk scores
detect_coordination_networks() — identifies clusters of coordinated accounts
simulate_actor_corpus() — generates synthetic ground-truth data for validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights — can be tuned against a gold-standard labelled actor set
# ---------------------------------------------------------------------------

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "velocity":    0.25,  # posting rate anomaly
    "newness":     0.20,  # account age (new = higher risk)
    "repetition":  0.25,  # content diversity (low = spam/coordination)
    "automation":  0.15,  # API usage fraction (proxy for bots)
    "enforcement": 0.10,  # prior enforcement actions
    "coordination": 0.05, # network-level coordination signal
}

RISK_TIER_THRESHOLDS: dict[str, float] = {
    "high":   0.60,
    "medium": 0.35,
    "low":    0.00,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ActorFeatures:
    """
    Raw behavioral feature vector for a single actor.

    Attributes
    ----------
    actor_id : str
        Unique platform actor identifier.
    account_age_days : int
        Days since account creation.
    post_count_7d : int
        Posts in the last 7 days.
    post_count_30d : int
        Posts in the last 30 days.
    post_count_total : int
        All-time post count.
    unique_content_ratio : float
        Distinct posts / total posts. Low ratio indicates repetitive or
        copied content — a strong spam and coordination signal.
    avg_content_length : float
        Mean post length in characters.
    api_usage_fraction : float
        Fraction of posts submitted via API. High values are a bot signal.
    prior_enforcement_count : int
        Number of previous enforcement actions against this actor.
    is_verified : bool
        Whether the actor holds a verified identity badge.
    """

    actor_id: str
    account_age_days: int
    post_count_7d: int
    post_count_30d: int
    post_count_total: int
    unique_content_ratio: float
    avg_content_length: float
    api_usage_fraction: float
    prior_enforcement_count: int
    is_verified: bool = False


@dataclass
class ActorRiskScore:
    """
    Computed risk score for a single actor.

    Attributes
    ----------
    actor_id : str
    velocity_score : float
        Posting rate anomaly score (0–1). High = suspicious posting speed or spike.
    newness_score : float
        Account age risk (0–1). High = very new account.
    repetition_score : float
        Content diversity risk (0–1). High = repetitive / copied content.
    automation_score : float
        Bot likelihood (0–1). High = predominantly API-driven.
    enforcement_score : float
        Prior enforcement history (0–1). Log-scaled.
    coordination_score : float
        Network coordination signal (0–1). Populated by
        :func:`detect_coordination_networks`.
    composite_score : float
        Weighted combination of all component scores (0–1).
    risk_tier : str
        ``"high"`` | ``"medium"`` | ``"low"`` based on composite score.
    """

    actor_id: str
    velocity_score: float
    newness_score: float
    repetition_score: float
    automation_score: float
    enforcement_score: float
    coordination_score: float
    composite_score: float
    risk_tier: str


@dataclass
class NetworkCluster:
    """
    A detected coordinated inauthentic behavior network.

    Attributes
    ----------
    cluster_id : int
    actor_ids : list[str]
    size : int
        Number of accounts in the cluster.
    avg_behavioral_similarity : float
        Mean pairwise cosine similarity of actor feature vectors.
    dominant_harm_vertical : str | None
        Most common harm vertical among flagged content, if available.
    """

    cluster_id: int
    actor_ids: list[str]
    size: int
    avg_behavioral_similarity: float
    dominant_harm_vertical: str | None = None


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class ActorFeatureExtractor:
    """
    Extracts behavioral risk signals from actor posting history.

    Operates on a pandas DataFrame where each row represents one actor.
    Required columns: all fields from :class:`ActorFeatures`.
    Returns the input DataFrame augmented with score columns.

    Parameters
    ----------
    score_weights : dict[str, float], optional
        Weights for each score component. Must sum to 1.0. Defaults to
        :data:`DEFAULT_SCORE_WEIGHTS`.
    velocity_baseline_posts_per_day : float
        Expected daily posting rate for a typical legitimate account.
        Used to normalise the velocity score.
    new_account_threshold_days : int
        Accounts younger than this (days) are considered high-risk on newness.
    """

    def __init__(
        self,
        score_weights: dict[str, float] | None = None,
        velocity_baseline_posts_per_day: float = 3.0,
        new_account_threshold_days: int = 30,
    ) -> None:
        self.weights = score_weights or DEFAULT_SCORE_WEIGHTS
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Score weights must sum to 1.0")
        self.velocity_baseline = velocity_baseline_posts_per_day
        self.new_account_threshold = new_account_threshold_days

    def compute_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all risk score components and composite score for each actor.

        Parameters
        ----------
        df : pd.DataFrame
            Actor feature DataFrame. Must contain columns matching
            :class:`ActorFeatures` field names.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added columns: ``velocity_score``,
            ``newness_score``, ``repetition_score``, ``automation_score``,
            ``enforcement_score``, ``coordination_score`` (initialised to 0;
            update via :func:`detect_coordination_networks`), ``composite_score``,
            ``risk_tier``.
        """
        out = df.copy()

        out["velocity_score"] = self._velocity_score(out)
        out["newness_score"] = self._newness_score(out)
        out["repetition_score"] = self._repetition_score(out)
        out["automation_score"] = self._automation_score(out)
        out["enforcement_score"] = self._enforcement_score(out)

        if "coordination_score" not in out.columns:
            out["coordination_score"] = 0.0

        w = self.weights
        out["composite_score"] = (
            w["velocity"]    * out["velocity_score"]
            + w["newness"]     * out["newness_score"]
            + w["repetition"]  * out["repetition_score"]
            + w["automation"]  * out["automation_score"]
            + w["enforcement"] * out["enforcement_score"]
            + w["coordination"] * out["coordination_score"]
        ).clip(0, 1)

        out["risk_tier"] = out["composite_score"].apply(self._assign_tier)

        logger.info(
            "Risk scores computed: %d actors | high=%d medium=%d low=%d",
            len(out),
            (out["risk_tier"] == "high").sum(),
            (out["risk_tier"] == "medium").sum(),
            (out["risk_tier"] == "low").sum(),
        )
        return out

    def stratum_sampling_weights(self, scored_df: pd.DataFrame) -> pd.Series:
        """
        Compute per-actor sampling weights for risk-stratified sampling.

        Actors with higher composite risk scores receive proportionally higher
        weights, concentrating expensive LLM review budget on the most likely
        bad actors.

        Parameters
        ----------
        scored_df : pd.DataFrame
            Output of :meth:`compute_risk_scores`.

        Returns
        -------
        pd.Series
            Normalised sampling weights (sum = 1.0) indexed like ``scored_df``.
        """
        raw_weights = scored_df["composite_score"] + 0.05  # floor so no actor is excluded
        return raw_weights / raw_weights.sum()

    # ------------------------------------------------------------------
    # Component score methods
    # ------------------------------------------------------------------

    def _velocity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        High posting rate or sudden velocity spike → elevated risk.

        Combines:
        - Absolute rate: daily_rate_7d vs baseline
        - Spike ratio: 7d rate vs 30d baseline (detects burst campaigns)
        """
        daily_7d = df["post_count_7d"] / 7.0
        daily_30d = (df["post_count_30d"] / 30.0).clip(lower=0.1)
        spike_ratio = daily_7d / daily_30d

        rate_score = (daily_7d / (self.velocity_baseline * 10)).clip(0, 1)
        spike_score = ((spike_ratio - 1.0) / 9.0).clip(0, 1)

        return (0.55 * rate_score + 0.45 * spike_score).clip(0, 1)

    def _newness_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Exponential decay: new accounts are higher risk.
        Score ≈ 1.0 at day 0, ≈ 0.05 at threshold, approaches 0 for old accounts.
        """
        return np.exp(-df["account_age_days"] / self.new_account_threshold).clip(0, 1)

    def _repetition_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Low content diversity signals spam or coordinated amplification.
        unique_content_ratio near 0 → score near 1.
        """
        return (1.0 - df["unique_content_ratio"].clip(0, 1)).clip(0, 1)

    def _automation_score(self, df: pd.DataFrame) -> pd.Series:
        """API usage fraction is a direct bot signal."""
        return df["api_usage_fraction"].clip(0, 1)

    def _enforcement_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Log-scaled prior enforcement count.
        0 actions → 0.0; 10+ actions → approaching 1.0.
        """
        return (np.log1p(df["prior_enforcement_count"]) / np.log1p(10)).clip(0, 1)

    @staticmethod
    def _assign_tier(score: float) -> str:
        if score >= RISK_TIER_THRESHOLDS["high"]:
            return "high"
        if score >= RISK_TIER_THRESHOLDS["medium"]:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Network coordination detection
# ---------------------------------------------------------------------------


def detect_coordination_networks(
    scored_df: pd.DataFrame,
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> tuple[pd.DataFrame, list[NetworkCluster]]:
    """
    Detect coordinated inauthentic behavior networks using behavioral similarity.

    Groups actors into coordination clusters based on the cosine similarity of
    their behavioral feature vectors. In production this would be augmented with
    content fingerprint similarity (e.g., perceptual hashes, n-gram shingles) and
    temporal co-occurrence of posting activity.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of :meth:`ActorFeatureExtractor.compute_risk_scores`.
    similarity_threshold : float
        Minimum cosine similarity for two actors to be considered coordinated.
        Empirically calibrated; lower values increase recall at the cost of FPR.
    min_cluster_size : int
        Minimum cluster size to be reported as a coordination network.

    Returns
    -------
    actor_df : pd.DataFrame
        Input DataFrame with added ``network_cluster_id`` (−1 = unclustered)
        and updated ``coordination_score`` columns.
    clusters : list[NetworkCluster]
        Detected coordination networks, sorted by size descending.

    Notes
    -----
    This implementation uses a greedy threshold-based approach for simplicity.
    Production systems typically use graph community detection (e.g., Louvain)
    on a similarity graph built from content fingerprint overlap + temporal
    proximity, with the behavioral features used as a secondary signal.
    """
    feature_cols = [
        "unique_content_ratio",
        "api_usage_fraction",
        "velocity_score",
        "newness_score",
        "repetition_score",
    ]
    available = [c for c in feature_cols if c in scored_df.columns]
    if not available:
        logger.warning("No feature columns found for network detection. Returning input unchanged.")
        scored_df = scored_df.copy()
        scored_df["network_cluster_id"] = -1
        scored_df["coordination_score"] = 0.0
        return scored_df, []

    X = MinMaxScaler().fit_transform(scored_df[available].fillna(0))
    sim = cosine_similarity(X)

    cluster_ids = np.full(len(scored_df), -1, dtype=int)
    cluster_counter = 0

    for i in range(len(scored_df)):
        if cluster_ids[i] != -1:
            continue
        neighbors = np.where(sim[i] >= similarity_threshold)[0]
        if len(neighbors) >= min_cluster_size:
            # Only assign to new cluster if not already clustered
            unassigned = neighbors[cluster_ids[neighbors] == -1]
            cluster_ids[unassigned] = cluster_counter
            cluster_counter += 1

    out = scored_df.copy()
    out["network_cluster_id"] = cluster_ids

    # Compute coordination score: log-scaled cluster size
    cluster_sizes = pd.Series(cluster_ids).value_counts()
    valid_clusters = cluster_sizes[cluster_sizes.index >= 0]
    max_size = valid_clusters.max() if len(valid_clusters) > 0 else 1

    def _coord_score(cid: int) -> float:
        if cid < 0:
            return 0.0
        size = cluster_sizes.get(cid, 1)
        return float(np.log1p(size) / np.log1p(max_size))

    out["coordination_score"] = out["network_cluster_id"].apply(_coord_score)

    # Recompute composite score with coordination signal
    w = DEFAULT_SCORE_WEIGHTS
    score_cols = ["velocity_score", "newness_score", "repetition_score",
                  "automation_score", "enforcement_score", "coordination_score"]
    if all(c in out.columns for c in score_cols):
        out["composite_score"] = (
            w["velocity"]     * out["velocity_score"]
            + w["newness"]      * out["newness_score"]
            + w["repetition"]   * out["repetition_score"]
            + w["automation"]   * out["automation_score"]
            + w["enforcement"]  * out["enforcement_score"]
            + w["coordination"] * out["coordination_score"]
        ).clip(0, 1)
        out["risk_tier"] = out["composite_score"].apply(ActorFeatureExtractor._assign_tier)

    # Build NetworkCluster objects — only include clusters meeting min_cluster_size
    clusters: list[NetworkCluster] = []
    qualified_clusters = valid_clusters[valid_clusters >= min_cluster_size]
    for cid in sorted(qualified_clusters.index):
        mask = out["network_cluster_id"] == cid
        actor_ids = out.loc[mask, "actor_id"].tolist()
        indices = np.where(cluster_ids == cid)[0]
        if len(indices) >= 2:
            sub_sim = sim[np.ix_(indices, indices)]
            np.fill_diagonal(sub_sim, np.nan)
            avg_sim = float(np.nanmean(sub_sim))
        else:
            avg_sim = 1.0
        clusters.append(
            NetworkCluster(
                cluster_id=int(cid),
                actor_ids=actor_ids,
                size=len(actor_ids),
                avg_behavioral_similarity=avg_sim,
            )
        )

    clusters.sort(key=lambda c: c.size, reverse=True)
    logger.info(
        "Network detection: %d clusters found (min_size=%d), %d actors clustered",
        len(clusters),
        min_cluster_size,
        (cluster_ids >= 0).sum(),
    )
    return out, clusters


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def simulate_actor_corpus(
    n_actors: int = 10_000,
    bad_actor_fraction: float = 0.05,
    network_fraction: float = 0.40,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic actor corpus with known ground-truth labels.

    Creates three actor populations with distinct behavioral signatures:
    - **Good actors:** Organic, diverse content, normal posting rates
    - **Solo bad actors:** High velocity, repetitive content, new accounts
    - **Network bad actors:** Extremely repetitive, API-driven, tightly clustered

    Parameters
    ----------
    n_actors : int
        Total number of actors to generate.
    bad_actor_fraction : float
        True prevalence of bad actors (used as ground truth for evaluation).
    network_fraction : float
        Fraction of bad actors operating as part of a coordinated network.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Actor corpus with all feature columns plus ground-truth labels
        ``is_bad_actor`` and ``is_network_actor``.
    """
    rng = np.random.default_rng(random_seed)

    n_bad = int(n_actors * bad_actor_fraction)
    n_good = n_actors - n_bad
    n_network_bad = int(n_bad * network_fraction)
    n_solo_bad = n_bad - n_network_bad

    def _make_actors(n: int, offset: int, params: dict) -> pd.DataFrame:
        return pd.DataFrame({
            "actor_id": [f"actor_{offset + i:06d}" for i in range(n)],
            **{k: v(n) for k, v in params.items()},
        })

    good_params = {
        "is_bad_actor":            lambda n: np.zeros(n, dtype=bool),
        "is_network_actor":        lambda n: np.zeros(n, dtype=bool),
        "account_age_days":        lambda n: rng.integers(60, 2000, n),
        "post_count_7d":           lambda n: rng.integers(0, 15, n),
        "post_count_30d":          lambda n: rng.integers(5, 50, n),
        "post_count_total":        lambda n: rng.integers(20, 1500, n),
        "unique_content_ratio":    lambda n: rng.beta(8, 2, n).clip(0.2, 1.0),
        "avg_content_length":      lambda n: rng.normal(160, 60, n).clip(30, 600),
        "api_usage_fraction":      lambda n: rng.beta(1, 9, n),
        "prior_enforcement_count": lambda n: rng.poisson(0.04, n),
        "is_verified":             lambda n: rng.random(n) < 0.05,
    }

    solo_bad_params = {
        "is_bad_actor":            lambda n: np.ones(n, dtype=bool),
        "is_network_actor":        lambda n: np.zeros(n, dtype=bool),
        "account_age_days":        lambda n: rng.integers(1, 45, n),
        "post_count_7d":           lambda n: rng.integers(80, 600, n),
        "post_count_30d":          lambda n: rng.integers(150, 1500, n),
        "post_count_total":        lambda n: rng.integers(200, 3000, n),
        "unique_content_ratio":    lambda n: rng.beta(2, 8, n).clip(0.01, 0.5),
        "avg_content_length":      lambda n: rng.normal(70, 25, n).clip(10, 250),
        "api_usage_fraction":      lambda n: rng.beta(6, 2, n).clip(0.3, 1.0),
        "prior_enforcement_count": lambda n: rng.poisson(2.0, n),
        "is_verified":             lambda n: np.zeros(n, dtype=bool),
    }

    network_bad_params = {
        "is_bad_actor":            lambda n: np.ones(n, dtype=bool),
        "is_network_actor":        lambda n: np.ones(n, dtype=bool),
        "account_age_days":        lambda n: rng.integers(0, 20, n),
        "post_count_7d":           lambda n: rng.integers(30, 250, n),
        "post_count_30d":          lambda n: rng.integers(35, 300, n),
        "post_count_total":        lambda n: rng.integers(35, 600, n),
        "unique_content_ratio":    lambda n: rng.beta(1, 12, n).clip(0.01, 0.2),
        "avg_content_length":      lambda n: rng.normal(55, 15, n).clip(10, 150),
        "api_usage_fraction":      lambda n: rng.beta(8, 1.5, n).clip(0.6, 1.0),
        "prior_enforcement_count": lambda n: rng.poisson(0.3, n),
        "is_verified":             lambda n: np.zeros(n, dtype=bool),
    }

    good_df = _make_actors(n_good, 0, good_params)
    solo_df = _make_actors(n_solo_bad, n_good, solo_bad_params)
    network_df = _make_actors(n_network_bad, n_good + n_solo_bad, network_bad_params)

    corpus = (
        pd.concat([good_df, solo_df, network_df], ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    # Convenience alias used by notebooks and tests
    corpus["true_label"] = corpus["is_bad_actor"].astype(int)

    logger.info(
        "Simulated actor corpus: n=%d | bad_actors=%d (%.2f%%) | network_actors=%d",
        n_actors, n_bad, 100 * bad_actor_fraction, n_network_bad,
    )
    return corpus
