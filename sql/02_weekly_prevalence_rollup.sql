/*
  02_weekly_prevalence_rollup.sql
  ──────────────────────────────────────────────────────────────────────────────
  Compute weekly harm prevalence estimates using three methods:

    1. Horvitz–Thompson (HT) direct proportion from gold-review sample
       (design-unbiased under stratified sampling)

    2. Classifier-adjusted estimate correcting for TPR/FPR
       π̂ = (q̂ - FPR) / (TPR - FPR)

    3. Simple observed detection rate (naive, for comparison only)

  Rolling 4-week window included for smoothing and trend detection.

  All estimates are in items-per-item (i.e., fractions, not counts).
*/

-- ── Step 1: Compute TPR / FPR from the gold-review sample ─────────────────
WITH classifier_performance AS (
  SELECT
    week,
    harm_vertical,
    -- True positive rate: P(predicted=1 | true=1)
    COUNTIF(predicted_harmful AND true_harmful)
      / NULLIF(COUNTIF(true_harmful), 0)                    AS tpr,
    -- False positive rate: P(predicted=1 | true=0)
    COUNTIF(predicted_harmful AND NOT true_harmful)
      / NULLIF(COUNTIF(NOT true_harmful), 0)                AS fpr,
    -- Sample sizes
    COUNTIF(true_harmful)                                   AS n_positive_gold,
    COUNT(*)                                                AS n_reviewed
  FROM gold_review_samples
  WHERE reviewed_at IS NOT NULL
  GROUP BY week, harm_vertical
),

-- ── Step 2: Horvitz–Thompson prevalence from gold-review sample ────────────
-- HT estimator: Σ(y_i / π_i) / Σ(1 / π_i)
-- Equivalent to: weighted_positive_count / effective_sample_size
ht_estimates AS (
  SELECT
    week,
    harm_vertical,
    -- Numerator: sum of weights for true positive items
    SUM(IF(true_harmful, sample_weight, 0))   AS ht_num,
    -- Denominator: total effective sample (corpus-representative)
    SUM(sample_weight)                        AS ht_denom,
    SUM(IF(true_harmful, sample_weight, 0))
      / NULLIF(SUM(sample_weight), 0)         AS ht_prevalence,
    COUNT(*)                                  AS n_reviewed
  FROM gold_review_samples
  WHERE reviewed_at IS NOT NULL
  GROUP BY week, harm_vertical
),

-- ── Step 3: Observed classifier detection rate on full corpus ──────────────
observed_rate AS (
  SELECT
    week,
    classifier_label                          AS harm_vertical,
    COUNTIF(classifier_score >= 0.5)
      / NULLIF(COUNT(*), 0)                   AS observed_positive_rate,
    COUNT(*)                                  AS n_corpus
  FROM content_items
  WHERE classifier_label IS NOT NULL
  GROUP BY week, harm_vertical
),

-- ── Step 4: Classifier-adjusted prevalence estimate ───────────────────────
adjusted_estimates AS (
  SELECT
    o.week,
    o.harm_vertical,
    o.observed_positive_rate               AS q_hat,
    p.tpr,
    p.fpr,
    o.n_corpus,
    -- Adjusted: π̂ = (q̂ - FPR) / (TPR - FPR)
    SAFE_DIVIDE(
      o.observed_positive_rate - p.fpr,
      p.tpr - p.fpr
    )                                      AS adjusted_prevalence,
    -- Delta-method SE approximation (variance from q̂ only, SE from TPR/FPR ignored)
    SQRT(
      SAFE_DIVIDE(
        o.observed_positive_rate * (1 - o.observed_positive_rate),
        o.n_corpus
      )
    ) / NULLIF(ABS(p.tpr - p.fpr), 0)     AS adjusted_se
  FROM observed_rate o
  JOIN classifier_performance p
    ON o.week = p.week AND o.harm_vertical = p.harm_vertical
),

-- ── Step 5: Combine all estimates ─────────────────────────────────────────
combined AS (
  SELECT
    ht.week,
    ht.harm_vertical,
    -- HT (gold standard, design-unbiased)
    ROUND(ht.ht_prevalence, 6)             AS ht_prevalence,
    -- 95% CI using Wilson-like normal approximation on weighted estimate
    ROUND(ht.ht_prevalence - 1.96 * SQRT(
      SAFE_DIVIDE(ht.ht_prevalence * (1 - ht.ht_prevalence), ht.n_reviewed)
    ), 6)                                  AS ht_ci_lower,
    ROUND(ht.ht_prevalence + 1.96 * SQRT(
      SAFE_DIVIDE(ht.ht_prevalence * (1 - ht.ht_prevalence), ht.n_reviewed)
    ), 6)                                  AS ht_ci_upper,
    -- Classifier-adjusted estimate
    ROUND(
      GREATEST(0, LEAST(1, adj.adjusted_prevalence)), 6
    )                                      AS adjusted_prevalence,
    ROUND(
      GREATEST(0, adj.adjusted_prevalence - 1.96 * adj.adjusted_se), 6
    )                                      AS adj_ci_lower,
    ROUND(
      LEAST(1, adj.adjusted_prevalence + 1.96 * adj.adjusted_se), 6
    )                                      AS adj_ci_upper,
    -- Naive observed rate (do NOT use for prevalence — for reference only)
    ROUND(adj.q_hat, 6)                    AS naive_observed_rate,
    -- Metadata
    adj.tpr,
    adj.fpr,
    adj.n_corpus,
    ht.n_reviewed,
    CURRENT_TIMESTAMP()                    AS computed_at
  FROM ht_estimates ht
  LEFT JOIN adjusted_estimates adj
    ON ht.week = adj.week AND ht.harm_vertical = adj.harm_vertical
),

-- ── Step 6: Add 4-week rolling average for trend smoothing ─────────────────
rolling AS (
  SELECT
    *,
    AVG(ht_prevalence) OVER (
      PARTITION BY harm_vertical
      ORDER BY week
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    )                                      AS ht_prevalence_4w_avg,
    AVG(adjusted_prevalence) OVER (
      PARTITION BY harm_vertical
      ORDER BY week
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    )                                      AS adjusted_prevalence_4w_avg
  FROM combined
)

-- ── Final output ───────────────────────────────────────────────────────────
SELECT *
FROM rolling
ORDER BY harm_vertical, week
;

/*
  Example downstream usage — alert on significant week-over-week spike:

  SELECT
    week,
    harm_vertical,
    ht_prevalence,
    LAG(ht_prevalence) OVER (PARTITION BY harm_vertical ORDER BY week) AS prev_week,
    (ht_prevalence - LAG(ht_prevalence) OVER (PARTITION BY harm_vertical ORDER BY week))
      / NULLIF(LAG(ht_ci_upper - ht_ci_lower) OVER (PARTITION BY harm_vertical ORDER BY week) / 2, 0)
    AS z_score
  FROM rolling
  HAVING ABS(z_score) > 2.5
  ORDER BY week DESC;
*/
