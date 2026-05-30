/*
  01_sampling_frame.sql
  ──────────────────────────────────────────────────────────────────────────────
  Build the stratified sampling frame for weekly gold-standard review.

  Strategy: Risk-stratified sampling
    - high_risk   stratum: 60% of review budget
    - medium_risk stratum: 30% of review budget
    - low_risk    stratum: 10% of review budget

  Each item gets an inverse-probability weight (sample_weight) so that
  the Horvitz–Thompson estimator can recover unbiased corpus-level estimates.

  Outputs a table that the review tooling reads to assign items to annotators.

  Parameters (adjust for each weekly run):
    @review_week   DATE    Monday of the target week
    @total_budget  INT64   Total items to assign for review this week
*/

DECLARE @review_week DATE DEFAULT DATE_TRUNC(CURRENT_DATE(), WEEK(MONDAY));
DECLARE @total_budget INT64 DEFAULT 3000;

-- ── Step 1: Compute stratum sizes from the corpus ──────────────────────────
WITH corpus_week AS (
  SELECT
    item_id,
    actor_id,
    risk_tier,
    classifier_score,
    classifier_label,
    CASE
      WHEN risk_tier = 'high'   THEN 'high_risk'
      WHEN risk_tier = 'medium' THEN 'medium_risk'
      ELSE                           'low_risk'
    END AS stratum
  FROM content_items
  WHERE week = @review_week
    AND reviewed_at IS NULL          -- exclude already-reviewed items
),

stratum_counts AS (
  SELECT
    stratum,
    COUNT(*) AS N_h,                 -- stratum population size
    -- Budget allocation: 60/30/10 split
    CASE stratum
      WHEN 'high_risk'   THEN CAST(@total_budget * 0.60 AS INT64)
      WHEN 'medium_risk' THEN CAST(@total_budget * 0.30 AS INT64)
      WHEN 'low_risk'    THEN CAST(@total_budget * 0.10 AS INT64)
    END AS n_h                       -- target sample size for stratum h
  FROM corpus_week
  GROUP BY stratum
),

-- ── Step 2: Compute inclusion probabilities ────────────────────────────────
stratum_with_probs AS (
  SELECT
    stratum,
    N_h,
    n_h,
    -- Cap inclusion probability at 1.0 (census for small strata)
    LEAST(SAFE_DIVIDE(n_h, N_h), 1.0) AS inclusion_prob
  FROM stratum_counts
),

-- ── Step 3: Systematic sample within each stratum ─────────────────────────
-- Use a consistent hash for reproducibility across runs.
-- Items are ranked by hash(item_id || week) so the sample is stable.
ranked_items AS (
  SELECT
    c.item_id,
    c.actor_id,
    c.stratum,
    c.classifier_score,
    c.classifier_label,
    s.N_h,
    s.n_h,
    s.inclusion_prob,
    ROW_NUMBER() OVER (
      PARTITION BY c.stratum
      ORDER BY FARM_FINGERPRINT(CONCAT(c.item_id, CAST(@review_week AS STRING)))
    ) AS rn
  FROM corpus_week c
  JOIN stratum_with_probs s USING (stratum)
),

-- ── Step 4: Select items and attach Horvitz–Thompson weights ───────────────
sampled_items AS (
  SELECT
    item_id,
    actor_id,
    stratum,
    classifier_score,
    classifier_label,
    inclusion_prob,
    -- Horvitz–Thompson weight: 1 / π_i
    -- Correct for oversampling so corpus-level sums are unbiased
    ROUND(1.0 / inclusion_prob, 4)  AS sample_weight,
    @review_week                    AS review_week
  FROM ranked_items
  WHERE rn <= n_h
)

-- ── Final output ───────────────────────────────────────────────────────────
SELECT
  GENERATE_UUID()          AS sample_id,
  item_id,
  actor_id,
  stratum,
  classifier_score,
  classifier_label,
  inclusion_prob,
  sample_weight,
  review_week,
  CURRENT_TIMESTAMP()      AS queued_at,
  NULL                     AS assigned_to,
  NULL                     AS reviewed_at,
  NULL                     AS true_harmful,
  NULL                     AS harm_vertical
FROM sampled_items
ORDER BY stratum, RAND()   -- randomize assignment order within stratum
;

/*
  Validation checks (run after INSERT to gold_review_samples):

  -- Total budget check
  SELECT stratum, COUNT(*) AS n_sampled
  FROM gold_review_samples
  WHERE week = @review_week
  GROUP BY stratum;

  -- Verify weights are positive
  SELECT MIN(sample_weight) AS min_w, MAX(sample_weight) AS max_w
  FROM gold_review_samples
  WHERE week = @review_week;
*/
