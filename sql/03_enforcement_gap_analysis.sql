/*
  03_enforcement_gap_analysis.sql
  ──────────────────────────────────────────────────────────────────────────────
  Enforcement gap analysis: measure the fraction of detected harm that
  receives a corresponding enforcement action within a response window.

  Key metrics:
    - detection_count      : items with classifier_score >= threshold
    - enforced_count       : detected items with an enforcement action within
                             the response window (default: 7 days)
    - enforcement_rate     : enforced_count / detection_count
    - gap_count            : detected items with NO action (false negatives of
                             the enforcement pipeline)
    - gap_rate             : gap_count / detection_count
    - estimated_harm_missed: gap_count adjusted for classifier FPR
                             (gap items that are probably true harmful)

  Segmented by:
    - harm_vertical
    - risk_tier (actor-level)
    - actioned_by (classifier / human / policy_rule)

  This analysis surfaces the "two-stage" failure mode:
    Stage 1 — classifier misses harm (FN at score threshold)
    Stage 2 — classifier detects harm but enforcement pipeline drops it
              (policy decision, queue overflow, review backlog)
*/

DECLARE @analysis_start DATE DEFAULT DATE_SUB(CURRENT_DATE(), INTERVAL 12 WEEK);
DECLARE @analysis_end   DATE DEFAULT CURRENT_DATE();
DECLARE @score_threshold FLOAT64 DEFAULT 0.5;
DECLARE @response_window_days INT64 DEFAULT 7;

-- ── Step 1: Pull all classifier detections in the window ──────────────────
WITH detections AS (
  SELECT
    ci.item_id,
    ci.actor_id,
    ci.week,
    ci.created_at,
    ci.risk_tier                              AS content_risk_tier,
    ci.classifier_label                       AS harm_vertical,
    ci.classifier_score,
    a.risk_tier                               AS actor_risk_tier,
    a.composite_risk_score
  FROM content_items ci
  LEFT JOIN actors a USING (actor_id)
  WHERE ci.week BETWEEN @analysis_start AND @analysis_end
    AND ci.classifier_score >= @score_threshold
    AND ci.classifier_label IS NOT NULL
),

-- ── Step 2: Find matching enforcement actions ──────────────────────────────
-- An enforcement action "matches" a detection if:
--   - Same item_id, AND
--   - Action taken within @response_window_days of content creation
enforcement_matched AS (
  SELECT
    d.item_id,
    d.harm_vertical,
    d.week,
    d.content_risk_tier,
    d.actor_risk_tier,
    d.classifier_score,
    d.composite_risk_score,
    -- Enforcement attributes (NULL if no action taken)
    e.action_type,
    e.actioned_by,
    e.actioned_at,
    -- Flag: was there an enforcement action within the response window?
    CASE
      WHEN e.action_id IS NOT NULL
        AND TIMESTAMP_DIFF(e.actioned_at, d.created_at, DAY) <= @response_window_days
      THEN TRUE
      ELSE FALSE
    END AS was_enforced,
    -- Time-to-action in hours (NULL if no action)
    CASE
      WHEN e.action_id IS NOT NULL
      THEN TIMESTAMP_DIFF(e.actioned_at, d.created_at, HOUR)
    END AS hours_to_action
  FROM detections d
  LEFT JOIN enforcement_actions e
    ON d.item_id = e.item_id
   AND e.harm_vertical = d.harm_vertical
),

-- ── Step 3: Deduplicate (item may have multiple actions; keep earliest) ────
deduped AS (
  SELECT
    item_id,
    harm_vertical,
    week,
    content_risk_tier,
    actor_risk_tier,
    classifier_score,
    composite_risk_score,
    -- Take the earliest action's attributes
    MAX(was_enforced)                         AS was_enforced,
    MIN(hours_to_action)                      AS min_hours_to_action,
    STRING_AGG(DISTINCT action_type)          AS action_types_taken,
    STRING_AGG(DISTINCT actioned_by)          AS enforced_by_channels
  FROM enforcement_matched
  GROUP BY
    item_id, harm_vertical, week, content_risk_tier,
    actor_risk_tier, classifier_score, composite_risk_score
),

-- ── Step 4: Aggregate by vertical + risk tier + week ──────────────────────
aggregated AS (
  SELECT
    week,
    harm_vertical,
    content_risk_tier,
    actor_risk_tier,
    COUNT(*)                                            AS detection_count,
    COUNTIF(was_enforced)                               AS enforced_count,
    COUNTIF(NOT was_enforced)                           AS gap_count,

    -- Enforcement rate
    ROUND(
      SAFE_DIVIDE(COUNTIF(was_enforced), COUNT(*)), 4
    )                                                   AS enforcement_rate,

    -- Gap rate (fraction of detections without action)
    ROUND(
      SAFE_DIVIDE(COUNTIF(NOT was_enforced), COUNT(*)), 4
    )                                                   AS gap_rate,

    -- Median time-to-action for enforced items (hours)
    APPROX_QUANTILES(
      IF(was_enforced, min_hours_to_action, NULL), 100
    )[OFFSET(50)]                                       AS median_hours_to_action,

    -- 95th pct response time
    APPROX_QUANTILES(
      IF(was_enforced, min_hours_to_action, NULL), 100
    )[OFFSET(95)]                                       AS p95_hours_to_action,

    -- High-confidence gap items (score >= 0.8, not enforced) — most likely
    -- true positives left unactioned; severity = highest priority backlog
    COUNTIF(NOT was_enforced AND classifier_score >= 0.80)
                                                        AS high_confidence_gap_count,
    ROUND(
      SAFE_DIVIDE(
        COUNTIF(NOT was_enforced AND classifier_score >= 0.80),
        COUNTIF(classifier_score >= 0.80)
      ), 4
    )                                                   AS high_confidence_gap_rate

  FROM deduped
  GROUP BY week, harm_vertical, content_risk_tier, actor_risk_tier
),

-- ── Step 5: Add rolling 4-week enforcement rate (trend) ───────────────────
with_rolling AS (
  SELECT
    *,
    AVG(enforcement_rate) OVER (
      PARTITION BY harm_vertical, content_risk_tier, actor_risk_tier
      ORDER BY week
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    )                                                   AS enforcement_rate_4w_avg,
    AVG(gap_rate) OVER (
      PARTITION BY harm_vertical, content_risk_tier, actor_risk_tier
      ORDER BY week
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    )                                                   AS gap_rate_4w_avg
  FROM aggregated
)

-- ── Final output ───────────────────────────────────────────────────────────
SELECT
  week,
  harm_vertical,
  content_risk_tier,
  actor_risk_tier,
  detection_count,
  enforced_count,
  gap_count,
  enforcement_rate,
  gap_rate,
  enforcement_rate_4w_avg,
  gap_rate_4w_avg,
  high_confidence_gap_count,
  high_confidence_gap_rate,
  median_hours_to_action,
  p95_hours_to_action,

  -- Alert flag: gap rate worsened >10pp vs 4-week baseline
  CASE
    WHEN gap_rate - gap_rate_4w_avg > 0.10 THEN 'REGRESSION'
    WHEN gap_rate - gap_rate_4w_avg < -0.10 THEN 'IMPROVEMENT'
    ELSE 'STABLE'
  END                                                   AS trend_flag

FROM with_rolling
ORDER BY week DESC, harm_vertical, content_risk_tier, actor_risk_tier
;

/*
  ── Summary view: overall enforcement gap by vertical ─────────────────────

  SELECT
    harm_vertical,
    SUM(detection_count)         AS total_detected,
    SUM(gap_count)               AS total_gaps,
    ROUND(
      SAFE_DIVIDE(SUM(gap_count), SUM(detection_count)), 4
    )                            AS overall_gap_rate,
    SUM(high_confidence_gap_count) AS high_conf_gaps
  FROM with_rolling
  WHERE week >= DATE_SUB(CURRENT_DATE(), INTERVAL 4 WEEK)
  GROUP BY harm_vertical
  ORDER BY overall_gap_rate DESC;

  ── Interpretation guide ───────────────────────────────────────────────────
  gap_rate > 0.30  → URGENT: 30%+ of detected harm has no enforcement response
  gap_rate 0.10-0.30 → Review queue backlog or policy threshold mismatch
  gap_rate < 0.10  → Expected noise; monitor for trend changes
*/
