# SQL Examples

Production-grade SQL patterns for harm prevalence measurement at scale.

These queries assume a data warehouse schema typical of a large content platform
(BigQuery / Snowflake dialect). They demonstrate the end-to-end workflow from
raw content logs to weekly executive dashboards.

## Schema

### `content_items`
Raw content corpus with classifier scores.

| Column | Type | Description |
|---|---|---|
| `item_id` | STRING | Unique content identifier |
| `actor_id` | STRING | Author / account identifier |
| `created_at` | TIMESTAMP | Content creation time |
| `content_type` | STRING | post / comment / message / media |
| `risk_tier` | STRING | high / medium / low (pre-assigned by routing rules) |
| `classifier_score` | FLOAT64 | LLM harm probability [0, 1] |
| `classifier_label` | STRING | Predicted harm category or NULL |
| `sample_weight` | FLOAT64 | Inverse sampling probability for stratified design |
| `reviewed_at` | TIMESTAMP | Timestamp of human review (NULL if not reviewed) |
| `true_label` | STRING | Gold-standard label from review (NULL if not reviewed) |
| `week` | DATE | Truncated to Monday of the content week |

### `enforcement_actions`
Actions taken on content items and actors.

| Column | Type | Description |
|---|---|---|
| `action_id` | STRING | Unique action identifier |
| `item_id` | STRING | FK → content_items |
| `actor_id` | STRING | FK → actors |
| `action_type` | STRING | remove / suspend / restrict / warn |
| `harm_vertical` | STRING | sexual_content_minors / violent_extremism / etc. |
| `actioned_at` | TIMESTAMP | When action was taken |
| `actioned_by` | STRING | classifier / human / policy_rule |
| `week` | DATE | Truncated to Monday of action week |

### `actors`
Account-level behavioral features (updated nightly).

| Column | Type | Description |
|---|---|---|
| `actor_id` | STRING | Unique account identifier |
| `account_age_days` | INT64 | Days since account creation |
| `post_count_7d` | INT64 | Posts in past 7 days |
| `post_count_30d` | INT64 | Posts in past 30 days |
| `unique_content_ratio` | FLOAT64 | Fraction of non-duplicate posts |
| `api_usage_fraction` | FLOAT64 | API vs UI session ratio |
| `prior_enforcement_count` | INT64 | Historical actions on this account |
| `risk_tier` | STRING | high / medium / low (derived from actor risk model) |
| `composite_risk_score` | FLOAT64 | Weighted actor risk score [0, 1] |

### `gold_review_samples`
Stratified human-review sample with ground-truth labels.

| Column | Type | Description |
|---|---|---|
| `sample_id` | STRING | Unique sample identifier |
| `item_id` | STRING | FK → content_items |
| `week` | DATE | Sampling week |
| `stratum` | STRING | Stratum used for sampling |
| `sample_weight` | FLOAT64 | Inverse inclusion probability |
| `true_harmful` | BOOL | Gold-standard human label |
| `predicted_harmful` | BOOL | Classifier prediction at threshold 0.5 |
| `harm_vertical` | STRING | Specific harm category |

## Query Files

| File | Purpose |
|---|---|
| `01_sampling_frame.sql` | Build stratified sampling frame for weekly review |
| `02_weekly_prevalence_rollup.sql` | Compute weighted prevalence estimates by vertical |
| `03_enforcement_gap_analysis.sql` | Measure gap between detected harm and enforcement |

## Usage Notes

- All prevalence estimates use the **Horvitz–Thompson estimator** (weighted sum of
  true labels divided by weighted sum of sample weights) to correct for the
  stratified design.
- The enforcement gap query joins on `item_id`; items with classifier detections
  but no corresponding enforcement action within 7 days are counted as gaps.
- Replace `<PROJECT>.<DATASET>` with your actual warehouse schema prefix.
