# Notebooks — Guidance & Reference

All notebooks run in sequence from `00` through `09`, each building on the previous.
They can also be run standalone: every notebook generates its own synthetic data if no
real dataset or API key is available.

**Run order:**
```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

**Quick start:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add API key(s)
jupyter notebook notebooks/
```

---

## 00 — Data Exploration (`00_data_exploration.ipynb`)

**Purpose:** Understand the BeaverTails dataset structure and establish why naive random
sampling cannot measure harm prevalence accurately.

**Prerequisites:** No API key required. Downloads BeaverTails from HuggingFace at runtime
(~1 GB). Falls back to `src/simulation.py` synthetic data if the download fails.

### Sections

**1. Load BeaverTails**
Loads the `PKU-Alignment/BeaverTails` dataset via the HuggingFace `datasets` library.
Inspects the schema: 333,963 QA pairs, 14 binary harm category columns, a `is_safe` flag,
and the raw `prompt`/`response` text. Prints row count, column types, and a sample row.

**2. Label Distribution**
Bar chart of the fraction of items flagged per harm category. Shows that harm labels are
highly imbalanced — `hate_speech` and `non_violent_unethical_behavior` are the most
common; `animal_abuse` and `terrorism` are the rarest. Highlights that most categories
fall below 5% prevalence, meaning naive flagging will be dominated by false positives.

**3. Why Naive Random Sampling Fails**
Quantitative demonstration of the base-rate problem: even a highly accurate classifier
(TPR=95%, FPR=1%) produces a positive predictive value below 50% when prevalence is under
5%. Plots PPV as a function of prevalence for three classifier quality levels. The
takeaway: a 2% flag rate does not mean 2% prevalence.

**4. Correlation Between Harm Verticals**
Heatmap of pairwise Pearson correlations between harm category labels. Shows that harm
verticals tend to co-occur — an item labeled `violent_speech` often also carries
`hate_speech`. This correlation motivates stratified sampling: items flagged by any signal
are enriched for all harm types, making a risk-stratified sample more efficient than SRS.

**5. Key Takeaways**
Markdown summary of three implications for measurement design:
- Stratified sampling is necessary, not optional
- The raw flag rate is an unreliable prevalence proxy
- Vertical-specific TPR/FPR estimation is required because classifiers behave differently
  across harm types

**Key outputs:** Console distribution table, base-rate PPV plot, correlation heatmap.

---

## 01 — LLM Classifier (`01_llm_classifier.ipynb`)

**Purpose:** Run a real LLM (Claude, GPT, or Groq) as a harm classifier on a stratified
sample, evaluate its accuracy, compare prompting strategies and providers, and estimate
per-label cost.

**Prerequisites:** At least one API key in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
or `GROQ_API_KEY`). Without a key, the notebook generates synthetic classification outputs
via `src/simulation.py`.

Set the provider at the top of the notebook:
```python
PROVIDER = 'anthropic'   # or 'openai' or 'groq'
```

### Sections

**1. Load Stratified Sample**
Loads the stratified sample produced by notebook 02 (or generates a fresh one). Inspects
the distribution of harm verticals across high-risk and low-risk strata to confirm that
the sample is appropriately enriched in the high-risk stratum.

**2. Run LLM Classifier**
Calls `src/classifier.HarmClassifier` in batch with exponential-backoff retry logic.
Each item produces a `ClassificationResult` with fields: `label` (bool), `confidence`
(float 0–1), `key_signals` (list of strings), `reasoning` (short paragraph), `provider`,
and `tokens_used`. Results are written to `data/processed/01_classifier_results.jsonl`.

**3. Evaluation Metrics**
Computes TPR, FPR, precision, recall, F1, and AUC-ROC against BeaverTails ground-truth
labels. Displays a confusion matrix and a ROC curve. Typical results: TPR ≈ 82%,
FPR ≈ 4–8% depending on provider and harm vertical.

**4. Prompt Sensitivity Analysis**
Compares three prompting strategies on a held-out 200-item set:
- Zero-shot with policy definition
- Few-shot with 3 positive and 3 negative examples
- Chain-of-thought with explicit reasoning step

Reports TPR/FPR deltas. Few-shot typically improves FPR by 1–2pp with negligible TPR
change. Chain-of-thought improves precision for borderline cases.

**5. Cost Analysis (Provider Comparison)**
Calculates cost per 1,000 items and cost per label for each provider based on token counts.
Shows that Groq (Llama-3.1-8b-instant) is effectively free-tier; GPT-4o-mini costs roughly
$0.30/1K items; Claude Sonnet costs roughly $1.50/1K items but yields the highest accuracy.
Plots cost vs F1 tradeoff.

**6. Summary**
Markdown table comparing all three providers on: TPR, FPR, AUC-ROC, cost/1K items, and
recommended use case. Recommends Claude for production prevalence measurement and Groq for
rapid prototyping.

**Key outputs:** `data/processed/01_classifier_results.jsonl`, ROC curve, cost comparison
table.

---

## 02 — Sampling Design (`02_sampling_design.ipynb`)

**Purpose:** Compare simple random sampling (SRS) against three risk-stratified designs
and show how stratification reduces the sample size required to detect a given prevalence
level with a target CI width.

**Prerequisites:** None. All data is generated by `src/simulation.py` and `src/sampling.py`.

### Sections

**1. Baseline SRS**
Simulates SRS from a 10M-item corpus at 1% true prevalence. Demonstrates that SRS requires
roughly 9,600 items to achieve a ±0.2pp CI, and that most sampled items will be true
negatives — wasted review budget.

**2. Stratified Sampling (Three Designs)**
Uses `src/sampling.StratifiedHarmSampler` to implement and compare:
- **Equal allocation:** Equal budget across strata (simple but inefficient)
- **Proportional allocation:** Budget proportional to stratum size (better coverage)
- **Neyman optimal allocation:** Budget proportional to stratum size × within-stratum SD
  (optimal for variance reduction under fixed total cost)

Reports stratum sample sizes, estimated prevalence, and CI width for each design.

**3. Visualization**
Side-by-side bar charts showing allocated sample per stratum for each design, and a CI
width comparison plot. Neyman allocation concentrates budget in the high-risk stratum
where variance is highest, producing the narrowest CI for a fixed total n.

**4. Sample Size × CI Width × Cost Tradeoff**
Plots CI width as a function of total sample size for each design, overlaid with an
estimated cost curve (review cost assumed at $2/item). Shows the point of diminishing
returns: beyond ~5,000 items, CI width improvement per additional review dollar is minimal
under Neyman allocation.

**5. Design Recommendations**
Markdown table with recommended design for each scenario:
- Small budget (<500 items): use Neyman optimal, focus on high-risk stratum only
- Medium budget (500–5,000): full Neyman stratification across all risk tiers
- Large budget (>5,000): consider capture-recapture to leverage two detection systems

**Key outputs:** Stratum allocation table, CI width comparison plot.

---

## 03 — Prevalence Estimation (`03_prevalence_estimation.ipynb`)

**Purpose:** Validate all three prevalence estimators against known ground truth via
Monte Carlo simulation, then apply them to the five harm verticals from BeaverTails.

**Prerequisites:** None. Uses `src/simulation.py` for validation; optionally uses
classifier outputs from notebook 01 for the applied estimation.

### Sections

**1. Estimator Validation via Simulation**
Runs 500 Monte Carlo trials for each estimator at five true prevalence levels (0.1%, 0.5%,
1%, 2%, 5%). For each trial, generates a synthetic corpus, draws a stratified sample, and
applies all three estimators. Reports empirical CI coverage, RMSE, and bias.

Key results:
- Direct proportion: 95.2% coverage, near-zero bias (requires gold labels)
- Classifier-adjusted: 94.6% coverage, near-zero bias (requires known TPR/FPR)
- Raw Chapman: poor coverage at low prevalence due to false-positive contamination

**2. Apply All Estimators Across Harm Verticals**
Applies all three estimators to the five harm verticals using:
- TPR/FPR from notebook 01 classifier evaluation
- Stratified sample from notebook 02
- Capture lists from both detection systems in `src/simulation`

Produces a results table with point estimate and 95% CI per vertical per estimator.

**3. Visualization**
Forest plot: one row per harm vertical, showing point estimates and CI bars for all three
estimators side-by-side. Horizontal dashed line at the true prevalence (from BeaverTails
labels) for visual calibration check.

**4. Sensitivity Analysis (TPR/FPR)**
Sweeps TPR from 0.60 to 0.95 and FPR from 0.01 to 0.15, holding other parameters fixed.
Shows how CI width and bias of the classifier-adjusted estimator respond to classifier
quality degradation. Key finding: FPR error is more damaging than TPR error at low base
rates.

**5. Final Comparison Table**
Summary table replicating the README findings:

| Harm Vertical | True Prevalence | Naive Flag Rate | Classifier-Adjusted | 95% CI |
|---|---|---|---|---|
| sexual_content_minors | 0.40% | 6.77% | 1.01% | [0.00%, 3.82%] |
| violent_extremism | 1.10% | 7.33% | 1.75% | [0.00%, 4.57%] |
| ... | | | | |

**6. Key Takeaways**
Documents the three-step production workflow: calibrate classifier → estimate TPR/FPR from
gold review → apply classifier-adjusted estimator. Notes that raw flag rate overestimates
prevalence by 3–17× across verticals.

**Key outputs:** Monte Carlo coverage table, forest plot, sensitivity heatmap.

---

## 04 — Calibration Analysis (`04_calibration_analysis.ipynb`)

**Purpose:** Show that raw LLM confidence scores are miscalibrated (ECE ≈ 10–18%),
calibrate them using Platt scaling and isotonic regression, and demonstrate how
calibration error propagates into prevalence CI width.

**Prerequisites:** None. All data is synthetic.

### Sections

**1. Generate Calibration Dataset**
Simulates 5,000 items with a deliberately overconfident classifier: true positives get
scores from Beta(6, 1.5) (clustered near 1.0) and negatives from Beta(1.5, 6) (clustered
near 0.0). The raw ECE is ~0.18 — far too high for reliable prevalence measurement.
Splits 70/30 into calibration and test sets.

**2. Reliability Diagram (Before Calibration)**
Three-panel reliability diagram comparing:
- Uncalibrated scores (shaded red region shows miscalibration gap)
- Platt-scaled scores
- Isotonic regression scores

Reports ECE and MCE (Maximum Calibration Error) for each. Both calibration methods reduce
ECE from ~0.18 to ~0.02–0.04 on the test set.

**3. Score Distributions Before and After Calibration**
Overlapping histograms of true positive vs true negative score distributions for each
method. Calibration reduces the extreme bi-modal clustering (scores near 0 or 1) and
produces distributions that better reflect true label rates at each confidence level.

**4. Calibration Error → Prevalence CI Widening**
Quantifies the amplification effect: plots CI width amplification factor (calibrated vs
uncalibrated) as a function of ECE for four prevalence levels (0.1%, 1%, 5%, 15%).

Key finding: at 0.1% prevalence, a 5% ECE causes a 7× widening of the prevalence CI.
At 15% prevalence, the same ECE causes only a 4× widening. Low base rates are
disproportionately harmed by calibration error.

Sensitivity table: ECE × prevalence → amplification factor, for quick lookup.

**5. Practical Implications**
Concrete example using `sexual_content_minors` (π = 0.4%):
- Without calibration (ECE = 12%): CI width = ±15.8%, amplification = 17×
- After Platt scaling (ECE = 2%): CI width = ±3.1%, amplification = 3×
- Net reduction: 82% narrower CIs

**6. Takeaways**
Production workflow table:
1. Train detection system
2. Calibrate on ≥500-item gold standard review set
3. Re-calibrate quarterly or after model updates
4. Report ECE alongside every prevalence estimate

Recommends Platt scaling for calibration sets under 1,000 items; isotonic regression for
larger sets.

**Key outputs:** `data/processed/04_reliability_diagrams.png`,
`data/processed/04_calibration_amplification.png`, sensitivity table.

---

## 05 — Operational Dashboard (`05_dashboard_mockup.ipynb`)

**Purpose:** Simulate 52 weeks of platform data and render the operational integrity
metrics dashboard that a trust & safety team would review weekly.

**Prerequisites:** None. All data is synthetic.

### Sections

**1. Simulate 52 Weeks of Platform Data**
Generates weekly corpus snapshots (100K items/week) across three regimes:
- Weeks 1–26: baseline prevalence (1.5%)
- Weeks 27–35: spike (+50% to 2.3%) — simulating a bad actor campaign
- Weeks 36–52: return to baseline after intervention

For each week: draws a 2,000-item stratified sample, runs the Chapman capture-recapture
estimator, records point estimate and 95% CI, and logs detection and enforcement counts.

**2. Anomaly Detection: Rolling Z-Score**
Computes a rolling 8-week mean and standard deviation of estimated prevalence. Flags weeks
where `|z-score| > 2.0`. The Z-score anomaly correctly fires at the start of the spike
regime (1-week detection lag). Output: list of anomaly weeks.

**3. Dashboard Panels**
Five-panel matplotlib figure saved to `data/processed/05_integrity_dashboard.png`:

| Panel | Content |
|-------|---------|
| Top (full width) | Weekly prevalence trend — estimated + true + 95% CI band, spike period shaded, anomaly markers |
| Middle left | Rolling Z-score bar chart with ±2σ threshold lines |
| Middle right | Weekly detection rate (System A) over time |
| Bottom left | Stacked bar: weekly detections vs enforcement actions |
| Bottom right | 7-day vs 30-day rolling prevalence average |

**4. Executive Summary Table**
Last-4-week KPI summary:
- Average prevalence and 95% CI
- Trend vs prior 4 weeks (percentage point delta)
- Average weekly detections
- Enforcement rate (target: ≥75%)
- Anomaly count

**5. CUSUM Control Chart**
Implements a CUSUM (Cumulative Sum) control chart using:
- Target: mean prevalence from first 8 weeks
- Allowance (slack): 0.2pp
- Alarm threshold: 1.0pp cumulative excess

CUSUM detects the bad actor campaign 1–2 weeks earlier than Z-score because it accumulates
evidence across weeks rather than comparing to a rolling window. Saved to
`data/processed/05_cusum_chart.png`.

**6. Takeaways for Integrity Teams**
Comparison table: Z-score vs CUSUM detection lag, CI coverage validation, enforcement
rate against SLA. Recommends running both monitors in parallel: Z-score for burst
detection (sudden spikes), CUSUM for ramp-up detection (gradual campaigns).

**Key outputs:** `data/processed/05_integrity_dashboard.png`,
`data/processed/05_cusum_chart.png`.

---

## 06 — Actor Network Features (`06_actor_network_features.ipynb`)

**Purpose:** Shift from content-level to actor-level analysis. Engineer behavioral risk
features for each actor, detect coordinated inauthentic behavior networks, and show that
actor risk scoring improves sample efficiency over content-only classifiers.

**Prerequisites:** None. Uses `src/actor.ActorFeatureExtractor` and
`src/actor.simulate_actor_corpus`.

### Sections

**1. Simulate Actor Corpus**
Calls `src/actor.simulate_actor_corpus` to generate a synthetic platform with three
actor populations:
- Good actors (85%): low-risk behavioral profiles
- Solo bad actors (10%): high-risk signals but no coordination
- Network bad actors (5%): coordinated, similar behavioral fingerprints

Output DataFrame has columns: `actor_id`, `is_bad_actor`, `is_network_actor`,
`true_label`, plus raw behavioral metrics (post count, velocity, repetition rate,
enforcement history, account age).

**2. Feature Distributions**
Box plots and violin plots comparing each raw behavioral feature across the three actor
classes. Shows that velocity, repetition rate, and enforcement score are most
discriminative; account age is useful but overlaps significantly between classes.

**3. Compute Risk Scores**
Calls `src/actor.compute_risk_scores` to compute the seven component scores and the
composite weighted score for each actor:
- `velocity_score`, `newness_score`, `repetition_score`, `automation_score`
- `enforcement_score`, `coordination_score`, `composite_score`

Prints the weight vector and displays the score distribution per actor class.

**4. Component Score Breakdown**
Stacked bar chart showing the contribution of each component score to the composite for
good actors, solo bad actors, and network bad actors. Network bad actors score highest on
coordination and repetition; solo bad actors on velocity and enforcement.

**5. ROC vs Content-Only Baseline**
Compares ROC curves (AUC) for:
- Content-only classifier (from notebook 01)
- Actor risk score only
- Combined: content classifier × actor risk score (product rule)

Actor risk score alone typically achieves AUC ≈ 0.85 vs AUC ≈ 0.82 for the content
classifier. The combined signal reaches AUC ≈ 0.91.

**6. Sampling Efficiency Comparison**
Demonstrates that using actor risk score as a pre-filter before the stratified sample
reduces required sample size by ~30–40% for the same CI width, because high-risk actors
are much more likely to produce violating content.

**7. Coordination Network Detection**
Calls `src/actor.detect_coordination_networks` with cosine similarity threshold = 0.85
and minimum cluster size = 3. Prints detected clusters: actor count, mean risk score,
fraction that are true bad actors. Adds `network_cluster_id` column to the actor
DataFrame.

**8. Cluster Analysis**
Summary table of all detected clusters: size, centroid feature vector, precision (fraction
of cluster members who are true bad actors), and comparison to the overall bad actor
precision of the classifier. Shows that network clusters have 2–3× higher precision than
solo risk score thresholding.

**Key outputs:** Feature distribution plots, ROC comparison, cluster summary table.

---

## 07 — A/B Test Design (`07_ab_test_design.ipynb`)

**Purpose:** Design statistically rigorous A/B tests that use prevalence as the primary
outcome metric. Covers power analysis, MDE curves, proxy metrics, sequential testing,
and a worked end-to-end example.

**Prerequisites:** None. All calculations are analytical or simulation-based.

### Sections

**1. Power Analysis Fundamentals**
Reviews the four parameters of power analysis: significance level (α), power (1−β),
minimum detectable effect (MDE), and sample size. Implements `power_for_prevalence_test`
using the two-proportion Z-test. Shows that prevalence tests require larger samples than
engagement tests because the effect is measured on a rare outcome.

**2. MDE Curves (Corpus Size vs Review Budget)**
Plots the minimum detectable absolute effect (in percentage points) as a function of:
- Total corpus size (x-axis)
- Review budget (color scale: 1K, 5K, 10K, 25K items)

At 1% baseline prevalence with a 5,000-item review budget, the MDE is approximately
±0.3pp — sufficient to detect a 30% relative change in prevalence.

**3. Proxy Metrics (Flag Rate as Leading Indicator)**
Shows how flag rate (raw detection system output) can serve as a proxy for prevalence
in early weeks of an experiment, before sufficient gold labels are available. Quantifies
the bias: flag rate = TPR·π + FPR·(1−π). Derives the correction formula and the
additional uncertainty it introduces. Recommends using flag rate only for directional
signals, not for final prevalence comparison.

**4. Sensitivity Heatmap (MDE by Baseline × Corpus Size)**
Heatmap where:
- Rows: baseline prevalence (0.1%, 0.5%, 1%, 2%, 5%, 10%)
- Columns: corpus size (100K, 500K, 1M, 5M, 10M)
- Cell: MDE in percentage points (for fixed review budget = 5,000)

Highlights that low-prevalence verticals require either very large corpus sizes or larger
review budgets to achieve meaningful MDEs.

**5. Worked Example: Threshold Change Experiment**
Full end-to-end worked example for a detection threshold change experiment:
- Treatment: lower classifier threshold from 0.5 to 0.4
- Expected effect: +15% relative increase in detected violations (true positives)
- Calculation of required corpus exposure, review budget, and experiment duration
- Pre-registration checklist (primary metric, stopping rule, multiple comparisons plan)

**6. O'Brien–Fleming Sequential Testing**
Implements O'Brien–Fleming alpha spending for interim analyses. Allows stopping early if
the effect is large enough to cross the adjusted boundary, or for futility if the observed
effect is below the futility threshold. Shows how to maintain overall α = 0.05 with two
interim looks and one final analysis.

**Key outputs:** MDE curves, sensitivity heatmap, worked example summary table.

---

## 08 — Off-Platform Signals (`08_off_platform_signals.ipynb`)

**Purpose:** Integrate external evidence (law enforcement reports, CSAM hash databases,
GIFCT signals, researcher alerts) with on-platform classifier outputs using Bayesian
updating to produce sharper prevalence estimates.

**Prerequisites:** None. All signals are synthetic.

### Sections

**1. Bayesian Update Framework**
Introduces the Bayesian prevalence update formula:
```
P(harmful | signal) = P(signal | harmful) · P(harmful) /
                      [P(signal | harmful) · P(harmful) + P(signal | benign) · P(benign)]
```
Implements a single-signal update function and shows how a strong external signal (e.g.,
law enforcement tip with 90% signal-to-noise) can shift the prevalence prior from 1% to
8% after a positive match.

**2. FPR Sensitivity Analysis**
Plots the posterior prevalence as a function of the signal's false positive rate, for three
prior prevalence levels. Shows that signal FPR is critical: a signal with FPR = 20%
provides almost no update at low prevalence because the base rate of false positives
dominates.

**3. Sequential Multi-Signal Updating**
Simulates sequential updates from four independent external signals:
1. CSAM hash match (high precision, low recall)
2. GIFCT URL match (medium precision)
3. Researcher alert (moderate precision)
4. Cross-platform account linkage (moderate precision)

Updates the prevalence posterior at each step. Shows how independent signals compound:
three moderate-quality signals can together shift the posterior as much as one
high-quality signal.

**4. Decision Threshold Analysis**
Maps the posterior prevalence to operational decisions:
- Below 0.5%: monitor only, no additional sampling
- 0.5%–2%: increase sampling rate for this vertical
- 2%–5%: trigger enhanced review queue
- Above 5%: escalate to specialized enforcement team

Shows how the decision threshold shifts when incorporating external signals vs
on-platform data alone.

**5. Corpus-Scale Signal Integration**
Scales the Bayesian update to the full corpus: for each item, if an external signal
exists, the posterior prevalence is updated before the stratified sampling decision.
This effectively increases the sensitivity of the pre-filter for items with external
evidence, concentrating the review budget on the highest-risk items.

**6. Key Findings**
Table summarizing posterior shifts by signal type and quality, and the implied sample
size reduction (how much smaller the review sample needs to be when external signals
are available).

**Key outputs:** Signal update plots, posterior shift table, decision boundary chart.

---

## 09 — Metrics Funnel (`09_metrics_funnel.ipynb`)

**Purpose:** Comprehensive reference for all 25 integrity measurement metrics, organized
by the 7-stage measurement funnel. Includes definitions, formulas, interpretation pitfalls,
trend monitoring, and a metrics decision guide.

**Prerequisites:** None. All data is generated by `src/simulation.py`.

### Sections

**1. The 7-Stage Integrity Measurement Funnel**
Horizontal funnel chart showing the 7 stages from raw corpus to final enforcement:

```
Corpus → Pre-filter → LLM Classifier → Stratified Sample →
Human Review → Enforcement → Appeals
```

Each stage is annotated with typical volume ratios (e.g., 10M items in corpus →
500K flagged by pre-filter → 5,000 in stratified sample → 4,100 confirmed harmful →
3,200 actioned). Simulates a realistic platform and renders the funnel with stage-over-
stage retention percentages.

**2. Pre-Filter Metrics**
Computes and explains:
- **Pre-filter rate:** Fraction of corpus flagged by the cheap pre-filter signal
- **Pre-filter precision:** Among flagged items, true positive rate
- **Pre-filter recall:** Among all harmful items, fraction caught by pre-filter

Shows the precision-recall tradeoff: a lenient pre-filter (high recall) inflates
downstream review cost; a strict pre-filter (high precision) risks missing harmful items
in the un-sampled majority.

**3. Classifier Metrics + Base-Rate Problem**
Full classifier evaluation: TPR, FPR, precision, recall, F1, AUC-ROC at multiple
thresholds. Includes a precision-recall curve and an explicit base-rate demonstration:
- At 1% prevalence and 5% FPR, positive predictive value = only 14%
- Even a perfect classifier (FPR=0) cannot raise PPV above the true prevalence

Threshold sweep table showing how each metric varies with the decision threshold.
Recommends the operating point that minimizes weighted misclassification cost.

**4. Prevalence Estimator Comparison**
Side-by-side comparison of all three estimators on the same synthetic dataset:
- Direct proportion (Wilson CI)
- Classifier-adjusted (delta method CI)
- Chapman capture-recapture

Includes CI width comparison and a color-coded validity table showing which estimator to
use under which conditions (gold labels available, FPR known, two independent systems
available, etc.).

**5. Enforcement Metrics**
Computes:
- **Gap rate:** Fraction of detected violations not actioned within SLA
- **High-confidence gap rate:** Gap rate restricted to detections with score ≥ 0.8
- **Enforcement rate:** Fraction actioned among all confirmed violations
- **Final enforcement rate (FER):** Fraction actioned that survived appeal

Bar charts by harm vertical and by risk tier.

**6. Weekly Scorecard**
Generates a 52-week simulated dataset and renders a compact weekly scorecard table
(last 8 weeks shown) with sparkline-style trend indicators for each KPI:
- Prevalence estimate and CI
- Detection rate
- Enforcement rate
- Gap rate
- Anomaly flag (Z-score > 2)

**7. Z-Score + CUSUM Trend Monitor**
Side-by-side view of:
- Rolling Z-score with ±2σ threshold bands
- CUSUM upper and lower control statistics with alarm boundaries

Demonstrates a simulated bad actor campaign (weeks 28–34) and marks both Z-score and
CUSUM alarm weeks. CUSUM fires 1–2 weeks earlier in the ramp-up phase; Z-score is more
responsive to sudden spikes.

**8. Full Metrics Glossary**
25-row reference table with columns:
- **Metric:** Short name used in code and dashboards
- **Definition:** Plain-language description of what it measures
- **Stage:** Which funnel stage it belongs to
- **Formula:** Mathematical definition
- **Units:** Percentage, count, or ratio
- **Interpretation Pitfall:** The most common way this metric is misread in practice

Key pitfalls documented: naive flag rate vs prevalence, PPV sensitivity to base rate,
AUC masking class imbalance, enforcement rate excluding appeals, coverage rate vs recall.

**9. Metrics Decision Guide**
Flowchart-style decision guide (rendered as a structured markdown table) for choosing
which estimator and which monitoring approach to use:
- Gold labels available? → Direct proportion
- TPR/FPR known, no gold labels? → Classifier-adjusted
- Two independent high-precision detection systems? → Capture-recapture
- Need early warning? → CUSUM
- Need burst detection? → Z-score

**Key outputs:** Funnel chart, metrics glossary table, trend monitor plots.

---

## Running Without an API Key

All notebooks that call the LLM classifier (primarily notebook 01) detect the absence of
an API key and fall back to `src/simulation.py` synthetic outputs. The synthetic outputs
are parameterized to match typical real-model performance (TPR ≈ 0.82, FPR ≈ 0.06) and
produce equivalent visualizations for all downstream notebooks.

## Saving Outputs

Plots are saved to `data/processed/` as PNG files during notebook execution. The
`data/processed/` directory is created automatically if it does not exist. All plot files
are gitignored by default; run the notebooks locally to regenerate them.

## Environment Variables

| Variable | Required for | Default behavior if missing |
|----------|-------------|----------------------------|
| `ANTHROPIC_API_KEY` | Notebook 01 (Claude) | Synthetic data fallback |
| `OPENAI_API_KEY` | Notebook 01 (GPT) | Synthetic data fallback |
| `GROQ_API_KEY` | Notebook 01 (Groq) | Synthetic data fallback |

Set `PROVIDER = 'anthropic'` (or `'openai'` or `'groq'`) in notebook 01 to choose which
API to use.
