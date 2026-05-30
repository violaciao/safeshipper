# Dataset Provenance

## BeaverTails (Primary)

- **Source:** HuggingFace — `PKU-Alignment/BeaverTails`
- **License:** CC BY NC 4.0
- **Size:** ~333,963 QA pairs
- **Schema:** `prompt`, `response`, `is_safe` (bool), 14 harm category binary flags
- **Harm categories:** `hate_speech`, `violence`, `self_harm`, `illegal_activity`, `animal_abuse`, `child_abuse`, `controversial_topics`, `discrimination_stereotype_injustice`, `financial_crime`, `non_violent_unethical_behavior`, `privacy_violation`, `sexually_explicit`, `terrorism_organized_crime`, `weapons_of_mass_destruction`
- **Load:** `from datasets import load_dataset; ds = load_dataset("PKU-Alignment/BeaverTails")`

## RealToxicityPrompts (Secondary)

- **Source:** HuggingFace — `allenai/real-toxicity-prompts`
- **License:** Apache 2.0
- **Size:** ~100,000 prompts
- **Schema:** `prompt` dict (with `text`, `toxicity`), `continuation` dict (with `text`, `toxicity`, `severe_toxicity`, etc.)
- **Use:** Continuous-score calibration analysis; threshold sensitivity experiments
- **Load:** `from datasets import load_dataset; ds = load_dataset("allenai/real-toxicity-prompts")`

## Synthetic Data

Generated via `src/simulation.py`. No external license — fully controlled experiments with known ground truth. Use this for:
- Validating estimator coverage
- Demonstrating methodology when real data is unavailable
- Parameterizing edge cases (zero prevalence, near-100% prevalence)

## File Organization

```
data/
├── raw/         # Downloaded datasets (large files gitignored)
├── processed/   # Cleaned, stratified subsets (small, versioned)
└── README.md    # This file
```

Processed files follow the naming convention:
`{dataset}_{harm_category}_{n_samples}_{date}.parquet`

e.g., `beavertails_violence_500_20240601.parquet`
