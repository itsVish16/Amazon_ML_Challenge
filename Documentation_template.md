# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** The Backpropagators  
**Team Members:** Vishal Saini & Team  
**Submission Date:** October 2026  

---

## 1. Executive Summary
We developed a highly scalable, two-stage Business Entity Resolution framework designed to resolve noisy entity records across multi-source datasets spanning the US, India, and France. Our approach couples a multi-key inverted index micro-blocking mechanism with a calibrated LightGBM gradient boosted classifier specifically tuned to maximize macro-averaged $F_{0.5}$ score. By integrating multi-script Indic transliteration, corporate legal suffix separation, administrative region canonicalization (US states, Indian states/territories, and French regions/departments), and full-target hard-negative training against 10.3M records, the pipeline achieves an ultra-compact candidate pool (~20 candidates per Source 1 entity) with >97.4% recall and a production-calibrated macro $F_{0.5}$ score exceeding 0.909 on the full target universe.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the ground truth labels and multi-source data revealed distinct structural challenges:
- **Language and Transliteration**: Records from India frequently feature business names transliterated into Devanagari, Tamil, Bengali, or Marathi scripts, while reference entities exist in Roman script. Addresses often substitute local language state names (e.g., `दिल्ली` for Delhi, `हरियाणा` for Haryana).
- **Address Formatting Noise**: Addresses contain token transpositions, unit variations (e.g., `Unit UNIT 367` vs `Unit 367`), leading-zero padded numbers (`022899` vs `22899`), and punctuation noise (`##`, `--`, `<<`). Approximately 4% of records lack an address entirely, requiring robust name-based fallback matching.
- **Entity Alias Corruptions**: Synthetic variations introduce domain names (`siiainvestments.com`), DBA tags (`Miradova dba ...`), and former names (`formerly Painters Local Union 634`).
- **Country and Regional Invariance**: Matches never cross international boundaries; records in France, the US, and India strictly resolve within their respective country sets. Furthermore, businesses in different physical administrative regions (e.g. Dunkerque in Hauts-de-France vs Bordeaux in Nouvelle-Aquitaine) cannot be the same entity.

### 2.2 Solution Strategy
**Approach Type:** Multi-Key Inverted Index Blocking + Calibrated GBDT Matching Classifier + Ranked Confidence-Aware Decision Engine  
**Core Innovation:** A dual-anchor indexing scheme that decouples address number/street tokens from phonetic name representations, paired with full-target hard-negative training against the complete 10.3M record universe. Inference uses a ranked, per-entity confidence-gap decision engine that eliminates chain-store explosion and penalizes false merges $2\times$ heavier than false negatives in compliance with the $F_{0.5}$ metric.

---

## 3. Candidate Generation (Blocking)

To comply with the challenge's strict blocking efficiency evaluation ("the approach that generates a smaller candidate set per Source 1 entity will be ranked higher"), we avoided broad n-gram Cartesian explosions and deployed targeted inverted indices:

- **Country Partitioning**: Hard partition by country label (US, India, and France processed independently with zero cross-country leakage).
- **Address Channel Keys**:
  - `(country, house_number, street_token)`
  - `(country, house_number, canonical_state)`
  - `(country, street_token_1, street_token_2)`
- **Name Channel Keys**:
  - `(country, primary_name_token)`
  - `(country, name_token_1, name_token_2)`
  - `(country, domain_base)`
  - `(country, compact_name_prefix)`
- **Recall & Reduction Ratio**:
  - **Ground Truth Recall:** 97.43%
  - **Average Candidates per S1 Entity:** 15–20 candidates
  - **Hub Key Suppression:** Keys with excessive frequency ($> 2,500$) are automatically down-weighted using $1 / (1 + \text{freq})$ evidence accumulation to prevent candidate bloat.

---

## 4. Matching Model

**Feature Engineering (24 Pairwise Discriminative Features):**
- **Name Similarity (10 Features)**:
  - RapidFuzz Levenshtein Ratio, Token Sort Ratio, Token Set Ratio, Partial Ratio
  - Token Jaccard Similarity & Character 3-Gram Jaccard (handles transliteration variance)
  - Exact Name Match indicator
  - First Meaningful Token Match (catches prefix divergence such as "Novak Quinn" vs "Barbara Quinn")
  - Name Token Coverage (fraction of Source 1 tokens present in candidate)
  - Name Length Ratio (penalizes over-stripped suffix mismatches)
- **Domain Matching (1 Feature)**:
  - Domain base match indicator (e.g., `siiainvestments` in `Siia Investments Inc`)
- **Address Similarity (4 Features)**:
  - Full Address Levenshtein Ratio, Token Sort Ratio
  - Missing address indicators (Source 1 or Candidate)
- **House Number Agreement & Conflict (3 Features)**:
  - Number Jaccard overlap, Primary House Number exact match, Disjoint Number conflict flag
- **State & Regional Agreement (2 Features)**:
  - Canonical state/region match (+1.0 for match, -1.0 for conflict, 0.0 for missing)
  - Canonical state/region conflict indicator (with historical Telangana/Andhra Pradesh equivalence)
- **Street Agreement (2 Features)**:
  - Street Token Similarity (Jaccard + fuzzy best-pair ratio)
  - Disjoint Street Conflict binary flag
- **Structural Indicators (2 Features)**:
  - S2 vs S3 source indicator
  - Composite maximum similarity metric

**Model Type:** LightGBM Binary Classifier (GBDT)
- Objective: `binary`, metric: `binary_logloss`, `boosting_type: gbdt`
- Hyperparameters: `num_leaves: 127`, `max_depth: 10`, `learning_rate: 0.03`, `num_boost_round: 800`
- Regularization & Sampling: `feature_fraction: 0.85`, `bagging_fraction: 0.80`, `bagging_freq: 1`, `min_data_in_leaf: 30`
- Class Imbalance Handling: `scale_pos_weight` calibrated against hard-negative ratio
- Top features by Information Gain: `max_sim`, `name_partial_ratio`, `nums_overlap`, `num_conflict`, `name_ratio`, `name_sort_ratio`, `addr_sort_ratio`, `street_conflict`, `first_token_match`, `name_char_jaccard`

**Threshold Selection & Decision Logic:**
- Decision threshold grid search targeting macro-averaged $F_{0.5}$ directly against the complete 10.3M target universe:
  - Optimal calibrated threshold: $T = 0.89$
- **Ranked Confidence-Aware Matching (`select_matches`)**:
  - Scores all candidates for an entity simultaneously, sorting by probability descending.
  - Tier 1: High confidence ($P \ge 0.89, \text{name\_sim} \ge 0.50$)
  - Tier 2: Standard confidence ($P \ge T, \text{name\_sim} \ge 0.55$)
  - Tier 3: Domain bridge ($P \ge T \times 0.90, \text{name\_sim} \ge 0.60$)
  - Hard Vetoes: Impossibility veto on administrative state/region conflict, and address number conflict without domain rescue.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Full Target Universe Validation):** **0.9094** at threshold $0.89$
- **Blocking Recall:** 97.43% on holdout validation
- **Singletons Handled:** ~5.6% true singletons accurately predicted with empty match lists ($F_{0.5} = 1.0$)
- **Common False Positives (Wrong Merges):** Co-located businesses in identical commercial plazas/malls sharing identical street numbers with minimal name distinctiveness. Mitigated by `first_token_match` and `name_token_coverage`.
- **Common False Negatives (Missed Matches):** Extreme joint corruption where both address is completely omitted and name underwent severe multi-character transliteration mutations across scripts.

---

## 6. Conclusion
The two-stage framework delivers state-of-the-art accuracy by combining linguistic transliteration with targeted structural blocking, 24 discriminative pairwise features, and high-precision gradient boosting trained against real hard negatives. Achieving >0.909 macro $F_{0.5}$ on the full 10.3M record universe while maintaining compact candidate sets satisfies both competitive leaderboard accuracy and the organizer's blocking scalability mandate.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/normalization.py`: Universal text, Indic transliteration, corporate legal suffix, and US/India/France regional canonicalization.
- `code/business_entity_resolution/src/blocking.py`: Multi-key inverted index blocking engine with evidence accumulation.
- `code/business_entity_resolution/src/features.py`: 24-dimensional discriminative pairwise feature extraction using RapidFuzz.
- `code/business_entity_resolution/src/decision.py`: Ranked confidence-aware matching engine with physical impossibility vetoes.
- `code/business_entity_resolution/src/train.py`: Full-target hard-negative model training and $F_{0.5}$ grid threshold optimization.
- `code/business_entity_resolution/src/pipeline.py`: Country-partitioned multi-process parallel inference streaming output TSVs.
- `code/business_entity_resolution/run_aws.sh`: Automated multi-core runner with core auto-detection.
- `code/business_entity_resolution/train_full.sh`: Automated full-target training script.
