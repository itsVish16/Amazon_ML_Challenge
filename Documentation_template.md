# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** The Backpropagators  
**Team Members:** Vishal Saini & Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed a highly scalable, two-stage Business Entity Resolution framework designed to resolve noisy entity records across multi-source datasets spanning the US, India, and France. Our approach couples a multi-key inverted index micro-blocking mechanism with a calibrated LightGBM classifier specifically tuned to maximize macro-averaged $F_{0.5}$ score. By integrating multi-script Indic transliteration, address number canonicalization, and strict country-level partitioning, the pipeline achieves an ultra-compact candidate pool (~15 candidates per Source 1 entity) with >97.4% recall and a validation macro $F_{0.5}$ score of 0.9905.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the ground truth labels and multi-source data revealed distinct structural challenges:
- **Language and Transliteration**: Records from India frequently feature business names transliterated into Devanagari, Tamil, Bengali, or Marathi scripts, while reference entities exist in Roman script. Addresses often substitute local language state names (e.g., `दिल्ली` for Delhi, `हरियाणा` for Haryana).
- **Address Formatting Noise**: Addresses contain token transpositions, unit variations (e.g., `Unit UNIT 367` vs `Unit 367`), leading-zero padded numbers (`022899` vs `22899`), and punctuation noise (`##`, `--`, `<<`). Approximately 4% of records lack an address entirely, requiring robust name-based fallback matching.
- **Entity Alias Corruptions**: Synthetic variations introduce domain names (`siiainvestments.com`), DBA tags (`Miradova dba ...`), and former names (`formerly Painters Local Union 634`).
- **Country Invariance**: Matches never cross international boundaries; records in France, the US, and India strictly resolve within their respective country sets.

### 2.2 Solution Strategy
**Approach Type:** Multi-Key Inverted Index Blocking + Calibrated GBDT Matching Classifier  
**Core Innovation:** A dual-anchor indexing scheme that decouples address number/street tokens from phonetic name representations, paired with an $F_{0.5}$-calibrated decision threshold ($T = 0.70$) that penalizes false merges $2\times$ heavier than false negatives, safeguarding singletons and multi-match precision.

---

## 3. Candidate Generation (Blocking)

To comply with the challenge's strict blocking efficiency evaluation ("the approach that generates a smaller candidate set per Source 1 entity will be ranked higher"), we avoided broad n-gram Cartesian explosions and deployed targeted inverted indices:

- **Country Partitioning**: Hard partition by country label (open-set: supports US, India, and test-unseen France identically).
- **Address Channel Keys**:
  - `(country, house_number, street_token)`
  - `(country, house_number, state_code)`
  - `(country, street_token_1, street_token_2)`
- **Name Channel Keys**:
  - `(country, primary_token)`
  - `(country, name_token_1, name_token_2)`
  - `(country, domain_base)`
  - `(country, compact_name_prefix)`
- **Recall & Reduction Ratio**:
  - **Ground Truth Recall:** 97.43%
  - **Average Candidates per S1 Entity:** 15.48 (Median: 16.0)
  - **Hub Key Suppression:** Keys with frequency $> 1000$ are automatically truncated to prevent candidate bloat.

---

## 4. Matching Model

**Feature Engineering (17 Pairwise Features):**
- **Name Similarity**:
  - RapidFuzz Levenshtein Ratio, Token Sort Ratio, Token Set Ratio, Partial Ratio
  - Token Jaccard Similarity
  - Domain base match indicator (e.g., `siiainvestments` in `Siia Investments Inc`)
- **Address Similarity**:
  - Full Address Levenshtein Ratio, Token Sort Ratio, Token Set Ratio, Token Jaccard
  - Street/House Number Exact Agreement & Jaccard overlap
  - State Code Canonical Match (+1 for match, -1 for conflict, 0 for missing)
- **Structural Indicators**:
  - S2 vs S3 source indicator
  - Missing address flag (S1 or candidate)
  - Composite maximum similarity metric

**Model Type:** LightGBM Binary Classifier (GBDT)
- Objective: `binary`, metric: `binary_logloss`
- Hyperparameters: `num_leaves: 31`, `learning_rate: 0.05`, `max_depth: 6`, `feature_fraction: 0.8`
- Top features by Gain: `addr_jaccard`, `addr_set_ratio`, `max_sim`, `addr_sort_ratio`, `name_sort_ratio`

**Threshold Selection Method:**
Threshold grid search targeting macro-averaged $F_{0.5}$ directly on a holdout validation set:
- Optimal threshold identified at $T = 0.70$ (Macro $F_{0.5} = 0.9905$).
- High decision threshold filters out ambiguous candidates, suppressing false merges on both singletons and matched entities.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **0.9905**
- **Singletons Handled:** ~5.6% true singletons accurately predicted as empty match lists ($F_{0.5} = 1.0$).
- **Common False Positives (Wrong Merges):** Co-located businesses at identical commercial complexes/malls sharing identical street numbers with minimal name distinctiveness.
- **Common False Negatives (Missed Matches):** Extreme joint corruption where both address is completely omitted and name underwent severe multi-character transliteration mutations.

---

## 6. Conclusion
The two-stage framework delivers state-of-the-art accuracy by combining linguistic transliteration with targeted structural blocking and high-precision gradient boosting. Achieving a 0.9905 validation $F_{0.5}$ score while restricting the candidate set to an average of ~15 records per entity satisfies both competitive leaderboard accuracy and the organizer's blocking scalability mandate.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/normalization.py`: Universal text & address normalization.
- `code/business_entity_resolution/src/blocking.py`: Multi-key inverted index blocking engine.
- `code/business_entity_resolution/src/features.py`: 17-dimensional vector extraction using rapidfuzz.
- `code/business_entity_resolution/src/train.py`: Model training and $F_{0.5}$ grid threshold optimization.
- `code/business_entity_resolution/src/pipeline.py`: Country-partitioned streaming test set inference.
