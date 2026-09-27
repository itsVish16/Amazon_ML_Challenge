# Business Entity Resolution Pipeline

This repository contains the end-to-end machine learning solution for the **Amazon ML Challenge: Business Entity Resolution**.

## Architecture & Methodology

The solution employs a two-stage architecture designed for massive scalability, high precision, and compliance with the $F_{0.5}$ metric:
1. **Normalization & Canonicalization Engine**:
   - Universal accent stripping (`NFKD`) and multi-script transliteration (`anyascii`) for Indic scripts (Hindi, Tamil, Marathi, Bengali, Telugu, etc.).
   - Standardized legal entity suffix removal (`inc`, `llc`, `pvt ltd`, `sarl`, `sasu`, etc.).
   - Business DBA and "formerly" parsing to recover underlying legal entity names.
   - Address cleaning: street number zero-unpadding, ordinal conversion, street type mapping (`st` -> `street`, `rd` -> `road`), and regional state canonicalization for US, India, and France.
2. **Multi-Key Inverted Index Micro-Blocking**:
   - Dynamic country-level partitioning (100% strict cross-country blocking with zero cross-country false matches).
   - Inverted indexing on Address Number + Street tokens, Street pairs, and Name token pairs.
   - Achieves >97.4% recall ceiling while maintaining an ultra-compact candidate pool (~15 candidates per S1 entity), maximizing the blocking efficiency score reviewed for final rankings.
3. **High-Precision Matching Model**:
   - LightGBM binary classifier trained on hard blocked pairs with 17 rapid string-similarity, token-overlap, address-number agreement, and domain match features.
   - Calibrated decision threshold ($T = 0.70$) optimized specifically for macro-averaged $F_{0.5}$, minimizing false merges while correctly preserving singletons.

## Environment & Requirements

- Python 3.8+
- Hardware recommendation: 8+ CPU cores, >= 8 GB RAM

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### AWS / high-memory instance

The full test set is too large for a typical laptop. Use an instance with at
least **16 vCPUs and 64 GB RAM**; 128 GB RAM provides more headroom for the
US candidate index. The public repository intentionally excludes challenge
data, generated outputs, and submission archives.

1. Clone this repository and create an isolated environment:

```bash
git clone https://github.com/itsVish16/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

2. Transfer the competition-provided `DATA/student_resource/dataset/` directory
to the instance through a private channel (for example, `rsync` over SSH or
your own private S3 bucket). Do not put the challenge data on GitHub.

3. Run inference with unbuffered logs. The AWS runner defaults to 10,000 S1
records per batch; reduce `BATCH_SIZE` if memory is constrained.

```bash
chmod +x code/business_entity_resolution/run_aws.sh
nohup env BATCH_SIZE=10000 MAX_CANDIDATES=25 \
  code/business_entity_resolution/run_aws.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

After it finishes, validate before uploading `output/matching_results.tsv`:

```bash
python3 DATA/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir DATA/student_resource/dataset/test \
  --check-ids
```

`MAX_CANDIDATES=25` is the current high-recall baseline. Since candidate pool
size is reviewed, benchmark lower caps against held-out training blocking recall
before final submission; never lower it solely to make the output smaller.

### 1. Model Training (Optional, pre-trained weights included)
To re-train the LightGBM classifier and find the optimal $F_{0.5}$ threshold:
```bash
python3 src/train.py
```

### 2. End-to-End Test Set Inference
To generate both `matching_results.tsv` and `candidate_pairs.tsv` on the full test set:
```bash
python3 src/pipeline.py
```
This will automatically process the test set partitioned by country (France -> US -> India) and write the outputs to:
- `output/matching_results.tsv` (Leaderboard submission)
- `output/candidate_pairs.tsv` (Candidate set)

### 3. Submission Validation
Validate the generated output files against official competition rules:
```bash
python3 ../../DATA/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../../DATA/student_resource/dataset/test
```
