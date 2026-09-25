# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An end-to-end Machine Learning pipeline for multi-source Business Entity Resolution, evaluated on Macro $F_{0.5}$ score.

---

## 1. Overview & Architecture

The pipeline resolves records from three independent data sources (`Source 1`, `Source 2`, `Source 3`):
- **Source 1**: The deduplicated reference source.
- **Source 2 & 3**: Uncurated data feeds containing noisy name and address variations.
- **Target**: Find all matching Source 2 and Source 3 records for each Source 1 entity (including singletons with 0 matches).

```
Raw Data (S1, S2, S3)
         │
         ▼
[Preprocessing & Normalization]
  - Unicode/Accent stripping (multilingual, France support)
  - Legal entity normalization (Pvt Ltd, LLC, Inc, Corp, SARL)
  - Address abbreviation standardizer & numeric extraction
         │
         ▼
[High-Recall Multi-Index Blocking]
  - Country-partitioned hashing
  - Significant first-word & prefix index
  - TF-IDF character n-gram cosine retrieval
  - Address numeric token index
  (Yields candidate_pairs.tsv with >95% recall ceiling)
         │
         ▼
[Fine-Grained Feature Extraction]
  - 26 distance, phonetic, token set, and cross-interaction features
         │
         ▼
[LightGBM Precision-Tuned Classifier]
  - Class-imbalance aware gradient boosting
  - Macro F_0.5 threshold optimization (2x precision weighting)
  - Singleton confidence filter
         │
         ▼
[Validated Outputs]
  - output/matching_results.tsv  (Scored on Leaderboard)
  - output/candidate_pairs.tsv   (Audit candidate set)
```

---

## 2. Directory Structure

```
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── feature_engineering.py
│       │   ├── model.py
│       │   └── pipeline.py
│       ├── requirements.txt
│       └── README.md
├── utils/
│   └── validate_submission.py
└── Documentation_template.md
```

---

## 3. Quick Start & Reproduction

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2: Run End-to-End Pipeline
```bash
python3 code/business_entity_resolution/src/pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output
```

### Step 3: Validate Outputs
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
You should see:
```
PASS: Submission files are fully compliant and ready for upload!
```

---

## 4. Key Design Decisions

1. **Why Macro $F_{0.5}$ Optimization?**
   - The competition penalizes false merges (false positives) twice as heavily as missed matches ($0.25 \times \text{Precision} + \text{Recall}$).
   - The pipeline uses a fine-grained grid search on the held-out validation set to find the threshold $\tau^*$ that explicitly maximizes Macro $F_{0.5}$ instead of standard accuracy or ROC-AUC.
2. **Handling Singletons**:
   - Entities in Source 1 with no matches in Source 2 or 3 award 1.0 points when predicted as empty, but 0.0 if any false match is predicted. The calibrated high threshold prevents reckless merges.
3. **Open Country Generalization**:
   - The preprocessor strips unicode diacritics and handles international legal suffixes (such as French `SARL`, `SAS`, `SA`), ensuring seamless inference on unseen countries like France without hardcoding.
