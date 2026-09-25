# Fast Record Indexer & Business Entity Resolution

**Team Odin** | Amazon ML Challenge 2026  
High-Recall, Low-Latency Cross-Feed Entity Resolution Pipeline across France, United States, and India.

---

## 🚀 Key Features

* **High-Recall Multi-Index Blocker**: Rapid candidate pruning combining token inverted indexing, exact normalization, and phonetics.
* **Calibrated Decision Threshold ($\tau = 0.660$)**: Optimizes Macro $F_{0.5}$ metric (penalizing False Positives 2× heavier than False Negatives).
* **Streaming Country Inference**: Low memory footprint streaming 1.73M test Source 1 entities across 12M+ candidate records without OOM.
* **Universal 1-Click Execution**: Standalone cross-platform runner (`run.py` / `run_all.bat`) with auto-scaling to available CPU cores and auto-download of dataset assets.
* **Live Progress & ETA Dashboard**: Real-time terminal dashboard (`monitor.py`) tracking processed entities, throughput, and completion countdown.

---

## 📂 Repository Structure

```
├── run.py                              # Master cross-platform Python runner (Windows / Mac / Linux)
├── run_all.bat                         # Windows 1-click batch runner (auto-detects cores & Python)
├── run_all.ps1                         # Native Windows PowerShell runner
├── run_all.sh                          # Linux / macOS bash runner
├── monitor.py                          # Live progress & ETA tracking dashboard
├── monitor.bat                         # Windows 1-click monitor launcher
├── monitor.sh                          # Linux / macOS 1-click monitor launcher
├── RESEARCH_INNOVATION_BUCKET_LIST.md  # 0.990+ Academic foundations (PVLDB, VLDB, Amazon Science)
├── entity_model.joblib                 # Calibrated LightGBM model (1.2 MB)
├── guidelines_and_key_instructions.pdf # Hackathon submission rules & format
├── 6ab5628d5a817_...statement.pdf      # Official problem statement
├── 6ab509c5b7036_...video.mp4          # Hackathon walkthrough video
├── make_submission_zip.py              # Packaging script for Unstop submission
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt            # Python dependencies
│       └── src/
│           ├── blocker.py              # Inverted index candidate generation
│           ├── features.py             # Jaro-Winkler, Levenshtein, token set ratio features
│           ├── model.py                # LightGBM pairwise match classifier & threshold tuning
│           └── pipeline.py             # Country-partitioned streaming batch inference
└── utils/
    └── validate_submission.py          # Strict validator matching official Unstop criteria
```

---

## ⚡ Quick Start

### Windows
1. Open PowerShell or Command Prompt in the repository folder:
   ```cmd
   python run.py
   ```
   *(Or double-click `run_all.bat`)*
2. To watch live progress and countdown ETA:
   ```cmd
   python monitor.py
   ```
   *(Or double-click `monitor.bat`)*

### Linux / macOS
```bash
# Run pipeline:
python3 run.py

# Live monitoring:
./monitor.sh
```

---

## 📊 Dataset Asset
The 1.01 GB competition dataset archive (`6ab10eb3b23ba_student_resource.zip`) is hosted on the [v1.0.0 Release](https://github.com/Nikkilreddy01/fast-record-indexer/releases/tag/v1.0.0).  
`run.py` automatically downloads and extracts it if not already present locally.
