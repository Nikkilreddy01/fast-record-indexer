#!/usr/bin/env bash
# ==============================================================================
# Amazon ML Challenge 2026 - Master One-Click Runner (Auto Multi-Core)
# ==============================================================================
# This script automatically:
# 1. Detects all available CPU cores on the machine
# 2. Sets up Python virtual environment and installs dependencies
# 3. Verifies or extracts the dataset
# 4. Executes the full high-recall entity resolution pipeline
# 5. Validates the generated submission files (exit 0)
# 6. Packages Odin_submission.zip ready for upload
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================================"
echo "   AMAZON ML CHALLENGE 2026: ONE-CLICK END-TO-END PIPELINE RUNNER"
echo "============================================================================"

# 1. Auto-detect CPU cores and Operating System
OS_TYPE="$(uname -s)"
if [ "$OS_TYPE" = "Darwin" ]; then
    NUM_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
elif [ "$OS_TYPE" = "Linux" ]; then
    NUM_CORES=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)
else
    NUM_CORES=4
fi

echo "[System] Detected OS: $OS_TYPE"
echo "[System] Detected CPU Cores: $NUM_CORES (Auto-scaling batch size and parallelism)"

# Dynamically scale batch size with core count
if [ "$NUM_CORES" -ge 32 ]; then
    BATCH_SIZE=100000
elif [ "$NUM_CORES" -ge 16 ]; then
    BATCH_SIZE=50000
else
    BATCH_SIZE=25000
fi
echo "[System] Configured Streaming Batch Size: $BATCH_SIZE entities/batch"

# 2. Python Environment Setup
if [ ! -d ".venv" ]; then
    echo -e "\n[Step 1/5] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[Step 1/5] Activating virtual environment & installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r code/business_entity_resolution/requirements.txt --quiet
echo "  Dependencies verified."

# 3. Dataset Verification / Extraction
echo -e "\n[Step 2/5] Checking dataset files..."
if [ ! -d "dataset/test" ]; then
    if [ -d "student_resource/dataset" ]; then
        echo "  Linking dataset from student_resource/dataset..."
        ln -sfn "$PWD/student_resource/dataset" dataset
    elif [ -f "6ab10eb3b23ba_student_resource.zip" ]; then
        echo "  Extracting 6ab10eb3b23ba_student_resource.zip..."
        unzip -q 6ab10eb3b23ba_student_resource.zip -d student_resource
        ln -sfn "$PWD/student_resource/dataset" dataset
    else
        echo "ERROR: dataset/test not found! Please place the dataset/ folder or student_resource.zip in $PWD"
        exit 1
    fi
fi

if [ -f "dataset/test/test_source1.tsv" ]; then
    echo "  Dataset verified: $(wc -l < dataset/test/test_source1.tsv | tr -d ' ') test entities found."
else
    echo "ERROR: dataset/test/test_source1.tsv does not exist!"
    exit 1
fi

# 4. Run Inference Pipeline
echo -e "\n[Step 3/5] Launching High-Recall Entity Resolution Pipeline..."
echo "  Using calibrated model: entity_model.joblib (0.9832 Validation Score)"

# Prevent Mac sleep if on macOS
if [ "$OS_TYPE" = "Darwin" ]; then
    if command -v caffeinate >/dev/null 2>&1; then
        caffeinate -dimsu -w $$ &
        CAFF_PID=$!
        echo "  Activated macOS caffeinate (PID $CAFF_PID) to prevent system sleep."
    fi
fi

python3 code/business_entity_resolution/src/pipeline.py \
    --model-path entity_model.joblib \
    --batch-size "$BATCH_SIZE" \
    --max-candidates 40

# 5. Format & Submission Validation
echo -e "\n[Step 4/5] Running official submission validator..."
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 6. Package Submission ZIP
echo -e "\n[Step 5/5] Packaging final submission archive..."
python3 make_submission_zip.py --team-name Odin

echo ""
echo "============================================================================"
echo "🎉 SUCCESS: ALL STEPS COMPLETED!"
echo "============================================================================"
echo "1. Live Leaderboard File: output/matching_results.tsv (Upload this to Unstop!)"
echo "2. Final Package:         Odin_submission.zip"
echo "============================================================================"
