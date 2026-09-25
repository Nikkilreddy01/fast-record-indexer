#!/usr/bin/env python3
"""
Odin Master Runner — Universal Cross-Platform Execution Script
Works on Windows, Linux, and macOS without bash or external tools.
Usage:
    python run.py
    py -3 run.py
"""

import os
import sys
import shutil
import zipfile
import subprocess
import multiprocessing
import urllib.request

def banner(msg: str):
    print("\n" + "=" * 78)
    print(f"  {msg}")
    print("=" * 78)

def check_dependencies():
    """Verify and install dependencies only if missing."""
    banner("[Step 1/5] Checking Required Python Libraries...")
    missing = []
    
    for mod in ["pandas", "numpy", "sklearn", "lightgbm", "tqdm", "joblib"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if not missing:
        print("  ✅ All required libraries are already installed and ready.")
        return

    print(f"  Missing libraries detected: {', '.join(missing)}")
    print("  Installing missing packages...")
    req_file = os.path.join("code", "business_entity_resolution", "requirements.txt")
    
    install_cmds = [
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        [sys.executable, "-m", "pip", "install", "--user", "-r", req_file],
        [sys.executable, "-m", "pip", "install"] + missing,
        [sys.executable, "-m", "pip", "install", "--user"] + missing,
    ]

    success = False
    for cmd in install_cmds:
        try:
            print(f"  Trying: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            success = True
            break
        except Exception as e:
            print(f"  Attempt failed: {e}")

    if not success:
        print("\n[ERROR] Failed to install packages automatically.")
        print(f"Please manually run: pip install {' '.join(missing)}")
        sys.exit(1)

    print("  ✅ Libraries installed successfully.")

def ensure_dataset(root_dir: str):
    """Ensure dataset/test/test_source1.tsv is ready, downloading if missing."""
    banner("[Step 2/5] Verifying Dataset Files...")
    test_s1 = os.path.join(root_dir, "dataset", "test", "test_source1.tsv")
    
    if os.path.exists(test_s1):
        print(f"  ✅ Dataset verified: {test_s1}")
        return

    # Check student_resource folder
    alt_s1 = os.path.join(root_dir, "student_resource", "dataset", "test", "test_source1.tsv")
    if os.path.exists(alt_s1):
        print("  Found dataset in student_resource/dataset. Creating 'dataset' directory...")
        os.makedirs(os.path.join(root_dir, "dataset"), exist_ok=True)
        try:
            os.symlink(os.path.join(root_dir, "student_resource", "dataset"), os.path.join(root_dir, "dataset"))
        except Exception:
            shutil.copytree(os.path.join(root_dir, "student_resource", "dataset"), os.path.join(root_dir, "dataset"), dirs_exist_ok=True)
        return

    # Check for zip archive locally
    zip_candidates = [f for f in os.listdir(root_dir) if f.endswith(".zip") and ("student" in f or "dataset" in f or "resource" in f)]
    
    if not zip_candidates:
        zip_filename = "6ab10eb3b23ba_student_resource.zip"
        zip_dest = os.path.join(root_dir, zip_filename)
        url = f"https://github.com/Nikkilreddy01/fast-record-indexer/releases/download/v1.0.0/{zip_filename}"
        
        print(f"  Dataset not found locally. Downloading 1.01 GB dataset directly from GitHub Release...")
        print(f"  Source URL: {url}")
        
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, (downloaded / total_size) * 100.0)
                mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb:.1f} MB / {tot_mb:.1f} MB [{pct:.1f}%]")
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, zip_dest, reporthook=download_progress)
            print("\n  ✅ Download complete!")
            zip_candidates = [zip_filename]
        except Exception as dl_err:
            print(f"\n  [ERROR] Direct download failed: {dl_err}")
            print(f"  Please download the file manually in your browser:\n  {url}")
            print("  Save it in this repository folder and run python run.py again.")
            sys.exit(1)

    # Extract zip archive
    zip_path = os.path.join(root_dir, zip_candidates[0])
    print(f"  Extracting {zip_candidates[0]} (this takes ~30 seconds)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root_dir)

    # Verify extracted structure
    if os.path.exists(alt_s1) and not os.path.exists(test_s1):
        try:
            os.symlink(os.path.join(root_dir, "student_resource", "dataset"), os.path.join(root_dir, "dataset"))
        except Exception:
            shutil.copytree(os.path.join(root_dir, "student_resource", "dataset"), os.path.join(root_dir, "dataset"), dirs_exist_ok=True)

    if not os.path.exists(test_s1):
        print(f"\n[ERROR] Extraction finished, but {test_s1} was not found.")
        sys.exit(1)

    print("  ✅ Dataset extracted and ready.")

def run_pipeline(root_dir: str, num_cores: int):
    """Run the high-recall inference pipeline."""
    banner("[Step 3/5] Launching High-Recall Entity Resolution Pipeline...")
    
    model_path = os.path.join(root_dir, "entity_model.joblib")
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file '{model_path}' not found!")
        sys.exit(1)

    # Scale batch size to available hardware
    if num_cores >= 32:
        batch_size = 100000
    elif num_cores >= 16:
        batch_size = 50000
    elif num_cores >= 8:
        batch_size = 25000
    else:
        batch_size = 15000

    print(f"  Detected CPU Threads: {num_cores}")
    print(f"  Configured Batch Size: {batch_size:,} entities per batch")
    print(f"  Model Path:           {model_path}")
    print(f"  Decision Cutoff:       0.660 (Calibrated for Macro F0.5)")

    pipeline_script = os.path.join(root_dir, "code", "business_entity_resolution", "src", "pipeline.py")
    cmd = [
        sys.executable,
        pipeline_script,
        "--model-path", model_path,
        "--batch-size", str(batch_size),
        "--max-candidates", "40"
    ]

    print(f"\n  Starting inference process...")
    print(f"  (Tip: You can run 'python monitor.py' in another terminal to watch live ETA)\n")
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"\n[ERROR] Pipeline failed with return code {ret}")
        sys.exit(ret)

    print("  ✅ Pipeline completed successfully.")

def validate_submission(root_dir: str):
    """Run submission validation checks."""
    banner("[Step 4/5] Running Submission Validator...")
    val_script = os.path.join(root_dir, "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        val_script,
        "--matching", os.path.join(root_dir, "output", "matching_results.tsv"),
        "--candidate", os.path.join(root_dir, "output", "candidate_pairs.tsv"),
        "--test-dir", os.path.join(root_dir, "dataset", "test")
    ]
    subprocess.check_call(cmd)
    print("  ✅ Submission validation PASSED.")

def package_submission(root_dir: str):
    """Package the final submission zip."""
    banner("[Step 5/5] Packaging Final Submission ZIP...")
    zip_script = os.path.join(root_dir, "make_submission_zip.py")
    cmd = [sys.executable, zip_script, "--team-name", "Odin"]
    subprocess.check_call(cmd)
    print("  ✅ Archive created: Odin_submission.zip")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("*" * 78)
    print("   ODIN BUSINESS ENTITY RESOLUTION — MASTER EXECUTION SCRIPT")
    print("   Target: Amazon ML Challenge 2026 (Macro F0.5 >= 0.982)")
    print("*" * 78)

    cores = os.cpu_count() or multiprocessing.cpu_count() or 4
    print(f"  Operating System: {sys.platform} ({os.name})")
    print(f"  Python Version:   {sys.version.split()[0]}")
    print(f"  Available Cores:  {cores}")

    check_dependencies()
    ensure_dataset(root_dir)
    run_pipeline(root_dir, cores)
    validate_submission(root_dir)
    package_submission(root_dir)

    banner("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("  1. Main Submission File:  output/matching_results.tsv  (Upload to Unstop!)")
    print("  2. Full Backup Archive:   Odin_submission.zip")
    print("*" * 78)

if __name__ == "__main__":
    main()
