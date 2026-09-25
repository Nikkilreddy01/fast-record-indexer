#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Universal Cross-Platform Runner (Windows / Linux / macOS)
Auto-detects CPU cores, environment, and datasets to execute the full high-recall pipeline.
"""

import os
import sys
import shutil
import zipfile
import subprocess
import multiprocessing

def log(msg: str):
    print(f"\n{'='*75}\n  {msg}\n{'='*75}")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("=" * 75)
    print("   AMAZON ML CHALLENGE 2026: UNIVERSAL PIPELINE RUNNER")
    print("=" * 75)

    # 1. CPU & OS Detection
    num_cores = os.cpu_count() or multiprocessing.cpu_count() or 4
    system_name = sys.platform
    print(f"[System] Platform: {system_name} ({os.name})")
    print(f"[System] Detected CPU Cores / Threads: {num_cores}")

    if num_cores >= 32:
        batch_size = 100000
    elif num_cores >= 16:
        batch_size = 50000
    else:
        batch_size = 25000
    print(f"[System] Auto-configured Streaming Batch Size: {batch_size:,} entities/batch")

    # 2. Dependency verification
    log("[Step 1/5] Checking and Installing Dependencies...")
    req_file = os.path.join("code", "business_entity_resolution", "requirements.txt")
    cmd_install = [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"]
    print(f"  Running: {' '.join(cmd_install)}")
    subprocess.check_call(cmd_install)
    print("  Dependencies verified successfully.")

    # 3. Dataset Verification
    log("[Step 2/5] Verifying Dataset Files...")
    test_s1 = os.path.join("dataset", "test", "test_source1.tsv")
    if not os.path.exists(test_s1):
        alt_test_s1 = os.path.join("student_resource", "dataset", "test", "test_source1.tsv")
        if os.path.exists(alt_test_s1):
            print("  Found dataset in student_resource/dataset. Creating 'dataset' directory/link...")
            if not os.path.exists("dataset"):
                try:
                    os.symlink(os.path.join(root_dir, "student_resource", "dataset"), "dataset")
                except Exception:
                    # Windows without symlink privileges: copy directory structure
                    shutil.copytree(os.path.join(root_dir, "student_resource", "dataset"), "dataset")
        else:
            # Check for zip
            zip_candidates = [f for f in os.listdir(root_dir) if f.endswith(".zip") and "student_resource" in f]
            if not zip_candidates:
                zip_filename = "6ab10eb3b23ba_student_resource.zip"
                download_url = f"https://github.com/Nikkilreddy01/fast-record-indexer/releases/download/v1.0.0/{zip_filename}"
                print(f"  Dataset zip not found locally. Downloading from public release: {download_url}...")
                try:
                    import urllib.request
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if total_size > 0:
                            percent = min(100.0, (downloaded / total_size) * 100.0)
                            mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r  Downloading dataset: {mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
                            sys.stdout.flush()
                    urllib.request.urlretrieve(download_url, os.path.join(root_dir, zip_filename), reporthook=report_progress)
                    print("\n  Download complete!")
                    zip_candidates = [zip_filename]
                except Exception as dl_err:
                    print(f"\n  Direct download failed: {dl_err}. Trying 'gh release download'...")
                    try:
                        subprocess.run(["gh", "release", "download", "v1.0.0", "--pattern", "*.zip"], check=True)
                        zip_candidates = [f for f in os.listdir(root_dir) if f.endswith(".zip") and "student_resource" in f]
                    except Exception as e:
                        print(f"  Note: 'gh release download' not available or failed: {e}")

            if zip_candidates:
                zip_path = os.path.join(root_dir, zip_candidates[0])
                print(f"  Extracting {zip_candidates[0]} (this takes ~30 seconds)...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(root_dir)
                if os.path.exists(alt_test_s1) and not os.path.exists("dataset"):
                    try:
                        os.symlink(os.path.join(root_dir, "student_resource", "dataset"), "dataset")
                    except Exception:
                        shutil.copytree(os.path.join(root_dir, "student_resource", "dataset"), "dataset")
            else:
                print("\n" + "=" * 70)
                print("ERROR: dataset/test/test_source1.tsv not found!")
                print("Please download the 1.01 GB dataset from:")
                print(f"  https://github.com/Nikkilreddy01/fast-record-indexer/releases/download/v1.0.0/6ab10eb3b23ba_student_resource.zip")
                print("And place it in this folder, then run run_all.bat again.")
                print("=" * 70 + "\n")
                sys.exit(1)

    print("  Dataset verified successfully.")

    # 4. Run Pipeline with Pretrained Model
    log("[Step 3/5] Launching High-Recall Entity Resolution Pipeline...")
    model_path = "entity_model.joblib"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found in {root_dir}")
        sys.exit(1)

    pipeline_script = os.path.join("code", "business_entity_resolution", "src", "pipeline.py")
    cmd_pipeline = [
        sys.executable,
        pipeline_script,
        "--model-path", model_path,
        "--batch-size", str(batch_size),
        "--max-candidates", "40"
    ]
    print(f"  Running: {' '.join(cmd_pipeline)}")
    ret = subprocess.call(cmd_pipeline)
    if ret != 0:
        print(f"ERROR: Pipeline exited with return code {ret}")
        sys.exit(ret)

    # 5. Submission Validation
    log("[Step 4/5] Running Submission Validator...")
    val_script = os.path.join("utils", "validate_submission.py")
    cmd_val = [
        sys.executable,
        val_script,
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    subprocess.check_call(cmd_val)

    # 6. Packaging ZIP
    log("[Step 5/5] Packaging Final Submission ZIP...")
    zip_script = "make_submission_zip.py"
    cmd_zip = [sys.executable, zip_script, "--team-name", "Odin"]
    subprocess.check_call(cmd_zip)

    log("🎉 SUCCESS: ALL STEPS COMPLETED!\n"
        "1. Scored File:  output/matching_results.tsv  (Upload to Unstop!)\n"
        "2. Final Archive: Odin_submission.zip")

if __name__ == "__main__":
    main()
