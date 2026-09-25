#!/usr/bin/env python3
"""
Packages the complete Amazon ML Challenge submission zip archive.
Ensures correct directory structure, inclusion of output TSVs, source code,
requirements, and documentation.
"""

import argparse
import os
import zipfile
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Build submission zip package.")
    parser.add_argument("--team-name", default="ReddyTeam", help="Your team name for the zip archive")
    args = parser.parse_args()

    zip_filename = f"{args.team_name}_submission.zip"
    print(f"Creating submission package: {zip_filename}")

    # Validate first
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Pre-packaging validation failed:")
        print(res.stdout)
        print(res.stderr)
        sys.exit(1)

    print("Pre-packaging validation passed!")

    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/src/__init__.py", "code/business_entity_resolution/src/__init__.py"),
        ("code/business_entity_resolution/src/preprocessing.py", "code/business_entity_resolution/src/preprocessing.py"),
        ("code/business_entity_resolution/src/blocking.py", "code/business_entity_resolution/src/blocking.py"),
        ("code/business_entity_resolution/src/feature_engineering.py", "code/business_entity_resolution/src/feature_engineering.py"),
        ("code/business_entity_resolution/src/model.py", "code/business_entity_resolution/src/model.py"),
        ("code/business_entity_resolution/src/pipeline.py", "code/business_entity_resolution/src/pipeline.py"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_path, arc_path in files_to_pack:
            if not os.path.exists(local_path):
                print(f"ERROR: Missing expected file: {local_path}")
                sys.exit(1)
            zipf.write(local_path, arc_path)
            print(f"  Added {arc_path}")

    print(f"\nSuccessfully generated {zip_filename} ({os.path.getsize(zip_filename)} bytes)")

if __name__ == "__main__":
    main()
