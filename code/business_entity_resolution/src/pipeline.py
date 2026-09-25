#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - High-Recall, High-Precision Business Entity Resolution Pipeline.
End-to-end pipeline implementing:
1. True positive pair extraction + hard negative candidate generation for model training
2. Clean feature engineering (Unicode diacritics, domain stripping, concatenation, address numbers)
3. LightGBM classifier with validation-calibrated Macro F0.5 decision threshold
4. Scalable country-partitioned streaming batch inference over 1.73M test entities
5. Full compliance with official submission schema and automated validation
"""

import os
import sys
import csv
import time
import gc
import argparse
import subprocess
from typing import Dict, List, Set, Any, Generator
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from blocking import MultiIndexBlocker
from feature_engineering import extract_pair_features
from model import EntityMatcherModel, compute_macro_f05

def stream_tsv_country(filepath: str, target_country: str = None) -> Generator[Dict[str, str], None, None]:
    """Streams TSV records filtering by country without loading full file into RAM."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip header
        for row in reader:
            if not row or len(row) < 4:
                continue
            country = row[3].strip()
            if target_country is None or country == target_country:
                yield {
                    "entity_id": row[0].strip(),
                    "business_name": row[1].strip(),
                    "business_address": row[2].strip(),
                    "country": country
                }

def train_model(args) -> EntityMatcherModel:
    """Trains LightGBM model on real ground truth matches + hard negatives."""
    print(f"\n[Step 1/3] Preparing training data ({args.max_train_entities:,} S1 entities)...")
    s1_train_raw = []
    with open(os.path.join(args.train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= args.max_train_entities:
                break
            if row and len(row) >= 4:
                s1_train_raw.append({
                    "entity_id": row[0].strip(),
                    "business_name": row[1].strip(),
                    "business_address": row[2].strip(),
                    "country": row[3].strip()
                })

    s1_train_ids = {r["entity_id"] for r in s1_train_raw}

    # Load Ground Truth
    ground_truth = {}
    true_target_ids = set()
    with open(os.path.join(args.train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row and row[0] in s1_train_ids:
                matches = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 and row[1] else []
                ground_truth[row[0]] = matches
                true_target_ids.update(matches)

    print(f"  Loaded {len(s1_train_raw):,} S1 training entities with {len(true_target_ids):,} unique true target matches.")

    # Extract target records for true matches + background samples
    print("  Streaming target feeds to collect true matches and hard negative background...")
    targets_map = {}
    t0 = time.time()
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        src_path = os.path.join(args.train_dir, src_file)
        with open(src_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for i, row in enumerate(reader):
                if not row or len(row) < 4:
                    continue
                tid = row[0].strip()
                # Keep if it is a true match OR a sample for negative mining
                if tid in true_target_ids or (i % 80 == 0 and len(targets_map) < len(true_target_ids) + 50000):
                    targets_map[tid] = {
                        "entity_id": tid,
                        "business_name": row[1].strip(),
                        "business_address": row[2].strip(),
                        "country": row[3].strip()
                    }

    print(f"  Extracted {len(targets_map):,} target records in {time.time() - t0:.1f}s.")

    # Candidate Generation for training
    print("  Building training blocker and mining candidate pairs...")
    blocker = MultiIndexBlocker(max_candidates_per_s1=args.max_candidates)
    blocker.build_target_index(list(targets_map.values()), [])
    cand_dict = blocker.query_candidates(s1_train_raw)

    # Train / Val Split by S1 Entity
    all_s1_ids = list(s1_train_ids)
    np.random.seed(42)
    np.random.shuffle(all_s1_ids)
    n_val = int(len(all_s1_ids) * args.val_split)
    val_s1_ids = set(all_s1_ids[:n_val])

    s1_map = {r["entity_id"]: r for r in s1_train_raw}

    X_train_list, y_train_list = [], []
    val_pairs, X_val_list = [], []

    for s1_id in all_s1_ids:
        s1 = s1_map[s1_id]
        true_set = set(ground_truth.get(s1_id, []))
        cands = set(cand_dict.get(s1_id, []))
        is_val = s1_id in val_s1_ids

        # Always include true matches so positive examples are never lost
        all_eval_cands = cands | (true_set & set(targets_map.keys()))

        for cid in all_eval_cands:
            cand = targets_map.get(cid)
            if not cand:
                continue
            feat = extract_pair_features(s1, cand)
            label = 1 if cid in true_set else 0

            if is_val:
                val_pairs.append((s1_id, cid))
                X_val_list.append(feat)
            else:
                X_train_list.append(feat)
                y_train_list.append(label)

    print(f"  Training pairs: {len(X_train_list):,} (Pos: {sum(y_train_list):,}, Neg: {len(y_train_list) - sum(y_train_list):,})")
    print(f"  Validation pairs: {len(X_val_list):,}")

    matcher = EntityMatcherModel(n_estimators=args.n_estimators, learning_rate=0.05)
    matcher.fit(np.array(X_train_list, dtype=np.float32), np.array(y_train_list, dtype=np.int32))

    if X_val_list:
        val_probs = matcher.predict_proba(np.array(X_val_list, dtype=np.float32))
        val_gt = {sid: ground_truth.get(sid, []) for sid in val_s1_ids}
        best_thresh, best_f05 = matcher.optimize_threshold(val_pairs, val_probs, val_gt)
        print(f"  Optimized Threshold: {best_thresh:.3f} (Val Macro F0.5: {best_f05:.4f})")
    else:
        matcher.best_threshold = 0.45

    matcher.save(args.model_path)
    print(f"  Model saved to {args.model_path}")
    return matcher

def run_pipeline():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Business Entity Resolution Pipeline")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train data directory")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test data directory")
    parser.add_argument("--output-dir", default="output", help="Output directory for predictions")
    parser.add_argument("--model-path", default="entity_model.joblib", help="Path to save/load model")
    parser.add_argument("--force-train", action="store_true", help="Force retrain model even if model exists")
    parser.add_argument("--max-train-entities", type=int, default=25000, help="Number of S1 entities to train on")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation split fraction")
    parser.add_argument("--n-estimators", type=int, default=350, help="Number of trees in LightGBM")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max candidates per S1 entity")
    parser.add_argument("--batch-size", type=int, default=25000, help="Batch size for test streaming inference")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto-calibrated)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_t0 = time.time()

    print("=" * 75)
    print("AMAZON ML CHALLENGE 2026 - UPGRADED BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 75)

    # 1. Train or Load Model
    if args.force_train or not os.path.exists(args.model_path):
        matcher = train_model(args)
    else:
        print(f"\n[Step 1/3] Loading pre-trained model from {args.model_path}...")
        matcher = EntityMatcherModel()
        matcher.load(args.model_path)
        print(f"  Model loaded successfully. Calibrated threshold: {matcher.best_threshold:.3f}")

    threshold = args.threshold if args.threshold is not None else matcher.best_threshold
    print(f"  Inference Decision Threshold: {threshold:.3f}")

    # Master dictionaries
    final_matches: Dict[str, List[str]] = {}
    final_candidates: Dict[str, List[str]] = {}

    # Discover test countries dynamically
    countries = ["France", "US", "India"]

    for country in countries:
        c_t0 = time.time()
        print(f"\n[Step 2/3] Processing Country: {country.upper()}...")

        # 1. Stream target feeds for this country
        print(f"  Loading test target records for {country}...")
        targets_raw = []
        for src_file in ["test_source2.tsv", "test_source3.tsv"]:
            src_path = os.path.join(args.test_dir, src_file)
            for r in stream_tsv_country(src_path, target_country=country):
                targets_raw.append(r)

        print(f"  Total target records for {country}: {len(targets_raw):,}")

        print("  Building high-recall inverted index...")
        targets_map = {r["entity_id"]: r for r in targets_raw}

        blocker = MultiIndexBlocker(max_candidates_per_s1=args.max_candidates)
        blocker.build_target_index(targets_raw, [])
        del targets_raw
        gc.collect()
        print(f"  Inverted index built in {time.time() - c_t0:.1f}s.")

        # 2. Stream S1 test entities for this country
        s1_country_raw = list(stream_tsv_country(os.path.join(args.test_dir, "test_source1.tsv"), target_country=country))
        print(f"  Total Source 1 entities for {country}: {len(s1_country_raw):,}")

        # Batch Inference for this country
        pbar = tqdm(total=len(s1_country_raw), desc=f"  Inference {country}", unit=" entities")
        c_matches = 0
        c_singletons = 0

        for i in range(0, len(s1_country_raw), args.batch_size):
            batch = s1_country_raw[i:i + args.batch_size]

            # Query candidates
            batch_cand_dict = blocker.query_candidates(batch)

            # Feature extraction for candidate pairs
            batch_pairs = []
            X_batch_list = []
            for s1 in batch:
                s1_id = s1["entity_id"]
                cands = batch_cand_dict.get(s1_id, [])
                for cid in cands:
                    cand = targets_map.get(cid)
                    if cand:
                        feat = extract_pair_features(s1, cand)
                        batch_pairs.append((s1_id, cid))
                        X_batch_list.append(feat)

            # Prediction with LightGBM
            pair_probs = {}
            if X_batch_list:
                X_batch = np.array(X_batch_list, dtype=np.float32)
                batch_probs = matcher.predict_proba(X_batch)
                for (s1_id, cid), prob in zip(batch_pairs, batch_probs):
                    pair_probs[(s1_id, cid)] = prob

            # Filter with optimal F0.5 threshold
            batch_predictions = matcher.predict_matches(batch_cand_dict, pair_probs, threshold=threshold)

            # Store in master dictionary
            for s1 in batch:
                s1_id = s1["entity_id"]
                preds = batch_predictions.get(s1_id, [])
                cands = batch_cand_dict.get(s1_id, [])

                final_matches[s1_id] = preds
                final_candidates[s1_id] = cands

                if preds:
                    c_matches += 1
                else:
                    c_singletons += 1

            pbar.update(len(batch))

        pbar.close()
        print(f"  Completed {country} in {time.time() - c_t0:.1f}s | Matches: {c_matches:,}, Singletons: {c_singletons:,}")

        # Clean up country structures to free RAM
        del targets_map, blocker, s1_country_raw
        gc.collect()

    print(f"\nAll countries finished in {time.time() - total_t0:.1f}s!")

    # 3. Write Output Files in exact test_source1.tsv order
    print("\n[Step 3/3] Writing final compliant submission TSV files...")
    matching_tsv_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    written_count = 0
    singletons_count = 0

    with open(os.path.join(args.test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f_in, \
         open(matching_tsv_path, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_tsv_path, "w", encoding="utf-8", newline="") as f_cand:

        match_writer = csv.writer(f_match, delimiter="\t", lineterminator="\n")
        cand_writer = csv.writer(f_cand, delimiter="\t", lineterminator="\n")

        # Headers exactly per competition specification
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        reader = csv.reader(f_in, delimiter="\t")
        next(reader, None)  # Skip input header

        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            matches = final_matches.get(s1_id, [])
            cands = final_candidates.get(s1_id, [])

            match_writer.writerow([s1_id, ",".join(matches)])
            cand_writer.writerow([s1_id, ",".join(cands)])

            written_count += 1
            if not matches:
                singletons_count += 1

    print(f"Successfully generated {matching_tsv_path} ({written_count:,} rows, {singletons_count:,} singletons)")
    print(f"Successfully generated {candidate_tsv_path} ({written_count:,} rows)")

    # Validate output format
    print("\nValidating generated submission with official utils/validate_submission.py...")
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", matching_tsv_path,
        "--candidate", candidate_tsv_path,
        "--test-dir", args.test_dir
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode == 0:
        print("=" * 75)
        print("PASS: SUBMISSION FILES ARE 100% COMPLIANT AND READY FOR UPLOAD!")
        print("=" * 75)
    else:
        print("Validation errors encountered:")
        print(res.stderr)

if __name__ == "__main__":
    run_pipeline()
