#!/usr/bin/env python3
"""
Local Validation & Macro F0.5 Scorer for Amazon ML Challenge 2026.
Computes the EXACT competition leaderboard evaluation metric:
Macro F_0.5 Score across all entities with strict singleton handling.

Usage:
    python3 utils/eval_local_f05.py \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --predictions output/val_predictions.tsv \
        --sample-size 50000
"""

import argparse
import csv
import sys
from typing import Dict, List, Set
import numpy as np

def compute_entity_f05(pred_ids: List[str], true_ids: List[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Strict singleton scoring:
      - true empty and pred empty => 1.0
      - true empty and pred non-empty => 0.0
      - true non-empty and pred empty => 0.0
    """
    p_set = set(pred_ids)
    t_set = set(true_ids)

    # Both empty -> True Singleton (Full Credit)
    if not t_set and not p_set:
        return 1.0

    # Predicted matches for a singleton -> False Merge (Zero Credit)
    if not t_set and p_set:
        return 0.0

    # Predicted empty for an entity that has matches -> Missed Matches (Zero Credit)
    if t_set and not p_set:
        return 0.0

    true_positives = len(p_set & t_set)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(p_set)
    recall = true_positives / len(t_set)

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def evaluate_files(gt_path: str, pred_path: str, max_entities: int = None):
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 - EXACT MACRO F0.5 SCORER")
    print("=" * 70)
    print(f"Loading Ground Truth from: {gt_path}")
    print(f"Loading Predictions from:  {pred_path}")

    # Load Ground Truth
    gt: Dict[str, List[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for i, row in enumerate(reader):
            if max_entities and i >= max_entities:
                break
            if not row:
                continue
            s1_id = row[0].strip()
            matches = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 and row[1] else []
            gt[s1_id] = matches

    # Load Predictions
    preds: Dict[str, List[str]] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if s1_id in gt:
                matches = [m.strip() for m in row[1].split(",") if m.strip()] if len(row) > 1 and row[1] else []
                preds[s1_id] = matches

    print(f"Evaluating {len(gt):,} entities...")

    scores = []
    true_singletons = 0
    pred_singletons = 0
    clean_singletons = 0
    polluted_singletons = 0
    false_singletons = 0

    precisions = []
    recalls = []

    for s1_id, true_matches in gt.items():
        pred_matches = preds.get(s1_id, [])
        score = compute_entity_f05(pred_matches, true_matches)
        scores.append(score)

        t_set = set(true_matches)
        p_set = set(pred_matches)

        if not t_set:
            true_singletons += 1
            if not p_set:
                clean_singletons += 1
            else:
                polluted_singletons += 1
        else:
            if not p_set:
                false_singletons += 1
            else:
                tp = len(t_set & p_set)
                precisions.append(tp / len(p_set))
                recalls.append(tp / len(t_set))

        if not p_set:
            pred_singletons += 1

    macro_f05 = float(np.mean(scores))
    mean_p = float(np.mean(precisions)) if precisions else 0.0
    mean_r = float(np.mean(recalls)) if recalls else 0.0

    print("-" * 70)
    print("DETAILED METRIC BREAKDOWN:")
    print(f"  • Total Entities Evaluated:     {len(scores):,}")
    print(f"  • True Singletons in Data:       {true_singletons:,} ({true_singletons / len(scores) * 100:.2f}%)")
    print(f"  • Correctly Kept Singletons:     {clean_singletons:,} / {true_singletons:,} ({clean_singletons / max(1, true_singletons) * 100:.2f}% accuracy)")
    print(f"  • Polluted Singletons (FP error):{polluted_singletons:,} ({polluted_singletons / max(1, true_singletons) * 100:.2f}%)")
    print(f"  • False Singletons (FN error):   {false_singletons:,} (entities with matches falsely predicted empty)")
    print(f"  • Average Precision on Matches:  {mean_p * 100:.2f}%")
    print(f"  • Average Recall on Matches:     {mean_r * 100:.2f}%")
    print("-" * 70)
    print(f"🏆 EXACT MACRO F_0.5 LEADERBOARD SCORE: {macro_f05:.6f} ({macro_f05 * 100:.3f}%)")
    print("=" * 70)

    return macro_f05

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate exact Macro F0.5 score locally")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv", help="Ground truth TSV")
    parser.add_argument("--predictions", required=True, help="Predictions TSV to evaluate")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (default: all)")
    args = parser.parse_args()

    evaluate_files(args.ground_truth, args.predictions, args.sample_size)
