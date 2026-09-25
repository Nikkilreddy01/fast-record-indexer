"""
Machine Learning Model & Precision-Tuned F0.5 Optimization Module.
Trains LightGBM classifier and calibrates thresholds to maximize Macro F0.5 score.
"""

from typing import Dict, List, Tuple
import numpy as np
import lightgbm as lgb
import joblib

def compute_entity_f05(pred_ids: List[str], true_ids: List[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - true empty and pred empty => 1.0
      - true empty and pred non-empty => 0.0
      - true non-empty and pred empty => 0.0
    """
    p_set = set(pred_ids)
    t_set = set(true_ids)

    # Singleton case
    if len(t_set) == 0:
        return 1.0 if len(p_set) == 0 else 0.0

    if len(p_set) == 0:
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

def compute_macro_f05(predictions: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
    """Computes Macro F_0.5 across all Source 1 entities."""
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        scores.append(compute_entity_f05(pred_matches, true_matches))
    return float(np.mean(scores)) if scores else 0.0

class EntityMatcherModel:
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1
        )
        self.best_threshold = 0.70 # Default precision-heavy threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the underlying gradient booster."""
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probabilities for candidate pairs."""
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        val_candidate_pairs: List[Tuple[str, str]], # (s1_id, cand_id)
        val_probs: np.ndarray,
        val_ground_truth: Dict[str, List[str]]
    ) -> Tuple[float, float]:
        """
        Performs grid search to find the optimal decision threshold maximizing Macro F0.5.
        """
        # Group candidate probabilities by s1_id
        grouped_preds = {}
        for (s1_id, cand_id), prob in zip(val_candidate_pairs, val_probs):
            if s1_id not in grouped_preds:
                grouped_preds[s1_id] = []
            grouped_preds[s1_id].append((cand_id, prob))

        best_score = -1.0
        best_thresh = 0.45
        thresholds = np.linspace(0.20, 0.80, 61)

        for thresh in thresholds:
            current_preds = {}
            for s1_id in val_ground_truth.keys():
                cands = grouped_preds.get(s1_id, [])
                # Select candidates meeting threshold
                matched = [cid for cid, prob in cands if prob >= thresh]
                current_preds[s1_id] = matched

            score = compute_macro_f05(current_preds, val_ground_truth)
            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.best_threshold = float(best_thresh)
        return self.best_threshold, best_score

    def predict_matches(
        self,
        candidate_dict: Dict[str, List[str]],
        pair_probabilities: Dict[Tuple[str, str], float],
        threshold: float = None
    ) -> Dict[str, List[str]]:
        """
        Produces final matched entity IDs for every Source 1 entity.
        """
        if threshold is None:
            threshold = self.best_threshold

        predictions = {}
        for s1_id, cands in candidate_dict.items():
            matches = []
            for cid in cands:
                prob = pair_probabilities.get((s1_id, cid), 0.0)
                if prob >= threshold:
                    matches.append(cid)
            predictions[s1_id] = matches
        return predictions

    def save(self, filepath: str):
        joblib.dump({"model": self.clf, "threshold": self.best_threshold}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.clf = data["model"]
        self.best_threshold = data["threshold"]
