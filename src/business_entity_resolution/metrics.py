"""
src/business_entity_resolution/metrics.py
=========================================
Official competition evaluation metric implementation for Amazon ML Challenge 2026.
Computes entity-level macro F0.5 with exact singleton handling rules.
"""

from typing import Dict, Set, Tuple
import numpy as np


def compute_single_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_beta (default F0.5) for a single Source 1 entity.

    Official singleton rule:
    - If true_set is empty and pred_set is empty:
      Precision = 1.0, Recall = 1.0, F0.5 = 1.0
    - If true_set is empty and pred_set is not empty:
      Precision = 0.0, Recall = 0.0, F0.5 = 0.0
    - If true_set is not empty and pred_set is empty:
      Precision = 0.0, Recall = 0.0, F0.5 = 0.0
    - Otherwise:
      TP = |true & pred|
      FP = |pred - true|
      FN = |true - pred|
      Precision = TP / (TP + FP)
      Recall = TP / (TP + FN)
      F0.5 = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    beta_sq = beta ** 2

    # Case 1: True singleton (no matches in ground truth)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # Case 2: True non-singleton, but prediction is empty
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    # Case 3: Both non-empty
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0, 0.0, 0.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    denom = (beta_sq * prec) + rec
    f_score = ((1.0 + beta_sq) * prec * rec) / denom if denom > 0 else 0.0
    return prec, rec, f_score


def evaluate_entity_macro_f05(
    gt_mapping: Dict[str, Set[str]],
    pred_mapping: Dict[str, Set[str]],
    beta: float = 0.5,
) -> dict:
    """
    Compute macro-averaged entity-level F0.5 over all Source 1 entities.
    Returns comprehensive metrics dictionary.
    """
    all_s1_ids = set(gt_mapping.keys()) | set(pred_mapping.keys())
    if not all_s1_ids:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    precisions = []
    recalls = []
    f_scores = []

    # Singleton tracking
    singleton_total = 0
    singleton_correct = 0

    # Micro aggregations
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_matches = 0

    for s1_id in sorted(all_s1_ids):
        true_s = gt_mapping.get(s1_id, set())
        pred_s = pred_mapping.get(s1_id, set())
        total_pred_matches += len(pred_s)

        p, r, f = compute_single_entity_f05(true_s, pred_s, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

        if len(true_s) == 0:
            singleton_total += 1
            if len(pred_s) == 0:
                singleton_correct += 1

        tp = len(true_s & pred_s)
        fp = len(pred_s - true_s)
        fn = len(true_s - pred_s)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_prec = total_tp / max(1, total_tp + total_fp)
    micro_rec = total_tp / max(1, total_tp + total_fn)
    micro_denom = (beta ** 2 * micro_prec) + micro_rec
    micro_f05 = ((1.0 + beta ** 2) * micro_prec * micro_rec) / micro_denom if micro_denom > 0 else 0.0

    singleton_acc = (singleton_correct / singleton_total * 100) if singleton_total > 0 else 100.0

    return {
        "macro_f05": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f05": float(micro_f05),
        "total_entities_evaluated": len(all_s1_ids),
        "total_predicted_matches": total_pred_matches,
        "singleton_entities": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy_pct": round(singleton_acc, 2),
    }
