"""
src/business_entity_resolution/threshold.py
===========================================
Decision threshold optimization for precision-weighted F0.5 evaluation.
Finds the threshold that maximizes validation entity-level Macro F0.5.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from business_entity_resolution.config import config
from business_entity_resolution.metrics import evaluate_entity_macro_f05

logger = logging.getLogger("BER.Threshold")


def optimize_threshold(
    pair_list: List[Tuple[str, str]],
    probabilities: List[float],
    gt_mapping: Dict[str, Set[str]],
    all_val_s1_ids: Set[str],
    search_grid: List[float] = config.threshold_search_grid,
) -> Tuple[float, dict, pd.DataFrame]:
    """
    Search threshold values over validation candidate predictions.
    Computes Macro F0.5 for each threshold and returns:
    (best_threshold, best_metrics, summary_dataframe)
    """
    # Group predictions by s1_id: s1_id -> list of (cand_id, prob)
    pred_by_s1 = defaultdict(list)
    for (s1_id, cand_id), p in zip(pair_list, probabilities):
        pred_by_s1[s1_id].append((cand_id, float(p)))

    results = []
    best_thresh = config.default_match_threshold
    best_f05 = -1.0
    best_metrics = {}

    print(f"\n{'='*75}\n[THRESHOLD OPTIMIZATION] Searching grid: {search_grid}\n{'='*75}")
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Singleton Acc':<14} | {'Matches':<8}")
    print("-" * 75)

    for thresh in search_grid:
        # Build predictions at this threshold
        pred_mapping: Dict[str, Set[str]] = {}
        for s1_id in all_val_s1_ids:
            matches = {cand_id for cand_id, p in pred_by_s1.get(s1_id, []) if p >= thresh}
            pred_mapping[s1_id] = matches

        eval_res = evaluate_entity_macro_f05(gt_mapping, pred_mapping, beta=config.f_beta)
        f05 = eval_res["macro_f05"]
        prec = eval_res["macro_precision"]
        rec = eval_res["macro_recall"]
        sing_acc = eval_res["singleton_accuracy_pct"]
        n_matches = eval_res["total_predicted_matches"]

        print(f"{thresh:<10.2f} | {f05:<12.4f} | {prec:<12.4f} | {rec:<12.4f} | {sing_acc:<13.1f}% | {n_matches:<8}")

        results.append({
            "threshold": thresh,
            "macro_f05": f05,
            "macro_precision": prec,
            "macro_recall": rec,
            "singleton_accuracy_pct": sing_acc,
            "total_matches": n_matches,
        })

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = eval_res

    print("-" * 75)
    print(f"[THRESHOLD] Best Threshold: {best_thresh:.2f} (Macro F0.5: {best_f05:.4f})")

    # Save artifact
    thresh_data = {
        "best_threshold": best_thresh,
        "best_macro_f05": best_f05,
        "best_macro_precision": best_metrics.get("macro_precision", 0.0),
        "best_macro_recall": best_metrics.get("macro_recall", 0.0),
        "singleton_accuracy_pct": best_metrics.get("singleton_accuracy_pct", 0.0),
    }
    with open(config.threshold_artifact_path, "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)

    df_summary = pd.DataFrame(results)
    return best_thresh, best_metrics, df_summary


def load_best_threshold() -> float:
    """Load optimized threshold from JSON artifact, or fall back to default."""
    if config.threshold_artifact_path.is_file():
        try:
            with open(config.threshold_artifact_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("best_threshold", config.default_match_threshold))
        except Exception:
            pass
    return config.default_match_threshold
