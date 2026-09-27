"""
src/business_entity_resolution/pipeline.py
==========================================
Master execution pipeline and CLI for Amazon ML Challenge 2026.
Supports modes: eda, train, validate, predict, all.
"""

import argparse
import datetime
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from business_entity_resolution.config import config
from business_entity_resolution.data_loader import (
    load_train_data,
    load_test_data,
    parse_ground_truth_mapping,
    profile_dataset,
    profile_ground_truth,
)
from business_entity_resolution.preprocessing import preprocess_dataframe
from business_entity_resolution.blocking import MultiPassBlocker
from business_entity_resolution.dataset_builder import entity_aware_split, build_pairwise_dataset
from business_entity_resolution.model import MatchClassifier
from business_entity_resolution.threshold import optimize_threshold, load_best_threshold
from business_entity_resolution.metrics import evaluate_entity_macro_f05
from business_entity_resolution.predict import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BER.Pipeline")


def run_eda(cfg=config) -> dict:
    """Run exploratory data analysis on train and test datasets."""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA EXPLORATION AND PROFILING")
    print("=" * 80)

    train_data = load_train_data(cfg)
    profile_dataset(train_data["source1"], "TRAIN SOURCE 1")
    profile_dataset(train_data["source2"], "TRAIN SOURCE 2")
    profile_dataset(train_data["source3"], "TRAIN SOURCE 3")
    gt_stats = profile_ground_truth(train_data["ground_truth"])

    try:
        test_data = load_test_data(cfg)
        profile_dataset(test_data["source1"], "TEST SOURCE 1")
        profile_dataset(test_data["source2"], "TEST SOURCE 2")
        profile_dataset(test_data["source3"], "TEST SOURCE 3")
    except Exception as e:
        logger.warning("[DATA] Test dataset not available or incomplete: %s", e)

    return gt_stats


def run_train(cfg=config, model_type: Optional[str] = None) -> MatchClassifier:
    """Train the entity matching classifier with entity-aware validation split."""
    print("\n" + "=" * 80)
    print("PHASE 2: TRAINING PIPELINE")
    print("=" * 80)
    cfg.ensure_dirs()

    # 1. Load Data
    train_data = load_train_data(cfg)
    df_s1 = train_data["source1"]
    df_s2 = train_data["source2"]
    df_s3 = train_data["source3"]
    gt_mapping = parse_ground_truth_mapping(train_data["ground_truth"])

    # 2. Text Normalization
    logger.info("[PREPROCESSING] Normalizing train datasets...")
    df_s1_norm = preprocess_dataframe(df_s1, is_source1=True)
    df_s2_norm = preprocess_dataframe(df_s2, is_source1=False)
    df_s3_norm = preprocess_dataframe(df_s3, is_source1=False)
    df_targets = pd.concat([df_s2_norm, df_s3_norm], ignore_index=True)

    # 3. Entity-Aware Validation Split
    train_s1, val_s1 = entity_aware_split(df_s1_norm, val_ratio=cfg.val_split_ratio, random_seed=cfg.random_seed)

    # 4. Multi-Pass Blocking on Train Split
    logger.info("[BLOCKING] Generating candidates for training split (%d entities)...", len(train_s1))
    blocker = MultiPassBlocker(
        max_candidates_per_s1=cfg.max_candidates_per_s1,
        max_bucket_size=cfg.max_block_bucket_size,
        use_tfidf_fallback=cfg.use_tfidf_char_block,
    )
    train_cands = blocker.generate_candidates(train_s1, df_s2_norm, df_s3_norm)
    blocker.evaluate_blocking(train_cands, gt_mapping, len(df_targets))

    # 5. Build Training Dataset (Positives + Hard Negatives)
    logger.info("[FEATURES] Building pairwise training dataset...")
    X_train, y_train, train_pairs = build_pairwise_dataset(
        train_s1, df_targets, train_cands, gt_mapping, max_negatives_per_pos=cfg.max_negatives_per_positive, is_training=True
    )

    # 6. Fit Classifier
    m_type = model_type or cfg.model_type
    model = MatchClassifier(model_type=m_type)
    model.fit(X_train, y_train)
    model.save(cfg.model_artifact_path, cfg.feature_metadata_path)

    # 7. Evaluate on Validation Set
    logger.info("[EVALUATION] Generating candidates for validation split (%d entities)...", len(val_s1))
    val_cands = blocker.generate_candidates(val_s1, df_s2_norm, df_s3_norm)
    blocker.evaluate_blocking(val_cands, gt_mapping, len(df_targets))

    X_val, y_val, val_pairs = build_pairwise_dataset(
        val_s1, df_targets, val_cands, gt_mapping, max_negatives_per_pos=0, is_training=False
    )
    val_probs = model.predict_proba(X_val)

    # 8. Optimize Decision Threshold
    all_val_s1_ids = set(val_s1["entity_id"])
    best_thresh, best_metrics, df_summary = optimize_threshold(
        val_pairs, val_probs, gt_mapping, all_val_s1_ids, search_grid=cfg.threshold_search_grid
    )

    # 9. Log Experiment
    log_experiment(
        cfg=cfg,
        model_type=m_type,
        best_threshold=best_thresh,
        metrics=best_metrics,
        num_train_pairs=len(y_train),
        num_val_pairs=len(y_val),
    )

    return model


def log_experiment(cfg, model_type, best_threshold, metrics, num_train_pairs, num_val_pairs, notes=""):
    """Append experiment result to CSV log."""
    log_path = cfg.experiment_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "best_threshold": best_threshold,
        "macro_f05": round(metrics.get("macro_f05", 0.0), 4),
        "macro_precision": round(metrics.get("macro_precision", 0.0), 4),
        "macro_recall": round(metrics.get("macro_recall", 0.0), 4),
        "singleton_accuracy_pct": metrics.get("singleton_accuracy_pct", 0.0),
        "num_train_pairs": num_train_pairs,
        "num_val_pairs": num_val_pairs,
        "notes": notes,
    }

    df_record = pd.DataFrame([record])
    if log_path.is_file():
        df_record.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_record.to_csv(log_path, mode="w", header=True, index=False)
    logger.info("[EXPERIMENT] Recorded run into %s", log_path)


def run_official_validator(cfg=config) -> bool:
    """Execute the official submission validator script."""
    print("\n" + "=" * 80)
    print("PHASE 4: OFFICIAL SUBMISSION VALIDATION")
    print("=" * 80)

    validator_script = cfg.project_root / "utils" / "validate_submission.py"
    if not validator_script.is_file():
        logger.error("[VALIDATOR] Validator script not found at %s", validator_script)
        return False

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(cfg.matching_output_path),
        "--candidate", str(cfg.candidate_output_path),
        "--test-dir", str(cfg.test_dir),
    ]

    logger.info("[VALIDATOR] Running command: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode == 0:
        print("[VALIDATOR] SUCCESS: PASS")
        print("Your submission files are strictly compliant with official competition rules.")
        return True
    else:
        print("[VALIDATOR] FAILED with issues:")
        print(res.stdout)
        print(res.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["eda", "train", "validate", "predict", "all"],
        default="all",
        help="Pipeline execution mode.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override match decision threshold.")
    parser.add_argument("--model", type=str, default=None, help="Model type (logistic_regression, hist_gradient_boosting, lightgbm).")
    args = parser.parse_args()

    cfg = config
    cfg.ensure_dirs()

    if args.mode == "eda":
        run_eda(cfg)

    elif args.mode == "train":
        run_train(cfg, model_type=args.model)

    elif args.mode == "predict":
        run_inference(threshold=args.threshold, cfg=cfg)
        run_official_validator(cfg)

    elif args.mode == "validate":
        run_official_validator(cfg)

    elif args.mode == "all":
        run_eda(cfg)
        run_train(cfg, model_type=args.model)
        run_inference(threshold=args.threshold, cfg=cfg)
        run_official_validator(cfg)


if __name__ == "__main__":
    main()
