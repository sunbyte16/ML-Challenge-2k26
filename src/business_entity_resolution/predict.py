"""
src/business_entity_resolution/predict.py
=========================================
Inference pipeline for test dataset in Amazon ML Challenge 2026.
Runs preprocessing, candidate generation, pairwise feature extraction,
matching model probability inference, thresholding, and output file generation.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from business_entity_resolution.config import config
from business_entity_resolution.data_loader import load_test_data
from business_entity_resolution.preprocessing import preprocess_dataframe
from business_entity_resolution.blocking import MultiPassBlocker
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.model import MatchClassifier
from business_entity_resolution.threshold import load_best_threshold
from business_entity_resolution.output import write_matching_results, write_candidate_pairs

logger = logging.getLogger("BER.Predict")


def run_inference(
    test_data: Optional[Dict[str, pd.DataFrame]] = None,
    model: Optional[MatchClassifier] = None,
    threshold: Optional[float] = None,
    cfg=config,
) -> Tuple[Path, Path]:
    """
    Run full inference on test datasets and write official submission files.
    Returns: (matching_results_path, candidate_pairs_path)
    """
    # 1. Load test data
    if test_data is None:
        logger.info("[INFERENCE] Loading test dataset...")
        test_data = load_test_data(cfg)

    df_s1 = test_data["source1"]
    df_s2 = test_data["source2"]
    df_s3 = test_data["source3"]

    logger.info(
        "[INFERENCE] Test data loaded: %d S1, %d S2, %d S3 records",
        len(df_s1), len(df_s2), len(df_s3),
    )

    # 2. Text Normalization
    logger.info("[INFERENCE] Normalizing test records...")
    df_s1_norm = preprocess_dataframe(df_s1, is_source1=True)
    df_s2_norm = preprocess_dataframe(df_s2, is_source1=False)
    df_s3_norm = preprocess_dataframe(df_s3, is_source1=False)
    df_targets = pd.concat([df_s2_norm, df_s3_norm], ignore_index=True)

    # 3. Candidate Generation (Blocking)
    logger.info("[INFERENCE] Running multi-pass candidate generation...")
    blocker = MultiPassBlocker(
        max_candidates_per_s1=cfg.max_candidates_per_s1,
        max_bucket_size=cfg.max_block_bucket_size,
        use_tfidf_fallback=cfg.use_tfidf_char_block,
    )
    candidates = blocker.generate_candidates(df_s1_norm, df_s2_norm, df_s3_norm)

    # 4. Feature Extraction for Candidate Pairs
    logger.info("[INFERENCE] Extracting pairwise features for candidates...")
    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1_norm.iterrows()}
    target_dict = {row["entity_id"]: row.to_dict() for _, row in df_targets.iterrows()}

    pair_list: List[Tuple[str, str]] = []
    features_list: List[List[float]] = []

    for s1_id, cand_set in tqdm(candidates.items(), desc="Test Pair Features"):
        s1_rec = s1_dict[s1_id]
        for cid in cand_set:
            if cid in target_dict:
                pair_list.append((s1_id, cid))
                features_list.append(compute_pair_features(s1_rec, target_dict[cid]))

    logger.info("[INFERENCE] Computed features for %d candidate pairs.", len(pair_list))

    # 5. Load Model & Score Probabilities
    if model is None:
        logger.info("[INFERENCE] Loading trained model from %s...", cfg.model_artifact_path)
        model = MatchClassifier.load(cfg.model_artifact_path, cfg.feature_metadata_path)

    match_thresh = threshold if threshold is not None else load_best_threshold()
    logger.info("[INFERENCE] Using decision threshold: %.3f", match_thresh)

    if features_list:
        X_test = np.array(features_list, dtype=np.float32)
        probabilities = model.predict_proba(X_test)
    else:
        probabilities = np.empty((0,), dtype=np.float32)

    # 6. Apply Threshold & Singletons
    pred_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in df_s1["entity_id"]}

    for (s1_id, cid), prob in zip(pair_list, probabilities):
        if prob >= match_thresh:
            pred_matches[s1_id].add(cid)

    total_matches = sum(len(m) for m in pred_matches.values())
    singletons = sum(1 for m in pred_matches.values() if len(m) == 0)
    logger.info(
        "[INFERENCE] Matches predicted: %d across %d S1 entities (%d singletons, %.1f%%)",
        total_matches, len(df_s1), singletons, (singletons / max(1, len(df_s1))) * 100,
    )

    # 7. Write Submission Files
    all_s1_ids = df_s1["entity_id"].tolist()
    match_path = write_matching_results(all_s1_ids, pred_matches, cfg.matching_output_path)
    cand_path = write_candidate_pairs(all_s1_ids, candidates, cfg.candidate_output_path)

    print(f"\n{'='*70}\n[INFERENCE COMPLETED]\n{'='*70}")
    print(f"  matching_results.tsv: {match_path}")
    print(f"  candidate_pairs.tsv:  {cand_path}")
    print(f"  Total test entities:  {len(all_s1_ids):,}")
    print(f"  Total matches:        {total_matches:,}")
    print(f"  Singletons:           {singletons:,}")
    return match_path, cand_path
