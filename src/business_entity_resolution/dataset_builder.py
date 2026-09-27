"""
src/business_entity_resolution/dataset_builder.py
=================================================
Constructs pairwise training and validation datasets with:
- Entity-aware train/val split on Source 1 entities (avoids data leakage)
- Ground-truth positive pairs
- Hard negative pairs mined directly from blocking buckets
"""

import logging
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from business_entity_resolution.config import config
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES

logger = logging.getLogger("BER.DatasetBuilder")


def entity_aware_split(
    df_s1: pd.DataFrame,
    val_ratio: float = config.val_split_ratio,
    random_seed: int = config.random_seed,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Source 1 entities into train and validation sets.
    Ensures that validation Source 1 entities never leak into training.
    """
    s1_ids = df_s1["entity_id"].unique()
    train_ids, val_ids = train_test_split(s1_ids, test_size=val_ratio, random_state=random_seed)

    train_df = df_s1[df_s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df_s1[df_s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    logger.info(
        "[DATASET] Entity-aware split: %d train S1 entities, %d val S1 entities",
        len(train_df), len(val_df),
    )
    return train_df, val_df


def build_pairwise_dataset(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    candidates: Dict[str, Set[str]],
    gt_mapping: Dict[str, Set[str]],
    max_negatives_per_pos: int = config.max_negatives_per_positive,
    is_training: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build feature matrix X, label array y, and pair list [(s1_id, cand_id)]:
    - Positive pairs: candidate in gt_mapping[s1_id]
    - Hard negative pairs: candidate NOT in gt_mapping[s1_id]
    """
    # Index records by entity_id
    s1_dict = {row["entity_id"]: row.to_dict() for _, row in df_s1.iterrows()}
    target_dict = {row["entity_id"]: row.to_dict() for _, row in df_targets.iterrows()}

    rng = random.Random(config.random_seed)

    pair_list: List[Tuple[str, str]] = []
    features_list: List[List[float]] = []
    labels_list: List[int] = []

    pos_count = 0
    neg_count = 0

    for s1_id, cand_set in tqdm(candidates.items(), desc="Building Pairwise Dataset"):
        if s1_id not in s1_dict:
            continue
        s1_rec = s1_dict[s1_id]
        true_matches = gt_mapping.get(s1_id, set())

        # Separate candidates into positives and negatives
        pos_candidates = [cid for cid in cand_set if cid in true_matches and cid in target_dict]
        neg_candidates = [cid for cid in cand_set if cid not in true_matches and cid in target_dict]

        # For training, also guarantee any true match not captured in blocking is added (if in target)
        if is_training:
            for tm_id in true_matches:
                if tm_id in target_dict and tm_id not in pos_candidates:
                    pos_candidates.append(tm_id)

        # Subsample hard negatives for training to balance data
        if is_training and max_negatives_per_pos > 0:
            target_neg_count = max(3, len(pos_candidates) * max_negatives_per_pos)
            if len(neg_candidates) > target_neg_count:
                neg_candidates = rng.sample(neg_candidates, target_neg_count)

        # Process positives
        for cid in pos_candidates:
            feats = compute_pair_features(s1_rec, target_dict[cid])
            pair_list.append((s1_id, cid))
            features_list.append(feats)
            labels_list.append(1)
            pos_count += 1

        # Process hard negatives
        for cid in neg_candidates:
            feats = compute_pair_features(s1_rec, target_dict[cid])
            pair_list.append((s1_id, cid))
            features_list.append(feats)
            labels_list.append(0)
            neg_count += 1

    X = np.array(features_list, dtype=np.float32) if features_list else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(labels_list, dtype=np.int32) if labels_list else np.empty((0,))

    logger.info(
        "[DATASET] Built dataset: %d total pairs (%d positives, %d hard negatives, ratio 1:%.1f)",
        len(y), pos_count, neg_count, neg_count / max(1, pos_count),
    )
    return X, y, pair_list
