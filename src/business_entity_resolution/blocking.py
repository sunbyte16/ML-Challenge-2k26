"""
src/business_entity_resolution/blocking.py
==========================================
High-recall multi-pass blocking / candidate generation engine for
Amazon ML Challenge 2026 Business Entity Resolution.

Supports:
- Inverted index multi-pass blocking (Exact Name, Name Prefix, 2-Token Prefix,
  Address Prefix, Composite Name+Address, Country+Name)
- TF-IDF character n-gram retrieval for robust fuzzy recall
- Safe bucket size caps to prevent Cartesian explosion
- Candidate evaluation with recall ceiling and reduction ratio
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from business_entity_resolution.config import config

logger = logging.getLogger("BER.Blocking")


class MultiPassBlocker:
    """Multi-pass blocking candidate generator."""

    def __init__(
        self,
        max_candidates_per_s1: int = config.max_candidates_per_s1,
        max_bucket_size: int = config.max_block_bucket_size,
        use_tfidf_fallback: bool = config.use_tfidf_char_block,
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.max_bucket_size = max_bucket_size
        self.use_tfidf_fallback = use_tfidf_fallback

    def _build_inverted_index(self, df_target: pd.DataFrame, key_fn) -> Dict[str, List[str]]:
        """Build an inverted index mapping blocking_key -> list of target entity_ids."""
        index = defaultdict(list)
        for _, row in df_target.iterrows():
            entity_id = row["entity_id"]
            keys = key_fn(row)
            if isinstance(keys, str):
                keys = [keys]
            for k in keys:
                k_clean = str(k).strip()
                if k_clean and len(k_clean) >= 2:
                    index[k_clean].append(entity_id)
        return index

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Generate candidate target IDs (from S2 and S3) for each S1 entity.
        Returns: { s1_entity_id -> set of candidate entity_ids }
        """
        # Combine targets S2 and S3
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)
        logger.info(
            "[BLOCKING] Indexing %d target records (%d S2, %d S3) for %d S1 entities",
            len(df_target), len(df_s2), len(df_s3), len(df_s1),
        )

        candidates: Dict[str, Set[str]] = {row["entity_id"]: set() for _, row in df_s1.iterrows()}

        # -------------------------------------------------------------
        # PASS 1: Exact Normalized Business Name
        # -------------------------------------------------------------
        if config.use_exact_name_block:
            idx_name = self._build_inverted_index(df_target, lambda r: r["business_name_norm"])
            for _, row in df_s1.iterrows():
                k = row["business_name_norm"]
                if k in idx_name and len(idx_name[k]) <= self.max_bucket_size:
                    candidates[row["entity_id"]].update(idx_name[k])

        # -------------------------------------------------------------
        # PASS 2: Name First Token (min length 3)
        # -------------------------------------------------------------
        if config.use_name_prefix_block:
            idx_prefix1 = self._build_inverted_index(
                df_target,
                lambda r: r["name_first_token"] if len(r["name_first_token"]) >= 3 else ""
            )
            for _, row in df_s1.iterrows():
                k = row["name_first_token"]
                if k and k in idx_prefix1 and len(idx_prefix1[k]) <= self.max_bucket_size:
                    candidates[row["entity_id"]].update(idx_prefix1[k])

        # -------------------------------------------------------------
        # PASS 3: Name First Two Tokens
        # -------------------------------------------------------------
        if config.use_name_2prefix_block:
            idx_prefix2 = self._build_inverted_index(
                df_target,
                lambda r: r["name_two_tokens"] if len(r["name_two_tokens"]) >= 4 else ""
            )
            for _, row in df_s1.iterrows():
                k = row["name_two_tokens"]
                if k and k in idx_prefix2 and len(idx_prefix2[k]) <= self.max_bucket_size:
                    candidates[row["entity_id"]].update(idx_prefix2[k])

        # -------------------------------------------------------------
        # PASS 4: Name Prefix + Address Prefix Composite Key
        # -------------------------------------------------------------
        if config.use_name_address_composite_block:
            idx_comp = self._build_inverted_index(
                df_target,
                lambda r: f"{r['name_first_token']}|{r['address_first_token']}"
                if r["name_first_token"] and r["address_first_token"] else ""
            )
            for _, row in df_s1.iterrows():
                k = f"{row['name_first_token']}|{row['address_first_token']}"
                if k and k in idx_comp and len(idx_comp[k]) <= self.max_bucket_size:
                    candidates[row["entity_id"]].update(idx_comp[k])

        # -------------------------------------------------------------
        # PASS 5: Country + Name Prefix (useful for distinctive names)
        # -------------------------------------------------------------
        if config.use_country_name_block:
            idx_country_name = self._build_inverted_index(
                df_target,
                lambda r: f"{r['country_norm']}#{r['name_two_tokens']}"
                if r["country_norm"] and len(r["name_two_tokens"]) >= 4 else ""
            )
            for _, row in df_s1.iterrows():
                k = f"{row['country_norm']}#{row['name_two_tokens']}"
                if k and k in idx_country_name and len(idx_country_name[k]) <= self.max_bucket_size:
                    candidates[row["entity_id"]].update(idx_country_name[k])

        # -------------------------------------------------------------
        # PASS 6: TF-IDF Character n-gram fuzzy fallback (for entities with few candidates)
        # -------------------------------------------------------------
        if self.use_tfidf_fallback:
            low_candidate_s1 = [
                row for _, row in df_s1.iterrows()
                if len(candidates[row["entity_id"]]) < 5 and row["business_name_norm"]
            ]
            if low_candidate_s1 and len(df_target) > 0:
                logger.info(
                    "[BLOCKING] Running TF-IDF fallback for %d low-candidate entities",
                    len(low_candidate_s1),
                )
                try:
                    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=30000)
                    target_names = df_target["business_name_norm"].tolist()
                    X_target = vec.fit_transform(target_names)
                    s1_names = [r["business_name_norm"] for r in low_candidate_s1]
                    X_s1 = vec.transform(s1_names)

                    sims = cosine_similarity(X_s1, X_target)
                    target_ids = df_target["entity_id"].values

                    for i, r in enumerate(low_candidate_s1):
                        s1_id = r["entity_id"]
                        row_sims = sims[i]
                        # Top-k with threshold
                        top_indices = np.argsort(row_sims)[::-1][: config.tfidf_char_top_k]
                        for idx in top_indices:
                            if row_sims[idx] >= config.tfidf_char_min_sim:
                                candidates[s1_id].add(target_ids[idx])
                except Exception as e:
                    logger.warning("[BLOCKING] TF-IDF fallback failed: %s", e)

        # Enforce max candidates per entity cap
        for s1_id in candidates:
            if len(candidates[s1_id]) > self.max_candidates_per_s1:
                # Random/stable slice up to cap
                candidates[s1_id] = set(sorted(list(candidates[s1_id]))[: self.max_candidates_per_s1])

        total_pairs = sum(len(c) for c in candidates.values())
        avg_cands = total_pairs / max(1, len(df_s1))
        logger.info(
            "[BLOCKING] Completed candidate generation: %d total pairs across %d S1 entities (avg %.1f per S1)",
            total_pairs, len(df_s1), avg_cands,
        )
        return candidates

    @staticmethod
    def evaluate_blocking(
        candidates: Dict[str, Set[str]],
        gt_mapping: Dict[str, Set[str]],
        n_targets: int,
    ) -> dict:
        """
        Evaluate candidate recall against ground truth matches:
        - True match coverage (recall ceiling)
        - Average / median / max candidates per S1
        - Reduction ratio vs full Cartesian product
        """
        total_s1 = len(candidates)
        cand_counts = [len(cands) for cands in candidates.values()]
        total_pairs = sum(cand_counts)

        # Ground truth evaluation (scoped only to the entities being evaluated)
        relevant_gt = {s1_id: matches for s1_id, matches in gt_mapping.items() if s1_id in candidates}
        total_true_matches = sum(len(matches) for matches in relevant_gt.values())
        found_true_matches = 0
        lost_true_matches = 0

        for s1_id, true_matches in relevant_gt.items():
            cand_set = candidates.get(s1_id, set())
            for match_id in true_matches:
                if match_id in cand_set:
                    found_true_matches += 1
                else:
                    lost_true_matches += 1

        coverage = (found_true_matches / total_true_matches * 100) if total_true_matches > 0 else 100.0
        cartesian_size = total_s1 * n_targets
        reduction_ratio = ((1.0 - (total_pairs / cartesian_size)) * 100) if cartesian_size > 0 else 100.0

        metrics = {
            "total_s1_entities": total_s1,
            "total_candidate_pairs": total_pairs,
            "total_true_matches": total_true_matches,
            "found_true_matches": found_true_matches,
            "lost_true_matches": lost_true_matches,
            "blocking_recall_pct": round(coverage, 2),
            "avg_candidates_per_s1": round(np.mean(cand_counts), 2) if cand_counts else 0.0,
            "median_candidates_per_s1": float(np.median(cand_counts)) if cand_counts else 0.0,
            "max_candidates_per_s1": int(np.max(cand_counts)) if cand_counts else 0,
            "reduction_ratio_pct": round(reduction_ratio, 4),
        }

        print(f"\n{'='*70}\n[BLOCKING EVALUATION]\n{'='*70}")
        print(f"  Blocking Recall:          {metrics['blocking_recall_pct']:.2f}% ({found_true_matches:,}/{total_true_matches:,} true matches)")
        print(f"  Lost True Matches:        {lost_true_matches:,}")
        print(f"  Candidate Pairs:          {total_pairs:,}")
        print(f"  Avg Candidates / S1:      {metrics['avg_candidates_per_s1']}")
        print(f"  Median Candidates / S1:   {metrics['median_candidates_per_s1']}")
        print(f"  Max Candidates / S1:      {metrics['max_candidates_per_s1']}")
        print(f"  Reduction Ratio:          {metrics['reduction_ratio_pct']:.4f}% (Cartesian: {cartesian_size:,})")
        return metrics
