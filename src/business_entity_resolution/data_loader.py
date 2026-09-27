"""
src/business_entity_resolution/data_loader.py
=============================================
TSV data loader and profiling module for Amazon ML Challenge 2026.

All files are strictly tab-separated (.tsv).
Never use standard comma separation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Set

import pandas as pd
from business_entity_resolution.config import config

logger = logging.getLogger("BER.DataLoader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _read_tsv_safe(path: Path, expected_cols: list, label: str) -> pd.DataFrame:
    """Read a TSV file with tab separation and validation."""
    if not path.is_file():
        raise FileNotFoundError(f"[{label}] File not found at: {path}")

    # Read all columns as string to avoid type inference artifacts (e.g. zip codes as int)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns {missing}. Found: {list(df.columns)}")

    # Strip whitespace across columns
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    logger.info("[%s] Loaded %d rows and %d columns from %s", label, len(df), len(df.columns), path.name)
    return df


def load_source1(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or config.train_source1_path
    return _read_tsv_safe(p, SOURCE_COLUMNS, "Source 1")


def load_source2(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or config.train_source2_path
    return _read_tsv_safe(p, SOURCE_COLUMNS, "Source 2")


def load_source3(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or config.train_source3_path
    return _read_tsv_safe(p, SOURCE_COLUMNS, "Source 3")


def load_ground_truth(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or config.train_ground_truth_path
    return _read_tsv_safe(p, GROUND_TRUTH_COLUMNS, "Ground Truth")


def load_train_data(cfg=config) -> Dict[str, pd.DataFrame]:
    """Load all training datasets and ground truth."""
    return {
        "source1": load_source1(cfg.train_source1_path),
        "source2": load_source2(cfg.train_source2_path),
        "source3": load_source3(cfg.train_source3_path),
        "ground_truth": load_ground_truth(cfg.train_ground_truth_path),
    }


def load_test_data(cfg=config) -> Dict[str, pd.DataFrame]:
    """Load all test datasets."""
    return {
        "source1": load_source1(cfg.test_source1_path),
        "source2": load_source2(cfg.test_source2_path),
        "source3": load_source3(cfg.test_source3_path),
    }


def parse_ground_truth_mapping(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parse ground truth DataFrame into mapping:
    source1_entity_id -> set of matched_entity_ids (empty set for singletons).
    """
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        if raw:
            matches = {m.strip() for m in raw.split(",") if m.strip()}
        else:
            matches = set()
        gt_map[s1_id] = matches
    return gt_map


def profile_dataset(df: pd.DataFrame, label: str) -> dict:
    """Compute detailed profile metrics for a source dataset."""
    n_rows = len(df)
    n_cols = len(df.columns)
    dup_ids = int(df["entity_id"].duplicated().sum()) if "entity_id" in df.columns else 0
    empty_names = int((df["business_name"] == "").sum()) if "business_name" in df.columns else 0
    empty_addr = int((df["business_address"] == "").sum()) if "business_address" in df.columns else 0
    countries = df["country"].value_counts().to_dict() if "country" in df.columns else {}

    avg_name_len = float(df["business_name"].str.len().mean()) if "business_name" in df.columns and n_rows > 0 else 0.0
    avg_addr_len = float(df["business_address"].str.len().mean()) if "business_address" in df.columns and n_rows > 0 else 0.0

    stats = {
        "label": label,
        "rows": n_rows,
        "columns": list(df.columns),
        "duplicate_ids": dup_ids,
        "empty_names": empty_names,
        "empty_addresses": empty_addr,
        "countries": countries,
        "avg_name_length": round(avg_name_len, 2),
        "avg_address_length": round(avg_addr_len, 2),
    }

    print(f"\n{'='*70}\n[DATA PROFILE] {label}\n{'='*70}")
    print(f"  Rows:                 {n_rows:,}")
    print(f"  Columns:              {list(df.columns)}")
    print(f"  Duplicate Entity IDs: {dup_ids}")
    print(f"  Empty Names:          {empty_names} ({empty_names/max(1,n_rows):.1%})")
    print(f"  Empty Addresses:      {empty_addr} ({empty_addr/max(1,n_rows):.1%})")
    print(f"  Avg Name Length:      {avg_name_len:.1f} chars")
    print(f"  Avg Address Length:   {avg_addr_len:.1f} chars")
    print(f"  Countries:            {countries}")
    return stats


def profile_ground_truth(gt_df: pd.DataFrame) -> dict:
    """Analyze ground truth match counts, singletons, and multi-matches."""
    gt_map = parse_ground_truth_mapping(gt_df)
    match_counts = [len(matches) for matches in gt_map.values()]
    total_s1 = len(gt_map)
    total_matches = sum(match_counts)

    zero_matches = sum(1 for c in match_counts if c == 0)
    single_matches = sum(1 for c in match_counts if c == 1)
    multi_matches = sum(1 for c in match_counts if c > 1)

    stats = {
        "total_source1_entities": total_s1,
        "total_ground_truth_matches": total_matches,
        "zero_matches (singletons)": zero_matches,
        "zero_match_pct": round(zero_matches / max(1, total_s1) * 100, 2),
        "single_matches": single_matches,
        "single_match_pct": round(single_matches / max(1, total_s1) * 100, 2),
        "multi_matches": multi_matches,
        "multi_match_pct": round(multi_matches / max(1, total_s1) * 100, 2),
        "max_matches_for_single_s1": max(match_counts) if match_counts else 0,
    }

    print(f"\n{'='*70}\n[GROUND TRUTH PROFILE]\n{'='*70}")
    print(f"  Total Source 1 records:      {total_s1:,}")
    print(f"  Total true matches:          {total_matches:,}")
    print(f"  Zero-match (Singletons):     {zero_matches:,} ({stats['zero_match_pct']}%)")
    print(f"  Single-match (1:1):          {single_matches:,} ({stats['single_match_pct']}%)")
    print(f"  Multi-match (>1):            {multi_matches:,} ({stats['multi_match_pct']}%)")
    print(f"  Max matches for one S1:      {stats['max_matches_for_single_s1']}")
    return stats
