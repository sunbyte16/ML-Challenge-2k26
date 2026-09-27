"""
Unit and integration tests for Business Entity Resolution pipeline.
Covers preprocessing, blocking, features, metrics, and validator compliance.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from business_entity_resolution.preprocessing import (
    normalize_text_general,
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    preprocess_dataframe,
)
from business_entity_resolution.blocking import MultiPassBlocker
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.metrics import compute_single_entity_f05, evaluate_entity_macro_f05
from business_entity_resolution.output import write_matching_results, write_candidate_pairs
from utils.validate_submission import parse_tsv_ids


# ---------------------------------------------------------------------------
# 1. Normalization Tests
# ---------------------------------------------------------------------------
def test_normalization_null_and_empty():
    assert normalize_text_general("") == ""
    assert normalize_text_general(None) == ""
    assert normalize_business_name("") == ""
    assert normalize_business_address("") == ""
    assert normalize_country("") == ""


def test_normalization_unicode_and_case():
    raw = "CAFÉ & BISTRO, INC."
    norm = normalize_business_name(raw)
    assert "cafe" in norm
    assert "and" in norm
    assert "bistro" in norm
    assert "inc" in norm


def test_country_open_set():
    # Known mapped
    assert normalize_country("United States") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("France") == "france"
    # Unknown open-set passed through clean
    assert normalize_country("Germany") == "germany"
    assert normalize_country("Japan") == "japan"


# ---------------------------------------------------------------------------
# 2. Blocking Tests
# ---------------------------------------------------------------------------
def test_blocking_no_s1_in_candidates():
    df_s1 = pd.DataFrame([{
        "entity_id": "S1-001",
        "business_name_norm": "amazon web services",
        "business_address_norm": "seattle washington",
        "country_norm": "us",
        "name_first_token": "amazon",
        "name_two_tokens": "amazon web",
        "address_first_token": "seattle",
    }])

    df_s2 = pd.DataFrame([{
        "entity_id": "S2-001",
        "business_name_norm": "amazon web services inc",
        "business_address_norm": "seattle wa",
        "country_norm": "us",
        "name_first_token": "amazon",
        "name_two_tokens": "amazon web",
        "address_first_token": "seattle",
    }])

    df_s3 = pd.DataFrame([{
        "entity_id": "S3-001",
        "business_name_norm": "amazon aws",
        "business_address_norm": "seattle wa",
        "country_norm": "us",
        "name_first_token": "amazon",
        "name_two_tokens": "amazon aws",
        "address_first_token": "seattle",
    }])

    blocker = MultiPassBlocker(use_tfidf_fallback=False)
    cands = blocker.generate_candidates(df_s1, df_s2, df_s3)

    assert "S1-001" in cands
    cand_ids = cands["S1-001"]
    # Only S2 and S3, never S1
    for cid in cand_ids:
        assert cid.startswith("S2-") or cid.startswith("S3-")
        assert not cid.startswith("S1-")
    assert "S2-001" in cand_ids


# ---------------------------------------------------------------------------
# 3. Pairwise Features Tests
# ---------------------------------------------------------------------------
def test_features_identical_vs_different():
    rec1 = {
        "entity_id": "S1-001",
        "business_name_norm": "walmart supercenter",
        "business_address_norm": "100 main street bentonville arkansas",
        "country_norm": "us",
    }
    rec2 = {
        "entity_id": "S2-001",
        "business_name_norm": "walmart supercenter",
        "business_address_norm": "100 main street bentonville arkansas",
        "country_norm": "us",
    }
    rec3 = {
        "entity_id": "S2-002",
        "business_name_norm": "starbucks coffee",
        "business_address_norm": "pike place market seattle",
        "country_norm": "us",
    }

    feats_match = compute_pair_features(rec1, rec2)
    feats_diff = compute_pair_features(rec1, rec3)

    assert len(feats_match) == len(FEATURE_NAMES)
    # Name ratio and address ratio should be 1.0 for exact matches
    assert feats_match[0] == 1.0  # name_ratio
    assert feats_match[14] == 1.0  # address_ratio
    # Different records should have substantially lower similarity
    assert feats_diff[0] < 0.5


# ---------------------------------------------------------------------------
# 4. Metric Tests (Official Entity-level Macro F0.5)
# ---------------------------------------------------------------------------
def test_metrics_singleton_full_credit():
    # True is empty, pred is empty -> full credit (1.0)
    p, r, f05 = compute_single_entity_f05(set(), set())
    assert f05 == 1.0
    assert p == 1.0
    assert r == 1.0


def test_metrics_singleton_false_merge_penalty():
    # True is empty, pred has spurious match -> zero score (0.0)
    p, r, f05 = compute_single_entity_f05(set(), {"S2-999"})
    assert f05 == 0.0


def test_metrics_perfect_prediction():
    # 2 true matches, both correctly predicted -> 1.0
    p, r, f05 = compute_single_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002"})
    assert f05 == 1.0
    assert p == 1.0
    assert r == 1.0


def test_metrics_partial_prediction():
    # True: {A, B}, Pred: {A, C} -> TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5 -> F0.5=0.5
    p, r, f05 = compute_single_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S2-999"})
    assert p == 0.5
    assert r == 0.5
    assert round(f05, 4) == 0.5


def test_metrics_macro_average():
    gt = {
        "S1-1": set(),            # singleton
        "S1-2": {"S2-1"},         # single match
    }
    pred = {
        "S1-1": set(),            # correct singleton (1.0)
        "S1-2": {"S2-1"},         # correct match (1.0)
    }
    res = evaluate_entity_macro_f05(gt, pred)
    assert res["macro_f05"] == 1.0
    assert res["singleton_accuracy_pct"] == 100.0


# ---------------------------------------------------------------------------
# 5. Output Format Compliance Tests
# ---------------------------------------------------------------------------
def test_output_writer_and_subset_rule(tmp_path):
    all_s1 = ["S1-001", "S1-002", "S1-003"]
    candidates = {
        "S1-001": {"S2-10", "S3-20"},
        "S1-002": {"S3-30"},
        "S1-003": set(),
    }
    matches = {
        "S1-001": {"S2-10"},
        "S1-002": {"S3-30"},
        "S1-003": set(),
    }

    match_p = tmp_path / "matching_results.tsv"
    cand_p = tmp_path / "candidate_pairs.tsv"

    write_matching_results(all_s1, matches, match_p)
    write_candidate_pairs(all_s1, candidates, cand_p)

    m_data, m_errs = parse_tsv_ids(match_p, "source1_entity_id", "matched_entity_ids")
    c_data, c_errs = parse_tsv_ids(cand_p, "source1_entity_id", "candidate_entity_ids")

    assert len(m_errs) == 0
    assert len(c_errs) == 0
    assert len(m_data) == 3
    assert len(c_data) == 3

    # Check subset rule: matches ⊆ candidates
    for s1_id in all_s1:
        assert set(m_data[s1_id]).issubset(set(c_data[s1_id]))
