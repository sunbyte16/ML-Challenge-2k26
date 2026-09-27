"""
src/business_entity_resolution/features.py
==========================================
Pairwise feature engineering for candidate pairs (S1, Target).
Computes rapid string similarity metrics, token statistics,
open-set country alignment, and cross-field signals.
"""

from typing import Dict, List, Any
import numpy as np
from rapidfuzz import fuzz


FEATURE_NAMES = [
    # --- Name similarities ---
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_length_diff",
    "name_length_ratio",
    "name_token_count_diff",
    "name_exact_match",
    "name_contains_other",
    "same_first_name_token",
    "shared_name_token_count",
    "shared_name_token_ratio",
    "name_char3_jaccard",

    # --- Address similarities ---
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_jaccard",
    "address_length_diff",
    "address_length_ratio",
    "address_token_count_diff",
    "address_exact_match",
    "address_contains_other",
    "same_first_address_token",
    "shared_address_token_count",
    "shared_address_token_ratio",
    "address_char3_jaccard",

    # --- Country features (open-set safe) ---
    "country_exact_match",
    "country_missing",

    # --- Source / Cross features ---
    "both_name_present",
    "both_address_present",
    "target_is_s2",
    "target_is_s3",
]


def _jaccard_tokens(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def _char_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if len(text1) < n or len(text2) < n:
        return 1.0 if text1 == text2 and text1 != "" else 0.0
    grams1 = {text1[i : i + n] for i in range(len(text1) - n + 1)}
    grams2 = {text2[i : i + n] for i in range(len(text2) - n + 1)}
    union = len(grams1 | grams2)
    return (len(grams1 & grams2) / union) if union > 0 else 0.0


def compute_pair_features(s1_rec: dict, target_rec: dict) -> List[float]:
    """
    Compute pairwise features between a Source 1 record and a Target record.
    Returns a 1D float list corresponding to FEATURE_NAMES.
    """
    # Name fields
    name1 = s1_rec.get("business_name_norm", "")
    name2 = target_rec.get("business_name_norm", "")

    toks_n1 = name1.split()
    toks_n2 = name2.split()
    len_n1, len_n2 = len(name1), len(name2)

    # Name features
    if name1 and name2:
        name_ratio = fuzz.ratio(name1, name2) / 100.0
        name_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
        name_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
        name_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0
        name_jaccard = _jaccard_tokens(toks_n1, toks_n2)
        name_len_diff = abs(len_n1 - len_n2)
        name_len_ratio = min(len_n1, len_n2) / max(1, max(len_n1, len_n2))
        name_tok_diff = abs(len(toks_n1) - len(toks_n2))
        name_exact = 1.0 if name1 == name2 else 0.0
        name_contains = 1.0 if (name1 in name2 or name2 in name1) else 0.0
        same_first_n = 1.0 if (toks_n1 and toks_n2 and toks_n1[0] == toks_n2[0]) else 0.0
        shared_n_cnt = len(set(toks_n1) & set(toks_n2))
        shared_n_ratio = shared_n_cnt / max(1, max(len(toks_n1), len(toks_n2)))
        name_c3_jaccard = _char_ngram_jaccard(name1, name2, 3)
    else:
        name_ratio = name_partial_ratio = name_token_sort_ratio = name_token_set_ratio = 0.0
        name_jaccard = 0.0
        name_len_diff = abs(len_n1 - len_n2)
        name_len_ratio = 0.0
        name_tok_diff = abs(len(toks_n1) - len(toks_n2))
        name_exact = name_contains = same_first_n = 0.0
        shared_n_cnt = shared_n_ratio = name_c3_jaccard = 0.0

    # Address fields
    addr1 = s1_rec.get("business_address_norm", "")
    addr2 = target_rec.get("business_address_norm", "")
    toks_a1 = addr1.split()
    toks_a2 = addr2.split()
    len_a1, len_a2 = len(addr1), len(addr2)

    # Address features
    if addr1 and addr2:
        addr_ratio = fuzz.ratio(addr1, addr2) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(addr1, addr2) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(addr1, addr2) / 100.0
        addr_jaccard = _jaccard_tokens(toks_a1, toks_a2)
        addr_len_diff = abs(len_a1 - len_a2)
        addr_len_ratio = min(len_a1, len_a2) / max(1, max(len_a1, len_a2))
        addr_tok_diff = abs(len(toks_a1) - len(toks_a2))
        addr_exact = 1.0 if addr1 == addr2 else 0.0
        addr_contains = 1.0 if (addr1 in addr2 or addr2 in addr1) else 0.0
        same_first_a = 1.0 if (toks_a1 and toks_a2 and toks_a1[0] == toks_a2[0]) else 0.0
        shared_a_cnt = len(set(toks_a1) & set(toks_a2))
        shared_a_ratio = shared_a_cnt / max(1, max(len(toks_a1), len(toks_a2)))
        addr_c3_jaccard = _char_ngram_jaccard(addr1, addr2, 3)
    else:
        addr_ratio = addr_partial_ratio = addr_token_sort_ratio = addr_token_set_ratio = 0.0
        addr_jaccard = 0.0
        addr_len_diff = abs(len_a1 - len_a2)
        addr_len_ratio = 0.0
        addr_tok_diff = abs(len(toks_a1) - len(toks_a2))
        addr_exact = addr_contains = same_first_a = 0.0
        shared_a_cnt = shared_a_ratio = addr_c3_jaccard = 0.0

    # Country features (Open-set string comparison)
    c1 = s1_rec.get("country_norm", "").strip()
    c2 = target_rec.get("country_norm", "").strip()
    country_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    country_missing = 1.0 if (not c1 or not c2) else 0.0

    # Presence and source metadata
    both_names = 1.0 if (name1 and name2) else 0.0
    both_addrs = 1.0 if (addr1 and addr2) else 0.0

    target_id = target_rec.get("entity_id", "")
    target_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    target_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_jaccard,
        float(name_len_diff),
        name_len_ratio,
        float(name_tok_diff),
        name_exact,
        name_contains,
        same_first_n,
        float(shared_n_cnt),
        shared_n_ratio,
        name_c3_jaccard,
        addr_ratio,
        addr_partial_ratio,
        addr_token_sort_ratio,
        addr_token_set_ratio,
        addr_jaccard,
        float(addr_len_diff),
        addr_len_ratio,
        float(addr_tok_diff),
        addr_exact,
        addr_contains,
        same_first_a,
        float(shared_a_cnt),
        shared_a_ratio,
        addr_c3_jaccard,
        country_exact,
        country_missing,
        both_names,
        both_addrs,
        target_s2,
        target_s3,
    ]
