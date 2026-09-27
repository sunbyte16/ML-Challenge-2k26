"""
src/business_entity_resolution/preprocessing.py
===============================================
Text normalization and tokenization pipeline for Amazon ML Challenge 2026.
Preserves original columns while creating standardized normalized fields.
"""

import re
import unicodedata
from typing import List, Set, Tuple
import pandas as pd


# Common abbreviations mapping for addresses
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bext\b": "extension",
    r"\bhwy\b": "highway",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bctr\b": "center",
    r"\bpo\s*box\b": "pobox",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}

# Country variant normalization (Open-set: unknown countries pass through unchanged)
COUNTRY_MAP = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "india",
    "ind": "india",
    "india": "india",
    "bharat": "india",
    "fr": "france",
    "fra": "france",
    "france": "france",
}

# Legal suffixes to normalize or unify
LEGAL_SUFFIXES_REGEX = [
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\s*\.?\s*ltd\s*\.?\b", "pvt ltd"),
    (r"\bltd\s*\.?\b", "ltd"),
    (r"\blimited\b", "ltd"),
    (r"\bincorporated\b", "inc"),
    (r"\binc\s*\.?\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\s*\.?\b", "corp"),
    (r"\bllc\s*\.?\b", "llc"),
    (r"\bco\s*\.?\b", "co"),
    (r"\bcompany\b", "co"),
]


def normalize_unicode_ascii(text: str) -> str:
    """Normalize unicode characters to ASCII compatibility (NFKD)."""
    if not text:
        return ""
    text = str(text)
    # Decompose unicode, strip combining diacritics
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ASCII", "ignore").decode("utf-8")
    return ascii_text


def normalize_text_general(text: str) -> str:
    """General text normalization: lower, unicode, separators, whitespace."""
    if not text:
        return ""
    t = normalize_unicode_ascii(text).lower()
    # Normalize common symbols
    t = t.replace("&", " and ")
    t = t.replace("@", " at ")
    t = t.replace("/", " ")
    t = t.replace("-", " ")
    t = t.replace("_", " ")
    # Replace non-alphanumeric with spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # Collapse multiple whitespaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_business_name(name: str, unify_legal_suffixes: bool = True) -> str:
    """Normalize business name preserving key tokens and optionally unifying suffixes."""
    norm = normalize_text_general(name)
    if unify_legal_suffixes:
        for pattern, repl in LEGAL_SUFFIXES_REGEX:
            norm = re.sub(pattern, repl, norm)
        norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def normalize_business_address(address: str, expand_abbreviations: bool = True) -> str:
    """Normalize business address preserving numbers, street names, and postal codes."""
    norm = normalize_text_general(address)
    if expand_abbreviations:
        for pattern, repl in ADDRESS_ABBREVIATIONS.items():
            norm = re.sub(pattern, repl, norm)
        norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def normalize_country(country: str) -> str:
    """Open-set country normalization: map known synonyms, pass others through."""
    if not country:
        return ""
    norm = normalize_text_general(country)
    return COUNTRY_MAP.get(norm, norm)


def tokenize_whitespace(text: str) -> List[str]:
    """Tokenize by whitespace into non-empty tokens."""
    return [tok for tok in text.split() if tok]


def tokenize_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from normalized text."""
    clean = re.sub(r"\s+", "", text)
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i : i + n] for i in range(len(clean) - n + 1)]


def get_token_prefix(text: str, num_tokens: int = 1) -> str:
    """Get the first N tokens joined by space."""
    toks = tokenize_whitespace(text)
    return " ".join(toks[:num_tokens]) if toks else ""


def preprocess_dataframe(df: pd.DataFrame, is_source1: bool = False) -> pd.DataFrame:
    """
    Process raw source dataframe into standardized format with normalized columns.
    Preserves original columns: business_name_original, business_address_original.
    Adds: business_name_norm, business_address_norm, country_norm,
          name_first_token, name_two_tokens, address_first_token.
    """
    df_out = df.copy()

    # Preserve raw data
    if "business_name_original" not in df_out.columns:
        df_out["business_name_original"] = df_out["business_name"].astype(str)
    if "business_address_original" not in df_out.columns:
        df_out["business_address_original"] = df_out["business_address"].astype(str)

    # Normalize fields
    df_out["business_name_norm"] = df_out["business_name_original"].apply(normalize_business_name)
    df_out["business_address_norm"] = df_out["business_address_original"].apply(normalize_business_address)
    df_out["country_norm"] = df_out["country"].apply(normalize_country)

    # Blocking keys
    df_out["name_first_token"] = df_out["business_name_norm"].apply(lambda s: get_token_prefix(s, 1))
    df_out["name_two_tokens"] = df_out["business_name_norm"].apply(lambda s: get_token_prefix(s, 2))
    df_out["address_first_token"] = df_out["business_address_norm"].apply(lambda s: get_token_prefix(s, 1))

    return df_out
