"""
src/business_entity_resolution/output.py
========================================
Official submission file writer for Amazon ML Challenge 2026.
Generates:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv
Guarantees tab separation, ordering, duplicate elimination, and subset rules.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Union

from business_entity_resolution.config import config

logger = logging.getLogger("BER.Output")


def write_matching_results(
    all_s1_ids: List[str],
    pred_matches: Dict[str, Union[Set[str], List[str]]],
    output_path: Path = config.matching_output_path,
) -> Path:
    """
    Write matching_results.tsv with columns:
    source1_entity_id\tmatched_entity_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = pred_matches.get(s1_id, [])
            # Deduplicate, keep order or sort deterministically
            if isinstance(matches, set):
                match_list = sorted(list(matches))
            else:
                seen = set()
                match_list = [x for x in matches if not (x in seen or seen.add(x))]
            
            # Filter strictly for S2 or S3 IDs
            valid_matches = [m for m in match_list if m.startswith("S2-") or m.startswith("S3-")]
            match_str = ",".join(valid_matches)
            f.write(f"{s1_id}\t{match_str}\n")

    logger.info("[OUTPUT] Written %d records to %s", len(all_s1_ids), output_path)
    return output_path


def write_candidate_pairs(
    all_s1_ids: List[str],
    candidates: Dict[str, Union[Set[str], List[str]]],
    output_path: Path = config.candidate_output_path,
) -> Path:
    """
    Write candidate_pairs.tsv with columns:
    source1_entity_id\tcandidate_entity_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, [])
            if isinstance(cands, set):
                cand_list = sorted(list(cands))
            else:
                seen = set()
                cand_list = [x for x in cands if not (x in seen or seen.add(x))]

            valid_cands = [c for c in cand_list if c.startswith("S2-") or c.startswith("S3-")]
            cand_str = ",".join(valid_cands)
            f.write(f"{s1_id}\t{cand_str}\n")

    logger.info("[OUTPUT] Written %d records to %s", len(all_s1_ids), output_path)
    return output_path
