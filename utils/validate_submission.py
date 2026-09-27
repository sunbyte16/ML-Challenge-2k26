#!/usr/bin/env python3
"""
Official Submission Validator for Amazon ML Challenge 2026.
Checks matching_results.tsv and candidate_pairs.tsv against official challenge rules:
1. Files exist and are tab-separated (.tsv).
2. Exactly matching header columns:
   - matching:  source1_entity_id\tmatched_entity_ids
   - candidate: source1_entity_id\tcandidate_entity_ids
3. Exactly one row per Source 1 entity in the test set.
4. No duplicate source1_entity_id rows.
5. All matched/candidate IDs must belong to Source 2 or Source 3 (start with S2- or S3-).
6. No Source 1 IDs (S1-) in the match or candidate lists.
7. No duplicate IDs within any single comma-separated list.
8. All referenced S2 and S3 IDs must actually exist in the test dataset.
9. Every ID in matching_results.tsv must appear in candidate_pairs.tsv (matches ⊆ candidates).

Exit code:
  0: PASS
  1: Issues found (numbered list printed)
"""

import sys
import argparse
from pathlib import Path


def parse_tsv_ids(path, id_col, list_col):
    """Parse TSV into dict of s1_id -> list of target_ids."""
    if not path.is_file():
        return None, [f"File not found: {path}"]

    errors = []
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return None, [f"{path.name} is empty."]

        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        expected_header = [id_col, list_col]
        if header != expected_header:
            errors.append(
                f"{path.name}: Expected header {expected_header}, found {header}. Ensure tab-separated."
            )

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                s1_id = parts[0].strip()
                target_str = ""
            elif len(parts) == 2:
                s1_id = parts[0].strip()
                target_str = parts[1].strip()
            else:
                errors.append(f"{path.name} Line {line_num}: More than 2 tab-separated columns found.")
                continue

            if s1_id in data:
                errors.append(f"{path.name} Line {line_num}: Duplicate source1_entity_id '{s1_id}'.")
            
            if target_str:
                raw_ids = [m.strip() for m in target_str.split(",")]
                # Check within-row duplicates
                seen = set()
                dups = []
                for x in raw_ids:
                    if not x:
                        errors.append(f"{path.name} Line {line_num}: Empty ID between commas.")
                    if x in seen:
                        dups.append(x)
                    seen.add(x)
                if dups:
                    errors.append(
                        f"{path.name} Line {line_num}: Duplicate IDs within list: {list(set(dups))} for {s1_id}."
                    )
                data[s1_id] = raw_ids
            else:
                data[s1_id] = []

    return data, errors


def load_test_entity_ids(test_dir):
    """Load valid test entity IDs from test_source1, test_source2, test_source3."""
    test_dir = Path(test_dir)
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.is_file():
            return None, None, None, f"Required test file not found: {p}"

    def extract_ids(tsv_path):
        ids = set()
        with open(tsv_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
            try:
                id_idx = header.index("entity_id")
            except ValueError:
                return None, f"{tsv_path.name}: 'entity_id' column not found in {header}"
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) > id_idx:
                    ids.add(parts[id_idx].strip())
        return ids, None

    s1_ids, err1 = extract_ids(s1_path)
    if err1:
        return None, None, None, err1
    s2_ids, err2 = extract_ids(s2_path)
    if err2:
        return None, None, None, err2
    s3_ids, err3 = extract_ids(s3_path)
    if err3:
        return None, None, None, err3

    return s1_ids, s2_ids, s3_ids, None


def main():
    parser = argparse.ArgumentParser(description="Validate Amazon ML Challenge 2026 submission files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    matching_path = Path(args.matching)
    candidate_path = Path(args.candidate)
    test_dir = Path(args.test_dir)

    all_issues = []

    # 1. Load test entity IDs
    s1_ids, s2_ids, s3_ids, test_err = load_test_entity_ids(test_dir)
    if test_err:
        print(f"[VALIDATOR ERROR] {test_err}")
        sys.exit(1)

    valid_target_ids = s2_ids | s3_ids

    # 2. Parse matching_results.tsv
    matching_data, match_errors = parse_tsv_ids(matching_path, "source1_entity_id", "matched_entity_ids")
    all_issues.extend(match_errors)

    # 3. Parse candidate_pairs.tsv
    candidate_data, cand_errors = parse_tsv_ids(candidate_path, "source1_entity_id", "candidate_entity_ids")
    all_issues.extend(cand_errors)

    if matching_data is None or candidate_data is None:
        print("\n".join(f"{i+1}. {iss}" for i, iss in enumerate(all_issues)))
        sys.exit(1)

    # 4. Check that every test S1 entity has exactly one row
    missing_in_matching = s1_ids - set(matching_data.keys())
    if missing_in_matching:
        all_issues.append(f"matching_results.tsv is missing {len(missing_in_matching)} test S1 entities.")

    extra_in_matching = set(matching_data.keys()) - s1_ids
    if extra_in_matching:
        all_issues.append(f"matching_results.tsv contains {len(extra_in_matching)} unknown S1 entities.")

    missing_in_candidate = s1_ids - set(candidate_data.keys())
    if missing_in_candidate:
        all_issues.append(f"candidate_pairs.tsv is missing {len(missing_in_candidate)} test S1 entities.")

    extra_in_candidate = set(candidate_data.keys()) - s1_ids
    if extra_in_candidate:
        all_issues.append(f"candidate_pairs.tsv contains {len(extra_in_candidate)} unknown S1 entities.")

    # 5. Check target IDs validity and subset rule
    invalid_matched_ids = set()
    s1_in_matches = set()
    not_in_candidates = []

    for s1_id, match_list in matching_data.items():
        cand_list = set(candidate_data.get(s1_id, []))
        for m_id in match_list:
            if m_id.startswith("S1-"):
                s1_in_matches.add(m_id)
            if m_id not in valid_target_ids:
                invalid_matched_ids.add(m_id)
            if m_id not in cand_list:
                not_in_candidates.append((s1_id, m_id))

    if s1_in_matches:
        all_issues.append(f"matching_results.tsv contains Source 1 IDs in matched_entity_ids: {list(s1_in_matches)[:5]}")
    if invalid_matched_ids:
        all_issues.append(f"matching_results.tsv contains invalid/non-existent test IDs: {list(invalid_matched_ids)[:5]}")
    if not_in_candidates:
        all_issues.append(f"matching_results.tsv has {len(not_in_candidates)} matches NOT in candidate_pairs.tsv. First: {not_in_candidates[0]}")

    invalid_cand_ids = set()
    s1_in_cands = set()
    for s1_id, cand_list in candidate_data.items():
        for c_id in cand_list:
            if c_id.startswith("S1-"):
                s1_in_cands.add(c_id)
            if c_id not in valid_target_ids:
                invalid_cand_ids.add(c_id)

    if s1_in_cands:
        all_issues.append(f"candidate_pairs.tsv contains Source 1 IDs in candidate_entity_ids: {list(s1_in_cands)[:5]}")
    if invalid_cand_ids:
        all_issues.append(f"candidate_pairs.tsv contains invalid/non-existent test IDs: {list(invalid_cand_ids)[:5]}")

    if all_issues:
        print(f"FAILED: Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues[:20], 1):
            print(f"  {i}. {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more.")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
