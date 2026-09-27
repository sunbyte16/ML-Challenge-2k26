#!/usr/bin/env python3
"""
utils/package_submission.py
===========================
Packages the official Amazon ML Challenge 2026 submission zip archive:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── tests/
│       ├── utils/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Validates the submission with official validator before archiving.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge submission.")
    parser.add_argument("--team-name", default="my_team", help="Name of your team for the submission zip.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running the validator.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"
    test_dir = project_root / "dataset" / "test"
    doc_template = project_root / "Documentation_template.md"
    readme_path = project_root / "README.md"
    reqs_path = project_root / "requirements.txt"
    src_dir = project_root / "src"

    # 1. Validation check
    if not args.skip_validation:
        validator = project_root / "utils" / "validate_submission.py"
        print("[PACKAGER] Running official validation check...")
        cmd = [
            sys.executable,
            str(validator),
            "--matching", str(match_path),
            "--candidate", str(cand_path),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("[PACKAGER ERROR] Submission validator failed!")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)
        print("[PACKAGER] Validation PASSED.")

    # 2. Build Zip Archive
    submissions_dir = project_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    zip_path = submissions_dir / f"{args.team_name}_submission.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(match_path, arcname="output/matching_results.tsv")
        zf.write(cand_path, arcname="output/candidate_pairs.tsv")

        # Add documentation template
        if doc_template.is_file():
            zf.write(doc_template, arcname="Documentation_template.md")

        # Add code files
        if readme_path.is_file():
            zf.write(readme_path, arcname="code/business_entity_resolution/README.md")
        if reqs_path.is_file():
            zf.write(reqs_path, arcname="code/business_entity_resolution/requirements.txt")

        # Add src tree
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".yaml", ".json")):
                    file_p = Path(root) / file
                    rel = file_p.relative_to(project_root)
                    zf.write(file_p, arcname=f"code/business_entity_resolution/{rel.as_posix()}")

        # Add tests tree
        tests_dir = project_root / "tests"
        if tests_dir.is_dir():
            for root, _, files in os.walk(tests_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_p = Path(root) / file
                        rel = file_p.relative_to(project_root)
                        zf.write(file_p, arcname=f"code/business_entity_resolution/{rel.as_posix()}")

        # Add utils tree
        utils_dir = project_root / "utils"
        if utils_dir.is_dir():
            for root, _, files in os.walk(utils_dir):
                for file in files:
                    if file.endswith(".py"):
                        file_p = Path(root) / file
                        rel = file_p.relative_to(project_root)
                        zf.write(file_p, arcname=f"code/business_entity_resolution/{rel.as_posix()}")

    print(f"\n[PACKAGER] Submission archive successfully created:")
    print(f"  -> {zip_path}")
    print(f"  Size: {zip_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
