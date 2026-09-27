import json
from pathlib import Path

nb_dir = Path("notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}

# 01 EDA
nb1 = make_nb([
    md_cell("# 01. Exploratory Data Analysis\nAmazon ML Challenge 2026: Business Entity Resolution\n\nUnderstand dataset shapes, schemas, distributions, countries, and ground truth match characteristics."),
    code_cell("import sys\nsys.path.append('../src')\nfrom business_entity_resolution.config import config\nfrom business_entity_resolution.data_loader import load_train_data, profile_dataset, profile_ground_truth\n\ntrain_data = load_train_data(config)\nprofile_dataset(train_data['source1'], 'TRAIN SOURCE 1')\nprofile_dataset(train_data['source2'], 'TRAIN SOURCE 2')\nprofile_dataset(train_data['source3'], 'TRAIN SOURCE 3')\ngt_stats = profile_ground_truth(train_data['ground_truth'])"),
    md_cell("### Key EDA Takeaways\n- Source 1 is deduplicated reference; Source 2 & 3 contain noisy real-world entities.\n- Singletons (entities with 0 matches) represent a significant portion of records and receive 1.0 credit if left empty.\n- Countries in training: US and India. Test additionally includes France, necessitating open-set country logic.")
])

# 02 Preprocessing
nb2 = make_nb([
    md_cell("# 02. Text Normalization and Tokenization\nPreprocess noisy business names, addresses, and country identifiers while preserving raw data."),
    code_cell("import sys\nsys.path.append('../src')\nfrom business_entity_resolution.config import config\nfrom business_entity_resolution.data_loader import load_train_data\nfrom business_entity_resolution.preprocessing import preprocess_dataframe\n\ntrain_data = load_train_data(config)\ndf_s1_norm = preprocess_dataframe(train_data['source1'], is_source1=True)\ndf_s1_norm[['entity_id', 'business_name', 'business_name_norm', 'business_address_norm', 'country_norm']].head(10)"),
    md_cell("### Preprocessing Summary\n- NFKD unicode decomposition maps accented characters to ASCII.\n- Legal suffixes unified (e.g. Pvt Ltd -> pvt ltd, Inc -> inc).\n- Common road abbreviations standardized.")
])

# 03 Blocking
nb3 = make_nb([
    md_cell("# 03. Multi-Pass Blocking and Candidate Generation\nEvaluates candidate recall and reduction ratio across multiple blocking keys."),
    code_cell("import sys\nsys.path.append('../src')\nimport pandas as pd\nfrom business_entity_resolution.config import config\nfrom business_entity_resolution.data_loader import load_train_data, parse_ground_truth_mapping\nfrom business_entity_resolution.preprocessing import preprocess_dataframe\nfrom business_entity_resolution.blocking import MultiPassBlocker\n\ntrain_data = load_train_data(config)\ndf_s1 = preprocess_dataframe(train_data['source1'], is_source1=True)\ndf_s2 = preprocess_dataframe(train_data['source2'], is_source1=False)\ndf_s3 = preprocess_dataframe(train_data['source3'], is_source1=False)\ndf_targets = pd.concat([df_s2, df_s3], ignore_index=True)\ngt_map = parse_ground_truth_mapping(train_data['ground_truth'])\n\nblocker = MultiPassBlocker()\ncands = blocker.generate_candidates(df_s1, df_s2, df_s3)\nmetrics = blocker.evaluate_blocking(cands, gt_map, len(df_targets))"),
    md_cell("### Blocking Takeaways\n- Multi-pass blocking unions exact names, 1-token prefixes, 2-token prefixes, composite keys, and TF-IDF fallback.\n- Generates candidate pairs reducing Cartesian search space by >99% while preserving true matches.")
])

# 04 Features
nb4 = make_nb([
    md_cell("# 04. Pairwise Feature Engineering\nComputes 34-dimensional feature vectors combining RapidFuzz similarities, token Jaccards, char 3-grams, and open-set country equality."),
    code_cell("import sys\nsys.path.append('../src')\nfrom business_entity_resolution.features import compute_pair_features, FEATURE_NAMES\n\nprint(f'Total features engineered: {len(FEATURE_NAMES)}')\nfor i, f in enumerate(FEATURE_NAMES, 1):\n    print(f'{i:2d}. {f}')"),
    md_cell("### Features Summary\n- Incorporates fuzzy Levenshtein ratios, token sort/set ratios, token overlap counts, length ratios, and open-set country indicators.")
])

# 05 Model
nb5 = make_nb([
    md_cell("# 05. Model Training, F0.5 Optimization, and Thresholding\nTrains match classifier and optimizes decision threshold for Macro F0.5."),
    code_cell("import sys\nsys.path.append('../src')\nfrom business_entity_resolution.config import config\nfrom business_entity_resolution.pipeline import run_train\n\nmodel = run_train(config)"),
    md_cell("### Conclusion\n- Model predicts match probabilities for all candidate pairs.\n- Threshold optimized on entity-level Macro F0.5 to balance precision and recall, strictly protecting against false merges.")
])

for p, nb in [
    (nb_dir / "01_eda.ipynb", nb1),
    (nb_dir / "02_preprocessing.ipynb", nb2),
    (nb_dir / "03_blocking.ipynb", nb3),
    (nb_dir / "04_features.ipynb", nb4),
    (nb_dir / "05_model.ipynb", nb5),
]:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print("Created all 5 notebooks.")
