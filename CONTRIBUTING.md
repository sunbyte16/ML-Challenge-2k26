# Contributing to Amazon ML Challenge 2026: Business Entity Resolution

First off, thank you for considering contributing to this repository! Contributions from developers, researchers, and competitive ML practitioners are welcome.

Please take a moment to review this document to ensure a smooth and effective collaboration.

---

## 1. Code of Conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## 2. Hard Competition Constraints

If you are contributing code for submission to the official challenge, your code must adhere to these inviolable constraints:

1. **Strictly No External Data**: No external APIs (Google Maps, Places, OpenStreetMap), no web scraping, no commercial entity resolution APIs, and no external entity lookup dictionaries. All features must be derived purely from the provided dataset.
2. **Open-Set Country Handling**: Country is an open-set string label (e.g. France appears in test data while only US and India appear in train). Do not hardcode, filter, or one-hot encode country features to only `{US, India}`.
3. **Tab-Separated Formats**: All input and output files are `.tsv` (separated by `\t`). Never parse or write comma-separated formats for dataset or submission files.
4. **Official Evaluation Metric**: All models, blocking pipelines, and thresholds must be evaluated against **Entity-Level Macro $F_{0.5}$** with singleton credit (1.0 if empty predicted for empty true sets; 0.0 for false merges).
5. **No Source 1 Self-Matching**: Blocking must only generate candidates from Source 2 and Source 3 (never $S_1 \to S_1$).

---

## 3. Development Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.10, 3.11, 3.12)
* Git

### Step-by-Step Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/amazon-ml-challenge-2026.git
   cd amazon-ml-challenge-2026
   ```

2. **Create and activate a virtual environment**:
   * On Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

---

## 4. Repository Structure

```text
├── dataset/                    # Local datasets (train & test TSVs)
├── src/
│   └── business_entity_resolution/
│       ├── config.py           # Centralized configuration & hyperparameters
│       ├── data_loader.py      # Tab-separated dataset loading & profiling
│       ├── preprocessing.py    # Unicode normalization, legal suffix & address rules
│       ├── blocking.py         # Multi-pass inverted index blocking & TF-IDF
│       ├── features.py         # RapidFuzz string similarities & open-set country features
│       ├── dataset_builder.py  # Entity-aware split & hard negative mining
│       ├── model.py            # Matching classifiers (HistGradientBoosting / LightGBM)
│       ├── metrics.py          # Official Macro F0.5 calculation with singleton rules
│       ├── threshold.py        # F0.5-optimized decision threshold grid search
│       ├── predict.py          # Test candidate generation & inference
│       ├── output.py           # TSV output file generation
│       └── pipeline.py         # Master CLI runner
├── tests/                      # Unit and integration test suite
├── utils/
│   ├── validate_submission.py  # Official competition validator
│   ├── package_submission.py   # Automated submission zip packaging
│   └── generate_mock_dataset.py# Synthetic dataset generator for smoke testing
├── notebooks/                  # Step-by-step Jupyter notebooks (01 to 05)
├── requirements.txt            # Pinned dependencies
├── pyproject.toml              # Build & packaging metadata
├── README.md                   # Project documentation
├── Documentation_template.md   # Official competition methodology document
└── LICENSE                     # Open source license
```

---

## 5. Development Workflow & Git Guidelines

### Branch Naming Conventions
* `feature/<feature-name>`: New feature or blocking pass (e.g., `feature/phonetic-blocking`).
* `fix/<bug-name>`: Bug fix (e.g., `fix/threshold-division-by-zero`).
* `perf/<optimization>`: Performance or memory optimization.
* `docs/<doc-update>`: Documentation improvements.

### Commit Conventions
Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
* `feat: add character n-gram blocking pass`
* `fix: prevent duplicate IDs in candidate generator`
* `perf: vectorize pairwise feature extraction with numpy`
* `docs: update F0.5 optimization methodology in Documentation_template.md`
* `test: add unit test for singleton edge cases`

---

## 6. Testing & Validation

Before submitting a Pull Request, you must verify that all automated checks pass locally.

1. **Run Unit Tests**:
   ```bash
   pytest tests/ -v
   ```

2. **Run Pipeline Smoke Test**:
   ```bash
   python -m business_entity_resolution.pipeline --mode all
   ```

3. **Verify Submission Files with the Official Validator**:
   ```bash
   python utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test
   ```
   *The validator must print `PASS` with exit code `0`.*

---

## 7. Submitting a Pull Request (PR)

1. Push your branch to your GitHub fork.
2. Open a Pull Request targeting the `main` branch.
3. In the PR description:
   * Explain the purpose of the change.
   * Provide the before-and-after validation metrics (e.g., Blocking Recall, Validation Macro $F_{0.5}$).
   * Confirm that all tests and the official validator pass.
4. Maintainers will review your PR and provide constructive feedback.

Thank you for contributing to the Amazon ML Challenge 2026!
