"""
src/business_entity_resolution/config.py
========================================
Centralized configuration management for Amazon ML Challenge 2026.
Business Entity Resolution MVP.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # --- Reproducibility ---
    random_seed: int = 42

    # --- Directory Paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset")
    train_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "train")
    test_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "test")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "output")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "experiments")
    validation_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "validation")

    # --- File Paths ---
    train_source1_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "train" / "train_source1.tsv")
    train_source2_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "train" / "train_source2.tsv")
    train_source3_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "train" / "train_source3.tsv")
    train_ground_truth_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "train" / "train_ground_truth.tsv")

    test_source1_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "test" / "test_source1.tsv")
    test_source2_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "test" / "test_source2.tsv")
    test_source3_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "dataset" / "test" / "test_source3.tsv")

    matching_output_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "output" / "matching_results.tsv")
    candidate_output_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "output" / "candidate_pairs.tsv")
    model_artifact_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models" / "model.joblib")
    threshold_artifact_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models" / "threshold.json")
    feature_metadata_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models" / "feature_metadata.json")
    experiment_log_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "experiments" / "experiment_log.csv")

    # --- Validation Split ---
    val_split_ratio: float = 0.20  # Entity-aware split on Source 1 entities

    # --- Blocking Parameters ---
    # Multi-pass blocking switches
    use_exact_name_block: bool = True
    use_name_prefix_block: bool = True
    use_name_2prefix_block: bool = True
    use_address_token_block: bool = True
    use_name_address_composite_block: bool = True
    use_country_name_block: bool = True
    use_tfidf_char_block: bool = True

    # Blocking constraints
    max_candidates_per_s1: int = 150  # Cap candidates per S1 entity
    max_block_bucket_size: int = 500  # Avoid exploding huge uninformative buckets
    tfidf_char_top_k: int = 25
    tfidf_char_min_sim: float = 0.20

    # --- Negative Sampling ---
    # Max hard negatives per positive in training
    max_negatives_per_positive: int = 15

    # --- Modeling ---
    model_type: str = "hist_gradient_boosting"  # 'logistic_regression' or 'hist_gradient_boosting' or 'lightgbm'
    class_weight: str = "balanced"

    # --- Threshold Optimization ---
    threshold_search_grid: List[float] = field(
        default_factory=lambda: [
            0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97
        ]
    )
    default_match_threshold: float = 0.75

    # --- Metric Parameters ---
    f_beta: float = 0.5  # Precision-heavy F0.5

    def ensure_dirs(self):
        """Ensure all required project directories exist."""
        for p in [self.train_dir, self.test_dir, self.output_dir, self.models_dir, self.experiments_dir, self.validation_dir]:
            p.mkdir(parents=True, exist_ok=True)


config = Config()
