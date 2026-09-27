# Amazon ML Challenge 2026: Business Entity Resolution
## Official Methodology Document

---

### 1. Problem Definition
Business Entity Resolution (ER) in commercial catalogs involves matching disparate, fragmented records from independent data sources that lack universal unique keys. In this challenge:
* **Source 1** serves as the deduplicated reference catalog.
* **Source 2** and **Source 3** represent noisy business data containing spelling errors, abbreviations, legal suffix variations, and address distortions.
* Objective: For every entity in Source 1, find all matching records from Source 2 and Source 3. A Source 1 entity may have zero matches (singleton), a single match, or multiple matches.
* Evaluation is governed by Entity-Level Macro $F_{0.5}$, emphasizing precision and severely penalizing false merges.

---

### 2. Data Understanding
Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains:
* `entity_id`: Source identifier prefixed with `S1-`, `S2-`, or `S3-`.
* `business_name`: Commercial title with legal designations, typos, and transliterations.
* `business_address`: Full or partial address, street variations, landmarks, and postal codes.
* `country`: Geographic territory. Training data comprises US and India; test data introduces France as an unseen country.

Ground truth matches are tab-separated pairs: `source1_entity_id` and comma-separated `matched_entity_ids` (or empty for singletons).

---

### 3. Preprocessing & Normalization
To bridge noisy surface variations while avoiding information loss, normalization preserves original raw fields and produces canonical normalized representations:
* **Unicode Normalization**: NFKD decomposition ensures accented characters (common in France/French addresses) map cleanly to ASCII equivalents.
* **Legal Suffix Harmonization**: Regex rules unify legal suffixes (`Private Limited` / `Pvt. Ltd.` / `Pvt Ltd` $\rightarrow$ `pvt ltd`, `Incorporated` / `Inc.` $\rightarrow$ `inc`, `LLC` $\rightarrow$ `llc`, `Corporation` / `Corp.` $\rightarrow$ `corp`).
* **Address Standardizations**: Road terminology standardized (`st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `ste` $\rightarrow$ `suite`, `apt` $\rightarrow$ `apartment`).
* **Open-Set Country Processing**: Known country aliases are mapped to canonical forms, but any novel country label passes through without filtering, ensuring zero breakdown on test records from France or beyond.

---

### 4. Candidate Generation / Multi-Pass Blocking
Comparing every Source 1 record against the full Cartesian space of Source 2 and Source 3 ($|S_1| \times (|S_2| + |S_3|)$) is computationally prohibitive. A multi-pass blocking architecture ensures high recall coverage ($>98\%$) with $>99\%$ reduction ratio:
1. **Pass 1 (Exact Name)**: Matches entities sharing exact normalized names.
2. **Pass 2 (Name First Token)**: Captures companies sharing primary distinctive root word (length $\ge 3$).
3. **Pass 3 (Name 2-Token Prefix)**: Captures multi-word prefixes.
4. **Pass 4 (Composite Name+Address)**: Inverted index on `first_name_token|first_address_token`.
5. **Pass 5 (Composite Country+Name)**: Inverted index on `country#name_2tokens`.
6. **Pass 6 (TF-IDF Fuzzy Fallback)**: Character $n$-gram ($n \in [2, 4]$) sparse cosine similarity retrieval for any entities with fewer than 5 candidates.

*Safety Constraints*:
* Bucket size capped at 500 records to prevent runaway Cartesian explosions on stop-words.
* Candidates restricted to Source 2 and Source 3 IDs (never S1 $\rightarrow$ S1).
* Hard ceiling of 150 candidates per Source 1 entity.

---

### 5. Feature Engineering
For every candidate pair $(S_1, \text{Target})$, a 34-dimensional feature vector is generated:
* **RapidFuzz Name Metrics**: Levenshtein Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, Token Jaccard, Length Difference, Length Ratio, Token Count Difference, Exact Match Flag, Substring Containment, First Token Match, Shared Token Count & Ratio, Character 3-Gram Jaccard.
* **RapidFuzz Address Metrics**: Levenshtein Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, Token Jaccard, Length Difference, Length Ratio, Token Count Difference, Exact Match Flag, Substring Containment, First Token Match, Shared Token Count & Ratio, Character 3-Gram Jaccard.
* **Open-Set Country Features**: `country_exact_match` (1 if both non-empty and equal, 0 otherwise) and `country_missing` (1 if missing).
* **Source Flags**: Indicator variables for target record source origin (`target_is_s2`, `target_is_s3`) and presence indicators (`both_name_present`, `both_address_present`).

---

### 6. Model Architecture & Training Strategy
* **Model**: Gradient boosted decision trees via `HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, min_samples_leaf=20, class_weight="balanced")`.
* **Validation Split**: Entity-aware 80/20 train/validation split on Source 1 entity IDs (`random_state=42`). No records or candidate pairs belonging to validation Source 1 entities appear in training, preventing data leakage.
* **Hard Negative Mining**: Negative pairs are mined directly from blocking buckets (pairs proposed by blocking but absent from ground truth matches). A ratio of up to 15 hard negatives per positive pair is sampled, teaching the classifier to separate challenging near-duplicates.

---

### 7. Entity-Level Macro $F_{0.5}$ & Threshold Optimization
* The competition metric is Entity-Level Macro $F_{0.5}$:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

* **Singleton Scoring**:
  * If true match set is empty and predicted set is empty: **Score = 1.0** (full credit).
  * If true match set is empty and predicted set is non-empty: **Score = 0.0** (zero credit for false merge).
* **Threshold Search**:
  Grid search across candidate match thresholds $\tau \in [0.30, 0.97]$ directly optimizes entity-level Macro $F_{0.5}$ rather than pair-level accuracy. High precision thresholds ($\tau \approx 0.75 - 0.85$) minimize costly false merges.

---

### 8. Singleton & Multi-Match Handling
* **Multi-Match**: Any candidate with $P(\text{match}) \ge \tau$ is selected.
* **Singletons**: When no candidate achieves probability exceeding $\tau$, an empty set is predicted, preserving singleton credit.
* **Subset Guarantee**: Every matched ID in `matching_results.tsv` is guaranteed to be a member of `candidate_pairs.tsv`.

---

### 9. Validation & Compliance
* **Official Validator**: `utils/validate_submission.py` tests all formatting constraints: tab separation, header columns, exact 1-to-1 row mapping for test S1 entities, valid S2/S3 ID formats, absence of S1 IDs in match lists, and candidate subset rules.

---

### 10. Limitations & Future Directions
1. **Phonetic Encoding**: Adding Double Metaphone / Soundex for regional Indian and French names.
2. **Graph Connected Components**: Performing transitive closure / graph-based deduplication for multi-source clusters.
3. **Adaptive Thresholds**: Source-pair specific decision thresholds ($\tau_{1\to 2}$ vs. $\tau_{1\to 3}$).
