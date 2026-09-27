"""
utils/generate_mock_dataset.py
==============================
Generates a representative, strictly schema-compliant synthetic dataset
for pipeline smoke testing and verification.
Follows exact official constraints:
- Sources: entity_id, business_name, business_address, country
- Prefixes: S1-, S2-, S3-
- Train countries: US, India
- Test countries: US, India, France
- Ground truth: source1_entity_id, matched_entity_ids (singletons, 1-to-1, multi-matches)
- TSV format (tab-separated)
"""

from pathlib import Path
import pandas as pd


def generate_mock_data(base_dir: Path):
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # TRAIN DATASET
    # -------------------------------------------------------------
    s1_train = [
        {"entity_id": "S1-00001", "business_name": "Acme Industrial Supplies LLC", "business_address": "100 Market St, Suite 400, San Francisco, CA 94105", "country": "US"},
        {"entity_id": "S1-00002", "business_name": "Bharat Agro Foods Private Limited", "business_address": "Plot 42, Sector 18, Vashi, Navi Mumbai, Maharashtra 400703", "country": "India"},
        {"entity_id": "S1-00003", "business_name": "Pacific Horizon Logistics Inc", "business_address": "1200 Harbor Boulevard, Long Beach, CA 90802", "country": "US"},
        {"entity_id": "S1-00004", "business_name": "Himalaya Herbal Remedies", "business_address": "Main Road, Near Clock Tower, Dehradun, Uttarakhand 248001", "country": "India"},
        {"entity_id": "S1-00005", "business_name": "Apex Cloud Systems Corporation", "business_address": "500 West Madison St, Chicago, IL 60661", "country": "US"},
        {"entity_id": "S1-00006", "business_name": "Singleton Tech Solutions", "business_address": "77 Oak Ridge Lane, Austin, TX 78701", "country": "US"}, # Singleton (0 matches)
        {"entity_id": "S1-00007", "business_name": "Royal Bengal Spices Co", "business_address": "14 Strand Road, Burrabazar, Kolkata, West Bengal 700001", "country": "India"},
    ]

    s2_train = [
        {"entity_id": "S2-00010", "business_name": "Acme Industrial Supplies", "business_address": "100 Market Street Ste 400 San Francisco CA", "country": "USA"}, # Match to S1-00001
        {"entity_id": "S2-00020", "business_name": "Bharat Agro Foods Pvt Ltd", "business_address": "Plot No 42 Sec 18 Vashi Navi Mumbai", "country": "India"}, # Match to S1-00002
        {"entity_id": "S2-00030", "business_name": "Pacific Horizon Logistics", "business_address": "1200 Harbor Blvd Long Beach CA", "country": "US"}, # Match to S1-00003
        {"entity_id": "S2-00040", "business_name": "Apex Cloud Systems Corp", "business_address": "500 W Madison Street Chicago Illinois", "country": "US"}, # Match to S1-00005
        {"entity_id": "S2-00050", "business_name": "Delta Marine Supplies LLC", "business_address": "88 Waterfront Drive Seattle WA", "country": "US"}, # Hard negative
        {"entity_id": "S2-00060", "business_name": "Royal Bengal Spices", "business_address": "14 Strand Rd Kolkata WB", "country": "India"}, # Match to S1-00007
        {"entity_id": "S2-00070", "business_name": "Bharat Electricals Ltd", "business_address": "Plot 10 Industrial Area Mumbai", "country": "India"}, # Hard negative
    ]

    s3_train = [
        {"entity_id": "S3-00100", "business_name": "Acme Industrial", "business_address": "100 Market St San Francisco 94105", "country": "US"}, # Multi-match to S1-00001
        {"entity_id": "S3-00200", "business_name": "Bharat Agro Foods", "business_address": "Vashi Sector 18 Navi Mumbai Maharashtra", "country": "IN"}, # Multi-match to S1-00002
        {"entity_id": "S3-00300", "business_name": "Himalaya Herbal Remedies Dehradun", "business_address": "Near Clock Tower Main Rd Dehradun", "country": "India"}, # Match to S1-00004
        {"entity_id": "S3-00400", "business_name": "Apex Cloud", "business_address": "Madison St Chicago IL", "country": "USA"}, # Multi-match to S1-00005
        {"entity_id": "S3-00500", "business_name": "Pacific Shipping Co", "business_address": "90 Harbor Rd Long Beach CA", "country": "US"}, # Hard negative
        {"entity_id": "S3-00600", "business_name": "Omega Bio Health", "business_address": "101 Science Park Boston MA", "country": "US"}, # Negative
    ]

    # Ground truth:
    # S1-00001 -> S2-00010, S3-00100 (multi-match)
    # S1-00002 -> S2-00020, S3-00200 (multi-match)
    # S1-00003 -> S2-00030 (single-match)
    # S1-00004 -> S3-00300 (single-match)
    # S1-00005 -> S2-00040, S3-00400 (multi-match)
    # S1-00006 -> empty (singleton)
    # S1-00007 -> S2-00060 (single-match)
    gt_train = [
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00010,S3-00100"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00020,S3-00200"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": "S2-00030"},
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S3-00300"},
        {"source1_entity_id": "S1-00005", "matched_entity_ids": "S2-00040,S3-00400"},
        {"source1_entity_id": "S1-00006", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-00007", "matched_entity_ids": "S2-00060"},
    ]

    # -------------------------------------------------------------
    # TEST DATASET (Includes France)
    # -------------------------------------------------------------
    s1_test = [
        {"entity_id": "S1-90001", "business_name": "Boulangerie Patisserie Saint Honore SARL", "business_address": "28 Rue du Faubourg Saint Honore, 75008 Paris", "country": "France"},
        {"entity_id": "S1-90002", "business_name": "Silicon Valley Quantum Labs Inc", "business_address": "3000 Sand Hill Road, Menlo Park, CA 94025", "country": "US"},
        {"entity_id": "S1-90003", "business_name": "Kaveri Textiles Private Limited", "business_address": "88 Cross Cut Road, Gandhipuram, Coimbatore, Tamil Nadu 641012", "country": "India"},
        {"entity_id": "S1-90004", "business_name": "Test Singleton Enterprise", "business_address": "99 Mystery Street, Nowhere", "country": "US"}, # Singleton
    ]

    s2_test = [
        {"entity_id": "S2-90010", "business_name": "Boulangerie St Honore", "business_address": "28 Rue Faubourg St Honore Paris 75008", "country": "France"}, # Match to S1-90001
        {"entity_id": "S2-90020", "business_name": "Silicon Valley Quantum Labs", "business_address": "3000 Sand Hill Rd Menlo Park CA", "country": "US"}, # Match to S1-90002
        {"entity_id": "S2-90030", "business_name": "Kaveri Textiles Pvt Ltd", "business_address": "88 Cross Cut Rd Coimbatore", "country": "India"}, # Match to S1-90003
        {"entity_id": "S2-90040", "business_name": "Random Distractor Inc", "business_address": "123 Random Way New York NY", "country": "US"},
    ]

    s3_test = [
        {"entity_id": "S3-90050", "business_name": "St Honore Patisserie", "business_address": "Faubourg Saint Honore Paris", "country": "France"}, # Multi-match to S1-90001
        {"entity_id": "S3-90060", "business_name": "Quantum Labs Silicon Valley", "business_address": "Sand Hill Rd Menlo Park California", "country": "USA"}, # Multi-match to S1-90002
        {"entity_id": "S3-90070", "business_name": "Unrelated Bakery", "business_address": "10 Rue de la Paix Paris", "country": "France"},
    ]

    # Save all TSVs
    pd.DataFrame(s1_train).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_train).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_train).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt_train).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    pd.DataFrame(s1_test).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_test).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_test).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    print(f"Generated mock challenge dataset in {base_dir}")


if __name__ == "__main__":
    generate_mock_data(Path("dataset"))
