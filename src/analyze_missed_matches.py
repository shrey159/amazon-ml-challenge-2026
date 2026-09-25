
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from normalize import normalize_text
from blocking import (
    COMMON_NAME_TOKENS,
    create_name_block_index,
    get_candidates,
)


TRAIN_DIR = Path("dataset/train")

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
CHUNK_SIZE = 200_000

RARITY_THRESHOLDS = [10, 50, 100, 500, 1000, 5000]


def get_name_tokens_len3(name):
    text = normalize_text(name)

    tokens = set()

    for token in text.split():
        if token in COMMON_NAME_TOKENS:
            continue

        if len(token) >= 3:
            tokens.add(token)

    return tokens


def calculate_global_token_frequency():
    print("\n" + "=" * 60)
    print("CALCULATING GLOBAL 3-LETTER TOKEN FREQUENCIES")
    print("=" * 60)

    token_frequency = Counter()

    for source_name in ["train_source2.tsv", "train_source3.tsv"]:

        print(f"\nScanning {source_name}...")

        file_path = TRAIN_DIR / source_name

        processed = 0

        for chunk in pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["business_name"],
            chunksize=CHUNK_SIZE,
        ):

            for name in chunk["business_name"]:
                tokens = get_name_tokens_len3(name)

                for token in tokens:
                    if len(token) == 3:
                        token_frequency[token] += 1

            processed += len(chunk)

            if processed % 1_000_000 == 0:
                print(f"Processed {processed:,} rows...")

        print(f"Finished {source_name}: {processed:,} rows")

    print(
        f"\nTotal unique 3-letter tokens: "
        f"{len(token_frequency):,}"
    )

    return token_frequency


def main():

    # ---------------------------------------------------------
    # 1. LOAD GROUND TRUTH SAMPLE
    # ---------------------------------------------------------

    print("Loading ground truth...")

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    sample = ground_truth.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    truth_lookup = dict(
        zip(
            sample["source1_entity_id"],
            sample["matched_entity_ids"],
        )
    )

    required_s1 = set(sample["source1_entity_id"])

    required_s2 = set()
    required_s3 = set()

    for matched_ids in sample["matched_entity_ids"]:

        if not matched_ids:
            continue

        for entity_id in matched_ids.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                required_s2.add(entity_id)

            elif entity_id.startswith("S3-"):
                required_s3.add(entity_id)

    print(f"Sampled S1 records: {len(required_s1)}")
    print(f"Required S2 IDs: {len(required_s2)}")
    print(f"Required S3 IDs: {len(required_s3)}")

    # ---------------------------------------------------------
    # 2. LOAD REQUIRED RECORDS
    # ---------------------------------------------------------

    print("\nLoading required records...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1 = s1[
        s1["entity_id"].isin(required_s1)
    ].copy()

    # S2

    s2_chunks = []

    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_s2)
        ]

        if not matched.empty:
            s2_chunks.append(matched)

    s2 = pd.concat(
        s2_chunks,
        ignore_index=True,
    )

    # S3

    s3_chunks = []

    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_s3)
        ]

        if not matched.empty:
            s3_chunks.append(matched)

    s3 = pd.concat(
        s3_chunks,
        ignore_index=True,
    )

    print(f"Loaded S1 records: {len(s1)}")
    print(f"Loaded S2 records: {len(s2)}")
    print(f"Loaded S3 records: {len(s3)}")

    # ---------------------------------------------------------
    # 3. BUILD CURRENT BLOCKING INDEXES
    # ---------------------------------------------------------

    print("\nBuilding blocking indexes...")

    s2_indexes = create_name_block_index(s2)
    s3_indexes = create_name_block_index(s3)

    print("Blocking indexes built.")

    # ---------------------------------------------------------
    # 4. CALCULATE TRUE GLOBAL TOKEN FREQUENCIES
    # ---------------------------------------------------------

    token_frequency = calculate_global_token_frequency()

    # ---------------------------------------------------------
    # 5. LOOKUP TABLES
    # ---------------------------------------------------------

    s2_lookup = {
        row["entity_id"]: row
        for _, row in s2.iterrows()
    }

    s3_lookup = {
        row["entity_id"]: row
        for _, row in s3.iterrows()
    }

    # ---------------------------------------------------------
    # 6. EVALUATE CURRENT BLOCKER
    # ---------------------------------------------------------

    print("\nEvaluating blocking...")

    total_true_matches = 0
    recovered_matches = 0

    missed_pairs = []

    for counter, (_, s1_row) in enumerate(
        s1.iterrows(),
        start=1,
    ):

        s1_id = s1_row["entity_id"]

        matched_ids_string = truth_lookup.get(
            s1_id,
            "",
        )

        if not matched_ids_string:
            continue

        true_ids = {
            x.strip()
            for x in matched_ids_string.split(",")
            if x.strip()
        }

        total_true_matches += len(true_ids)

        candidate_ids = set(
            get_candidates(
                s1_row,
                s2_indexes,
                s3_indexes,
            )
        )

        recovered_matches += len(
            true_ids & candidate_ids
        )

        missed = true_ids - candidate_ids

        for target_id in missed:

            if target_id.startswith("S2-"):
                target_row = s2_lookup.get(target_id)
            else:
                target_row = s3_lookup.get(target_id)

            if target_row is None:
                continue

            s1_tokens = get_name_tokens_len3(
                s1_row["business_name"]
            )

            target_tokens = get_name_tokens_len3(
                target_row["business_name"]
            )

            shared_tokens = (
                s1_tokens & target_tokens
            )

            shared_3letter = {
                token: token_frequency[token]
                for token in shared_tokens
                if len(token) == 3
            }

            missed_pairs.append(
                {
                    "s1_id": s1_id,
                    "target_id": target_id,
                    "s1_name": s1_row["business_name"],
                    "target_name": target_row["business_name"],
                    "shared_3letter": shared_3letter,
                }
            )

        if counter % 2000 == 0:
            print(
                f"Processed "
                f"{counter}/{len(s1)} S1 records..."
            )

    # ---------------------------------------------------------
    # 7. CURRENT RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CURRENT BLOCKING")
    print("=" * 60)

    print(
        f"Total true matches: "
        f"{total_true_matches:,}"
    )

    print(
        f"Recovered matches: "
        f"{recovered_matches:,}"
    )

    current_missed = len(missed_pairs)

    print(
        f"Missed matches: "
        f"{current_missed:,}"
    )

    current_recall = (
        recovered_matches
        / total_true_matches
        * 100
    )

    print(
        f"Blocking Recall: "
        f"{current_recall:.2f}%"
    )

    # ---------------------------------------------------------
    # 8. GLOBAL RARITY ANALYSIS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GLOBAL 3-LETTER TOKEN RARITY ANALYSIS")
    print("=" * 60)

    for threshold in RARITY_THRESHOLDS:

        recovered = 0

        for pair in missed_pairs:

            useful_tokens = [
                token
                for token, frequency
                in pair["shared_3letter"].items()
                if frequency <= threshold
            ]

            if useful_tokens:
                recovered += 1

        estimated_recall = (
            recovered_matches
            + recovered
        ) / total_true_matches * 100

        print(
            f"\nFrequency <= {threshold:,}"
        )

        print(
            f"Missed pairs recovered: "
            f"{recovered}"
        )

        print(
            f"Estimated Blocking Recall: "
            f"{estimated_recall:.2f}%"
        )

        print(
            f"Remaining missed pairs: "
            f"{current_missed - recovered}"
        )

    # ---------------------------------------------------------
    # 9. SHOW RARE EXAMPLES
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXAMPLES USING GLOBAL FREQUENCY <= 500")
    print("=" * 60)

    shown = 0

    for pair in missed_pairs:

        rare_tokens = {
            token: frequency
            for token, frequency
            in pair["shared_3letter"].items()
            if frequency <= 500
        }

        if not rare_tokens:
            continue

        print("\nS1:", pair["s1_id"])
        print("   ", pair["s1_name"])

        print(
            "Target:",
            pair["target_id"],
        )

        print(
            "   ",
            pair["target_name"],
        )

        print(
            "Globally rare shared tokens:",
            rare_tokens,
        )

        shown += 1

        if shown >= 20:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
