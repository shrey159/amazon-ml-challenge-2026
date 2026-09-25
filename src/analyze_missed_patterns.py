"""Analyze why the current entity-resolution blocker misses true pairs.

This is an analysis-only script. It reproduces the sampled blocking result
using the existing blocker, then compares each missed S1/target pair using
lightweight, local string features.
"""

import difflib
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import (  # noqa: E402
    COMMON_ADDRESS_TOKENS,
    COMMON_NAME_TOKENS,
    create_name_block_index,
    get_address_tokens,
    get_candidates,
    get_name_tokens,
)
from normalize import normalize_address, normalize_name, normalize_text  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
CHUNK_SIZE = 200_000
EXAMPLE_COUNT = 30

NAME_SIMILARITY_THRESHOLD = 0.82
ADDRESS_SIMILARITY_THRESHOLD = 0.80


def load_required_records(path, required_ids, label):
    """Load only requested IDs while scanning a source file in chunks."""
    found = {}
    print(f"Scanning {label} in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        matches = chunk[chunk["entity_id"].isin(required_ids)]
        for _, row in matches.iterrows():
            found[row["entity_id"]] = row.to_dict()

        if chunk_number == 1 or chunk_number % 5 == 0:
            print(f"  {label}: scanned chunk {chunk_number:,}; found {len(found):,}")
        if len(found) == len(required_ids):
            break

    print(f"Loaded {label}: {len(found):,}/{len(required_ids):,} requested records")
    return found


def unique_tokens(value, common_tokens, minimum_length):
    normalized = normalize_text(value)
    return {
        token
        for token in normalized.split()
        if token not in common_tokens and len(token) >= minimum_length
    }


def three_letter_tokens(value):
    return {
        token
        for token in unique_tokens(value, COMMON_NAME_TOKENS, 3)
        if len(token) == 3
    }


def similarity(left, right):
    return difflib.SequenceMatcher(
        None,
        normalize_text(left),
        normalize_text(right),
    ).ratio()


def accent_fold(value):
    decomposed = unicodedata.normalize("NFKD", normalize_text(value))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def edit_distance(left, right):
    """Return Levenshtein distance using only standard Python operations."""
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def script_set(value):
    scripts = set()
    for char in str(value):
        if not unicodedata.category(char).startswith(("L", "M")):
            continue
        name = unicodedata.name(char, "")
        for script in (
            "LATIN",
            "DEVANAGARI",
            "BENGALI",
            "GURMUKHI",
            "GUJARATI",
            "ORIYA",
            "TAMIL",
            "TELUGU",
            "KANNADA",
            "MALAYALAM",
            "SINHALA",
            "THAI",
            "ARABIC",
            "HEBREW",
            "CYRILLIC",
            "GREEK",
        ):
            if script in name:
                scripts.add(script)
                break
    return scripts


def cross_script(left, right):
    left_scripts = script_set(left)
    right_scripts = script_set(right)
    if not left_scripts or not right_scripts:
        return False
    return left_scripts.isdisjoint(right_scripts) and left_scripts != right_scripts


def likely_typo_or_accent(left, right, left_similarity):
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized or left_normalized == right_normalized:
        return False

    folded_left = accent_fold(left)
    folded_right = accent_fold(right)
    if folded_left == folded_right:
        return True

    distance = edit_distance(left_normalized, right_normalized)
    maximum_length = max(len(left_normalized), len(right_normalized))
    small_edit = distance <= max(2, round(maximum_length * 0.12))
    return small_edit and left_similarity >= 0.85


def classify_pair(s1_row, target_row):
    s1_name = s1_row["business_name"]
    target_name = target_row["business_name"]
    s1_address = s1_row["business_address"]
    target_address = target_row["business_address"]

    s1_name_tokens = set(get_name_tokens(s1_name))
    target_name_tokens = set(get_name_tokens(target_name))
    shared_name_tokens = s1_name_tokens & target_name_tokens
    shared_three_letter = three_letter_tokens(s1_name) & three_letter_tokens(target_name)
    normalized_name_exact = normalize_name(s1_name) == normalize_name(target_name)
    name_overlap = (
        len(shared_name_tokens) / min(len(s1_name_tokens), len(target_name_tokens))
        if s1_name_tokens and target_name_tokens
        else 0.0
    )

    s1_address_tokens = set(get_address_tokens(s1_address))
    target_address_tokens = set(get_address_tokens(target_address))
    shared_address_tokens = s1_address_tokens & target_address_tokens
    shared_numeric = {
        token
        for token in shared_address_tokens
        if token.isdigit()
    }
    current_address_rule = len(shared_address_tokens) >= 2

    name_similarity = similarity(s1_name, target_name)
    address_similarity = similarity(s1_address, target_address)
    fuzzy_name = not shared_name_tokens and name_similarity >= NAME_SIMILARITY_THRESHOLD
    fuzzy_address = (
        len(shared_address_tokens) < 2
        and address_similarity >= ADDRESS_SIMILARITY_THRESHOLD
    )
    cross_script_signal = cross_script(s1_name, target_name) or cross_script(
        s1_address, target_address
    )
    typo_signal = likely_typo_or_accent(s1_name, target_name, name_similarity)
    typo_signal = typo_signal or likely_typo_or_accent(
        s1_address,
        target_address,
        address_similarity,
    )

    if cross_script_signal:
        category = "CROSS_SCRIPT"
    elif typo_signal:
        category = "TYPO_OR_ACCENT"
    elif fuzzy_name:
        category = "FUZZY_NAME_POTENTIAL"
    elif fuzzy_address:
        category = "FUZZY_ADDRESS_POTENTIAL"
    elif normalized_name_exact or shared_name_tokens or shared_three_letter or name_overlap:
        category = "NAME_MATCH"
    elif current_address_rule or shared_address_tokens or shared_numeric:
        category = "ADDRESS_MATCH"
    else:
        category = "NO_STRONG_SIGNAL"

    return {
        "shared_name_tokens": shared_name_tokens,
        "shared_three_letter": shared_three_letter,
        "exact_normalized_name": normalized_name_exact,
        "name_token_overlap": name_overlap,
        "shared_address_tokens": shared_address_tokens,
        "shared_numeric_address_tokens": shared_numeric,
        "shared_address_count": len(shared_address_tokens),
        "current_address_rule": current_address_rule,
        "name_similarity": name_similarity,
        "address_similarity": address_similarity,
        "fuzzy_name_potential": fuzzy_name,
        "fuzzy_address_potential": fuzzy_address,
        "cross_script": cross_script_signal,
        "typo_or_accent": typo_signal,
        "category": category,
    }


def main():
    print("Loading ground truth and selecting the fixed sample...")
    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    sample = ground_truth.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    truth_lookup = dict(zip(sample["source1_entity_id"], sample["matched_entity_ids"]))
    required_s1 = set(sample["source1_entity_id"])
    required_s2 = set()
    required_s3 = set()
    for matched_ids in sample["matched_entity_ids"]:
        for entity_id in matched_ids.split(","):
            entity_id = entity_id.strip()
            if entity_id.startswith("S2-"):
                required_s2.add(entity_id)
            elif entity_id.startswith("S3-"):
                required_s3.add(entity_id)

    s1_lookup = load_required_records(
        TRAIN_DIR / "train_source1.tsv", required_s1, "S1"
    )
    s2_lookup = load_required_records(
        TRAIN_DIR / "train_source2.tsv", required_s2, "S2"
    )
    s3_lookup = load_required_records(
        TRAIN_DIR / "train_source3.tsv", required_s3, "S3"
    )

    print("Building current blocking indexes for required targets...")
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))

    missed = []
    total_true = 0
    recovered = 0
    print("Reproducing current blocker and collecting missed pairs...")

    for position, s1_id in enumerate(required_s1, start=1):
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup.get(s1_id, "").split(",")
            if entity_id.strip()
        }
        total_true += len(true_ids)
        candidate_ids = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        recovered += len(true_ids & candidate_ids)

        for target_id in sorted(true_ids - candidate_ids):
            target_row = (s2_lookup if target_id.startswith("S2-") else s3_lookup).get(
                target_id
            )
            if target_row is None:
                continue
            features = classify_pair(s1_row, target_row)
            missed.append(
                {
                    "s1_id": s1_id,
                    "s1_name": s1_row["business_name"],
                    "s1_address": s1_row["business_address"],
                    "target_id": target_id,
                    "target_name": target_row["business_name"],
                    "target_address": target_row["business_address"],
                    **features,
                }
            )

        if position % 2_000 == 0:
            print(f"  Processed {position:,}/{len(required_s1):,} sampled S1 records...")

    print("\n" + "=" * 78)
    print("CURRENT BLOCKER CHECK")
    print("=" * 78)
    print(f"Sampled S1 entities: {len(required_s1):,}")
    print(f"True matches:        {total_true:,}")
    print(f"Recovered:           {recovered:,}")
    print(f"Missed pairs:        {len(missed):,}")
    if total_true:
        print(f"Recall:              {recovered / total_true * 100:.2f}%")

    print("\n" + "=" * 78)
    print("MISSED-PAIR CATEGORY SUMMARY (one primary category per pair)")
    print("=" * 78)
    category_counts = Counter(pair["category"] for pair in missed)
    categories = [
        "NAME_MATCH",
        "ADDRESS_MATCH",
        "FUZZY_NAME_POTENTIAL",
        "FUZZY_ADDRESS_POTENTIAL",
        "CROSS_SCRIPT",
        "TYPO_OR_ACCENT",
        "NO_STRONG_SIGNAL",
    ]
    for category in categories:
        print(f"{category:24} {category_counts[category]:6,}")

    print("\nFeature counts across all missed pairs:")
    feature_counts = {
        "shared useful name tokens": sum(bool(p["shared_name_tokens"]) for p in missed),
        "shared 3-letter tokens": sum(bool(p["shared_three_letter"]) for p in missed),
        "exact normalized name": sum(p["exact_normalized_name"] for p in missed),
        "shared informative address tokens": sum(bool(p["shared_address_tokens"]) for p in missed),
        "shared numeric address tokens": sum(bool(p["shared_numeric_address_tokens"]) for p in missed),
        "2+ shared address tokens": sum(p["current_address_rule"] for p in missed),
        "fuzzy name potential": sum(p["fuzzy_name_potential"] for p in missed),
        "fuzzy address potential": sum(p["fuzzy_address_potential"] for p in missed),
        "cross-script signal": sum(p["cross_script"] for p in missed),
        "typo/accent signal": sum(p["typo_or_accent"] for p in missed),
    }
    for label, count in feature_counts.items():
        print(f"{label:34} {count:6,}")

    print("\n" + "=" * 78)
    print(f"REPRESENTATIVE MISSED PAIRS (first {min(EXAMPLE_COUNT, len(missed))})")
    print("=" * 78)
    for number, pair in enumerate(missed[:EXAMPLE_COUNT], start=1):
        print(f"\n[{number}] {pair['category']}")
        print(f"S1 ID:                 {pair['s1_id']}")
        print(f"S1 name:               {pair['s1_name']}")
        print(f"Target ID:             {pair['target_id']}")
        print(f"Target name:           {pair['target_name']}")
        print(f"S1 address:            {pair['s1_address']}")
        print(f"Target address:        {pair['target_address']}")
        print(f"Shared name tokens:    {sorted(pair['shared_name_tokens'])}")
        print(f"Shared address tokens: {sorted(pair['shared_address_tokens'])}")
        print(f"Name similarity:       {pair['name_similarity']:.3f}")
        print(f"Address similarity:    {pair['address_similarity']:.3f}")
        print(f"2-address rule:        {pair['current_address_rule']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()