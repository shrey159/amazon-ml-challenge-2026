"""Analyze the true pairs recovered by hypothetical Rule F."""

from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import COMMON_ADDRESS_TOKENS, create_name_block_index, get_address_tokens, get_candidates  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
CHUNK_SIZE = 200_000
RULE_F_LIMIT = 5_000
REPRESENTATIVE_COUNT = 30

STREET_MARKERS = {
    "road", "street", "st", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "highway", "hwy", "boulevard", "blvd", "parkway", "pkwy", "way", "place", "pl",
}
LANDMARK_MARKERS = {
    "building", "mall", "market", "plaza", "tower", "complex", "temple", "hospital",
    "school", "college", "university", "airport", "station", "church", "hotel",
}
CITY_MARKERS = {"city", "town", "village", "municipality", "county", "district"}


def load_required_records(path, required_ids, label):
    """Load requested records while scanning a source file in chunks."""
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


def count_global_address_tokens(paths):
    """Count each informative token once per record across complete S2 and S3."""
    frequencies = Counter()
    for path in paths:
        print(f"Counting address tokens in {path.name}...")
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ):
            for address in chunk["business_address"]:
                frequencies.update(set(get_address_tokens(address)))
    return frequencies


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def normalized_words(value):
    return set(get_address_tokens(value)) | set(str(value).lower().split())


def classify_token(token, s1_address, target_address):
    """Apply transparent address-context heuristics to describe a shared token."""
    if token.isdigit():
        return "numeric/address component"

    combined = f"{s1_address} {target_address}".lower()
    words = normalized_words(combined)
    if token in COMMON_ADDRESS_TOKENS or len(token) <= 3:
        return "generic/noisy token"
    if any(
        re.search(rf"\b{re.escape(marker)}\b[^,;]*\b{re.escape(token)}\b", combined)
        or re.search(rf"\b{re.escape(token)}\b[^,;]*\b{re.escape(marker)}\b", combined)
        for marker in STREET_MARKERS
    ):
        return "street/road"
    if any(marker in words for marker in LANDMARK_MARKERS):
        return "landmark/building"
    if any(marker in words for marker in CITY_MARKERS):
        return "city/town"
    return "distinctive business/location term"


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

    s1_lookup = load_required_records(TRAIN_DIR / "train_source1.tsv", required_s1, "S1")
    s2_lookup = load_required_records(TRAIN_DIR / "train_source2.tsv", required_s2, "S2")
    s3_lookup = load_required_records(TRAIN_DIR / "train_source3.tsv", required_s3, "S3")
    target_lookup = {**s2_lookup, **s3_lookup}

    print("\nCounting global frequencies across complete S2 and S3 files...")
    global_frequencies = count_global_address_tokens(
        [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
    )

    print("\nReproducing the current baseline and collecting missed pairs...")
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))
    true_matches = 0
    baseline_recovered = 0
    missed_pairs = []
    for s1_id in required_s1:
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup.get(s1_id, "").split(",")
            if entity_id.strip()
        }
        true_matches += len(true_ids)
        candidate_ids = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        baseline_recovered += len(true_ids & candidate_ids)
        for target_id in sorted(true_ids - candidate_ids):
            target_row = target_lookup[target_id]
            s1_tokens = set(get_address_tokens(s1_row["business_address"]))
            target_tokens = set(get_address_tokens(target_row["business_address"]))
            shared_tokens = s1_tokens & target_tokens
            missed_pairs.append(
                {
                    "s1_id": s1_id,
                    "target_id": target_id,
                    "s1_name": s1_row["business_name"],
                    "target_name": target_row["business_name"],
                    "s1_address": s1_row["business_address"],
                    "target_address": target_row["business_address"],
                    "shared_tokens": shared_tokens,
                }
            )

    recovered = []
    for pair in missed_pairs:
        if len(pair["shared_tokens"]) != 1:
            continue
        token = next(iter(pair["shared_tokens"]))
        frequency = global_frequencies[token]
        if frequency <= RULE_F_LIMIT:
            recovered.append(
                {
                    **pair,
                    "token": token,
                    "frequency": frequency,
                    "category": classify_token(
                        token, pair["s1_address"], pair["target_address"]
                    ),
                }
            )

    print("\n" + "=" * 88)
    print("RULE F RECOVERY ANALYSIS")
    print("=" * 88)
    print(f"True matches:                 {true_matches:,}")
    print(f"Baseline recovered:           {baseline_recovered:,}")
    print(f"Baseline missed:              {len(missed_pairs):,}")
    print(f"Rule F recovered missed:      {len(recovered):,}")
    print(f"S1 entities affected:         {len({p['s1_id'] for p in recovered}):,}")
    print(f"Rule F limit:                  {RULE_F_LIMIT:,}")

    frequencies = [pair["frequency"] for pair in recovered]
    frequency_counts = Counter(frequencies)
    print("\nGLOBAL FREQUENCY DISTRIBUTION OF RECOVERED SHARED TOKENS")
    print(f"Minimum:                      {min(frequencies):,}")
    print(f"Median:                       {percentile(frequencies, .50):,.1f}")
    print(f"90th percentile:              {percentile(frequencies, .90):,.1f}")
    print(f"95th percentile:              {percentile(frequencies, .95):,.1f}")
    print(f"Maximum:                      {max(frequencies):,}")
    print("Exact frequency counts:")
    for frequency, count in sorted(frequency_counts.items()):
        print(f"  {frequency:>8,}: {count:,}")

    token_counts = Counter(pair["token"] for pair in recovered)
    print("\nTOP 30 SHARED ADDRESS TOKENS CAUSING RECOVERIES")
    print(f"{'Token':24} {'Global frequency':>17} {'Recovered pairs':>17}")
    for token, count in token_counts.most_common(30):
        print(f"{token:24} {global_frequencies[token]:17,} {count:17,}")

    category_counts = Counter(pair["category"] for pair in recovered)
    print("\nTOKEN TYPE IDENTIFICATION (heuristic, based on address context)")
    for category, count in category_counts.most_common():
        print(f"{category:35} {count:6,}")

    print("\n" + "=" * 88)
    print("30 REPRESENTATIVE RULE F RECOVERIES")
    print("=" * 88)
    for number, pair in enumerate(
        sorted(recovered, key=lambda item: (item["s1_id"], item["target_id"]))[
            :REPRESENTATIVE_COUNT
        ],
        start=1,
    ):
        source = "S2" if pair["target_id"].startswith("S2-") else "S3"
        print(f"\n[{number}] {pair['category']}")
        print(f"S1 ID:                 {pair['s1_id']}")
        print(f"Target ID:             {pair['target_id']}")
        print(f"Source:                {source}")
        print(f"S1 business name:      {pair['s1_name']}")
        print(f"Target business name:  {pair['target_name']}")
        print(f"S1 address:            {pair['s1_address']}")
        print(f"Target address:        {pair['target_address']}")
        print(f"Shared address token:  {pair['token']}")
        print(f"Token frequency:       {pair['frequency']:,}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()