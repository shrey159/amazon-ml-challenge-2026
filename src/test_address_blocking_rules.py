"""Measure address-blocking alternatives without changing the production blocker."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import create_name_block_index, get_address_tokens, get_candidates  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
CHUNK_SIZE = 200_000


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
    """Count records containing each informative address token across both sources."""
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


def build_address_index(records):
    """Build country/token indexes and per-entity token sets for required targets."""
    token_index = defaultdict(set)
    entity_tokens = {}
    for entity_id, row in records.items():
        country = str(row["country"]).strip().lower()
        tokens = set(get_address_tokens(row["business_address"]))
        entity_tokens[entity_id] = tokens
        for token in tokens:
            token_index[(country, token)].add(entity_id)
    return token_index, entity_tokens


def add_single_token_candidates(
    s1_row, base_candidates, token_index, entity_tokens, global_frequencies, limit
):
    """Add targets sharing exactly one address token and satisfying a frequency limit."""
    country = str(s1_row["country"]).strip().lower()
    s1_tokens = set(get_address_tokens(s1_row["business_address"]))
    possible = set()
    for token in s1_tokens:
        frequency = global_frequencies[token]
        if limit is not None and frequency > limit:
            continue
        possible.update(token_index.get((country, token), set()))

    for entity_id in possible:
        if len(s1_tokens & entity_tokens[entity_id]) == 1:
            base_candidates.add(entity_id)
    return base_candidates


def percentile(values, fraction):
    """Return a linearly interpolated percentile without requiring NumPy."""
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


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

    print("\nBuilding current indexes for the required target records...")
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))
    address_index, entity_tokens = build_address_index(target_lookup)

    base_candidates = {}
    missed_pairs = []
    true_matches = 0
    for s1_id in required_s1:
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup.get(s1_id, "").split(",")
            if entity_id.strip()
        }
        candidates = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        base_candidates[s1_id] = candidates
        true_matches += len(true_ids)
        for target_id in sorted(true_ids - candidates):
            target_row = target_lookup[target_id]
            shared = set(get_address_tokens(s1_row["business_address"])) & set(
                get_address_tokens(target_row["business_address"])
            )
            missed_pairs.append(
                {
                    "s1_id": s1_id,
                    "target_id": target_id,
                    "s1_address": s1_row["business_address"],
                    "target_address": target_row["business_address"],
                    "shared_tokens": shared,
                }
            )

    rules = {
        "A. Current rule": None,
        "B. One token, any frequency": "ANY",
        "C. One token, frequency <= 100": 100,
        "D. One token, frequency <= 500": 500,
        "E. One token, frequency <= 1,000": 1_000,
        "F. One token, frequency <= 5,000": 5_000,
    }
    results = {}
    for rule_name, limit in rules.items():
        recovered = 0
        affected = 0
        candidate_sizes = []
        rule_candidates = {}
        for s1_id in required_s1:
            if s1_id not in base_candidates:
                continue
            candidates = set(base_candidates[s1_id])
            if limit == "ANY":
                candidates = add_single_token_candidates(
                    s1_lookup[s1_id], candidates, address_index, entity_tokens,
                    global_frequencies, None
                )
            elif isinstance(limit, int):
                candidates = add_single_token_candidates(
                    s1_lookup[s1_id], candidates, address_index, entity_tokens,
                    global_frequencies, limit
                )
            rule_candidates[s1_id] = candidates
            candidate_sizes.append(len(candidates))
            if candidates != base_candidates[s1_id]:
                affected += 1
            true_ids = {
                entity_id.strip()
                for entity_id in truth_lookup.get(s1_id, "").split(",")
                if entity_id.strip()
            }
            recovered += len(true_ids & candidates)

        results[rule_name] = {
            "recovered": recovered,
            "missed": true_matches - recovered,
            "affected": affected,
            "sizes": candidate_sizes,
            "candidates": rule_candidates,
        }

    print("\n" + "=" * 88)
    print("ADDRESS BLOCKING RULE EXPERIMENT")
    print("=" * 88)
    print(f"Sampled S1 entities: {len(base_candidates):,}")
    print(f"Required S2/S3 target records: {len(target_lookup):,}")
    print(f"True matches: {true_matches:,}")
    print(f"Baseline missed pairs reproduced: {len(missed_pairs):,}")
    print("\nCandidate statistics are over all sampled S1 entities and required targets.")
    print("Affected means the candidate set changed versus Rule A.")
    print(
        f"{'Rule':34} {'Recovered':>10} {'Missed':>8} {'Recall':>9} "
        f"{'Affected S1':>12} {'Min':>8} {'Median':>8} {'P90':>8} {'P95':>8} {'Max':>8}"
    )
    for rule_name, result in results.items():
        sizes = result["sizes"]
        print(
            f"{rule_name:34} {result['recovered']:10,} {result['missed']:8,} "
            f"{result['recovered'] / true_matches * 100:8.2f}% {result['affected']:12,} "
            f"{min(sizes):8,} {percentile(sizes, .50):8.1f} "
            f"{percentile(sizes, .90):8.1f} {percentile(sizes, .95):8.1f} {max(sizes):8,}"
        )

    best_name = max(
        results,
        key=lambda name: (
            results[name]["recovered"],
            -percentile(results[name]["sizes"], .50),
            -max(results[name]["sizes"]),
        ),
    )
    best = results[best_name]
    recoverable_examples = []
    for pair in missed_pairs:
        if pair["target_id"] in best["candidates"].get(pair["s1_id"], set()):
            for token in sorted(pair["shared_tokens"]):
                recoverable_examples.append((pair, token, global_frequencies[token]))

    print("\n" + "=" * 88)
    print(f"EXAMPLES RECOVERED BY BEST-PERFORMING RULE: {best_name}")
    print("=" * 88)
    for number, (pair, token, frequency) in enumerate(recoverable_examples[:20], start=1):
        print(f"\n[{number}]")
        print(f"S1 ID:                 {pair['s1_id']}")
        print(f"Target ID:             {pair['target_id']}")
        print(f"S1 address:            {pair['s1_address']}")
        print(f"Target address:        {pair['target_address']}")
        print(f"Shared address token:  {token}")
        print(f"Global token frequency: {frequency:,}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()