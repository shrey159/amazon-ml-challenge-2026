"""Measure candidate cost and precision for hypothetical Rule F."""

from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import (  # noqa: E402
    COMMON_ADDRESS_TOKENS,
    create_name_block_index,
    get_address_tokens,
    get_candidates,
)
from analyze_rule_f_recoveries import (  # noqa: E402
    count_global_address_tokens,
    load_required_records,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
RULE_F_LIMIT = 5_000
CHUNK_SIZE = 200_000

STREET_MARKERS = {
    "road", "street", "st", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "highway", "hwy", "boulevard", "blvd", "parkway", "pkwy", "way", "place", "pl",
}
CITY_MARKERS = {"city", "town", "village", "county", "district", "township", "cdp"}
LANDMARK_MARKERS = {
    "building", "mall", "market", "plaza", "tower", "complex", "temple", "hospital",
    "school", "college", "university", "airport", "station", "church", "hotel",
}


def build_address_index(records):
    """Index required targets by country and informative address token."""
    token_index = defaultdict(set)
    entity_tokens = {}
    for entity_id, row in records.items():
        country = str(row["country"]).strip().lower()
        tokens = set(get_address_tokens(row["business_address"]))
        entity_tokens[entity_id] = tokens
        for token in tokens:
            token_index[(country, token)].add(entity_id)
    return token_index, entity_tokens


def rule_f_candidates(s1_row, token_index, entity_tokens, global_frequencies):
    """Return targets sharing exactly one qualifying informative address token."""
    country = str(s1_row["country"]).strip().lower()
    s1_tokens = set(get_address_tokens(s1_row["business_address"]))
    possible = set()
    for token in s1_tokens:
        if global_frequencies[token] <= RULE_F_LIMIT:
            possible.update(token_index.get((country, token), set()))

    additional = set()
    for entity_id in possible:
        shared = s1_tokens & entity_tokens[entity_id]
        if len(shared) == 1:
            additional.add(entity_id)
    return additional


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def address_words(value):
    return set(str(value).lower().split())


def collision_pattern(token, s1_address, target_address):
    """Describe the most apparent collision pattern from local address text."""
    if token.isdigit():
        return "common numeric address component"
    if token in COMMON_ADDRESS_TOKENS:
        return "generic address word"

    combined = f"{s1_address} {target_address}".lower()
    words = address_words(combined)
    if any(
        re.search(rf"\b{re.escape(marker)}\b[^,;]*\b{re.escape(token)}\b", combined)
        or re.search(rf"\b{re.escape(token)}\b[^,;]*\b{re.escape(marker)}\b", combined)
        for marker in STREET_MARKERS
    ):
        return "same street but unrelated business"
    if any(marker in words for marker in LANDMARK_MARKERS):
        return "same building/landmark but different business"
    if any(marker in words for marker in CITY_MARKERS):
        return "same city/town but unrelated business"
    return "other shared-location collision"


def print_stats(label, values):
    print(f"\n{label}")
    print(f"  total candidate pairs: {sum(values):,}")
    print(f"  mean:                  {sum(values) / len(values):,.2f}")
    print(f"  median:                {percentile(values, .50):,.1f}")
    print(f"  90th percentile:       {percentile(values, .90):,.1f}")
    print(f"  95th percentile:       {percentile(values, .95):,.1f}")
    print(f"  99th percentile:       {percentile(values, .99):,.1f}")
    print(f"  maximum:               {max(values):,}")


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

    print("\nBuilding current blocker indexes...")
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))
    address_index, entity_tokens = build_address_index(target_lookup)

    current_sizes = []
    rule_f_sizes = []
    additional_sizes = []
    additional_by_source = {"S2": [], "S3": []}
    additional_true_by_source = Counter()
    additional_false_by_source = Counter()
    false_candidates = []
    entity_new_candidates = 0
    entity_true_new = 0
    entity_only_false = 0
    entity_both = 0

    for s1_id in required_s1:
        s1_row = s1_lookup[s1_id]
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup[s1_id].split(",")
            if entity_id.strip()
        }
        current = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        rule_f = rule_f_candidates(
            s1_row, address_index, entity_tokens, global_frequencies
        )
        additional = rule_f - current
        current_sizes.append(len(current))
        rule_f_sizes.append(len(current | additional))
        additional_sizes.append(len(additional))

        true_additional = additional & true_ids
        false_additional = additional - true_ids
        if additional:
            entity_new_candidates += 1
        if true_additional:
            entity_true_new += 1
        if false_additional and not true_additional:
            entity_only_false += 1
        if true_additional and false_additional:
            entity_both += 1

        for candidate_id in additional:
            source = "S2" if candidate_id.startswith("S2-") else "S3"
            additional_by_source[source].append(candidate_id)
            if candidate_id in true_additional:
                additional_true_by_source[source] += 1
            else:
                additional_false_by_source[source] += 1
                candidate_row = target_lookup[candidate_id]
                s1_tokens = set(get_address_tokens(s1_row["business_address"]))
                candidate_tokens = set(get_address_tokens(candidate_row["business_address"]))
                shared_token = next(iter(s1_tokens & candidate_tokens))
                false_candidates.append(
                    {
                        "s1_id": s1_id,
                        "candidate_id": candidate_id,
                        "source": source,
                        "s1_name": s1_row["business_name"],
                        "candidate_name": candidate_row["business_name"],
                        "s1_address": s1_row["business_address"],
                        "candidate_address": candidate_row["business_address"],
                        "token": shared_token,
                        "frequency": global_frequencies[shared_token],
                        "pattern": collision_pattern(
                            shared_token,
                            s1_row["business_address"],
                            candidate_row["business_address"],
                        ),
                    }
                )

    total_additional = sum(additional_sizes)
    total_true_additional = sum(additional_true_by_source.values())
    total_false_additional = sum(additional_false_by_source.values())

    print("\n" + "=" * 88)
    print("CURRENT CANDIDATE STATISTICS")
    print("=" * 88)
    print_stats("CURRENT", current_sizes)
    print_stats("CURRENT + RULE F", rule_f_sizes)
    print_stats("ADDITIONAL CANDIDATES INTRODUCED BY RULE F", additional_sizes)

    print("\n" + "=" * 88)
    print("NEW-CANDIDATE PRECISION")
    print("=" * 88)
    print(f"Additional true matches:       {total_true_additional:,}")
    print(f"Additional false candidates:   {total_false_additional:,}")
    print(
        "Candidate precision:           "
        f"{total_true_additional / total_additional * 100:.2f}%"
    )
    print(
        "Percentage added candidates "
        f"that are true:                {total_true_additional / total_additional * 100:.2f}%"
    )

    print("\nBY SOURCE")
    print(f"{'Source':8} {'Added':>12} {'True':>12} {'False':>12} {'Precision':>12}")
    for source in ("S2", "S3"):
        added = len(additional_by_source[source])
        true_count = additional_true_by_source[source]
        false_count = additional_false_by_source[source]
        print(
            f"{source:8} {added:12,} {true_count:12,} {false_count:12,} "
            f"{true_count / added * 100:11.2f}%"
        )

    print("\nENTITY-LEVEL IMPACT")
    print(f"S1 entities receiving at least one new candidate:       {entity_new_candidates:,}")
    print(f"S1 entities receiving at least one new TRUE match:       {entity_true_new:,}")
    print(f"S1 entities receiving only FALSE additional candidates:  {entity_only_false:,}")
    print(f"S1 entities receiving both true and false additional:    {entity_both:,}")

    print("\nDANGEROUS COLLISION PATTERNS AMONG FALSE CANDIDATES")
    pattern_counts = Counter(candidate["pattern"] for candidate in false_candidates)
    for pattern, count in pattern_counts.most_common():
        print(f"{pattern:48} {count:8,}")

    print("\n" + "=" * 88)
    print("30 REPRESENTATIVE FALSE ADDITIONAL CANDIDATES")
    print("=" * 88)
    for number, candidate in enumerate(
        sorted(
            false_candidates,
            key=lambda item: (
                item["pattern"],
                item["frequency"],
                item["s1_id"],
                item["candidate_id"],
            ),
        )[:30],
        start=1,
    ):
        print(f"\n[{number}] {candidate['pattern']}")
        print(f"S1 ID:                 {candidate['s1_id']}")
        print(f"Candidate ID:          {candidate['candidate_id']}")
        print(f"Source:                {candidate['source']}")
        print(f"S1 business name:      {candidate['s1_name']}")
        print(f"Candidate business name:{candidate['candidate_name']}")
        print(f"S1 address:            {candidate['s1_address']}")
        print(f"Candidate address:     {candidate['candidate_address']}")
        print(f"Shared token:          {candidate['token']}")
        print(f"Token global frequency:{candidate['frequency']:,}")
        print(f"Collision pattern:      {candidate['pattern']}")

    print("\n" + "=" * 88)
    print("FACTUAL SUMMARY")
    print("=" * 88)
    print(f"Current candidate count:       {sum(current_sizes):,}")
    print(f"Rule F candidate count:         {sum(rule_f_sizes):,}")
    print(f"Additional candidates:          {total_additional:,}")
    print(f"Additional true matches:        {total_true_additional:,}")
    print(f"Additional false candidates:    {total_false_additional:,}")
    print(f"New-candidate precision:        {total_true_additional / total_additional * 100:.2f}%")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()