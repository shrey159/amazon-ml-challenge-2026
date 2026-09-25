"""Analyze true and false candidates produced by selective Rule 3."""

from collections import Counter, defaultdict
import difflib
from pathlib import Path
import statistics
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import create_name_block_index, get_address_tokens, get_candidates, get_name_tokens  # noqa: E402
from analyze_rule_f_recoveries import count_global_address_tokens, load_required_records  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42
RULE_F_LIMIT = 5_000
RULE_3_THRESHOLD = 0.80
CHUNK_SIZE = 200_000


def normalized(value):
    return " ".join(str(value).lower().split())


def similarity(left, right):
    return difflib.SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def containment(left, right):
    left = normalized(left)
    right = normalized(right)
    return bool(left and right and (left in right or right in left))


def address_parts(address):
    """Extract only direct trailing comma-separated city/state components."""
    parts = [normalized(part) for part in str(address).split(",") if normalized(part)]
    if len(parts) < 2:
        return "", ""
    return parts[-2], parts[-1]


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values):,}, min={min(values):.3f}, median={statistics.median(values):.3f}, "
        f"p90={percentile(values, .90):.3f}, max={max(values):.3f}"
    )


def build_address_index(records):
    token_index = defaultdict(set)
    entity_tokens = {}
    for entity_id, row in records.items():
        country = str(row["country"]).strip().lower()
        tokens = set(get_address_tokens(row["business_address"]))
        entity_tokens[entity_id] = tokens
        for token in tokens:
            token_index[(country, token)].add(entity_id)
    return token_index, entity_tokens


def rule_f_candidates(s1_row, token_index, entity_tokens, frequencies):
    country = str(s1_row["country"]).strip().lower()
    s1_tokens = set(get_address_tokens(s1_row["business_address"]))
    possible = set()
    for token in s1_tokens:
        if frequencies[token] <= RULE_F_LIMIT:
            possible.update(token_index.get((country, token), set()))
    return {
        entity_id
        for entity_id in possible
        if len(s1_tokens & entity_tokens[entity_id]) == 1
    }


def collision_reason(record):
    token = record["rule3_token"]
    if token.isdigit():
        return "numeric address collision"
    if record["same_city"]:
        return "same direct city component"
    if record["same_state_region"]:
        return "same direct state/region component"
    if record["address_similarity"] >= 0.80:
        return "high address similarity"
    return "single textual address-token collision"


def make_record(s1_id, s1_row, candidate_id, target_row, frequencies, true):
    s1_name_tokens = set(get_name_tokens(s1_row["business_name"]))
    target_name_tokens = set(get_name_tokens(target_row["business_name"]))
    s1_address_tokens = set(get_address_tokens(s1_row["business_address"]))
    target_address_tokens = set(get_address_tokens(target_row["business_address"]))
    shared_address_tokens = s1_address_tokens & target_address_tokens
    shared_numeric_tokens = {token for token in shared_address_tokens if token.isdigit()}
    s1_city, s1_state = address_parts(s1_row["business_address"])
    target_city, target_state = address_parts(target_row["business_address"])
    record = {
        "s1_id": s1_id,
        "candidate_id": candidate_id,
        "source": "S2" if candidate_id.startswith("S2-") else "S3",
        "true": true,
        "s1_name": s1_row["business_name"],
        "candidate_name": target_row["business_name"],
        "s1_address": s1_row["business_address"],
        "candidate_address": target_row["business_address"],
        "name_similarity": similarity(s1_row["business_name"], target_row["business_name"]),
        "exact_normalized_name": normalized(s1_row["business_name"]) == normalized(target_row["business_name"]),
        "name_containment": containment(s1_row["business_name"], target_row["business_name"]),
        "shared_name_tokens": s1_name_tokens & target_name_tokens,
        "name_token_overlap": (
            len(s1_name_tokens & target_name_tokens) / min(len(s1_name_tokens), len(target_name_tokens))
            if s1_name_tokens and target_name_tokens
            else 0.0
        ),
        "name_length_difference": abs(
            len(normalized(s1_row["business_name"]))
            - len(normalized(target_row["business_name"]))
        ),
        "shared_address_tokens": shared_address_tokens,
        "shared_numeric_tokens": shared_numeric_tokens,
        "shared_informative_address_count": len(shared_address_tokens),
        "rule3_token": next(iter(shared_address_tokens)),
        "rule3_frequency": frequencies[next(iter(shared_address_tokens))],
        "rule3_token_is_numeric": next(iter(shared_address_tokens)).isdigit(),
        "same_country": str(s1_row["country"]).strip().lower() == str(target_row["country"]).strip().lower(),
        "country": s1_row["country"],
        "s1_city": s1_city,
        "candidate_city": target_city,
        "same_city": bool(s1_city and target_city and s1_city == target_city),
        "s1_state_region": s1_state,
        "candidate_state_region": target_state,
        "same_state_region": bool(s1_state and target_state and s1_state == target_state),
        "address_similarity": similarity(s1_row["business_address"], target_row["business_address"]),
    }
    record["collision_reason"] = collision_reason(record)
    return record


def count_true_false(records, predicate):
    true_count = sum(predicate(record) for record in records if record["true"])
    false_count = sum(predicate(record) for record in records if not record["true"])
    return true_count, false_count


def print_feature_comparison(records):
    true_records = [record for record in records if record["true"]]
    false_records = [record for record in records if not record["true"]]
    print("\nFEATURE COMPARISON: TRUE VS FALSE")
    print("Boolean and set features show count; numeric features show distribution.")

    boolean_features = [
        ("exact normalized name", lambda record: record["exact_normalized_name"]),
        ("name containment", lambda record: record["name_containment"]),
        ("shared useful name token", lambda record: bool(record["shared_name_tokens"])),
        ("numeric Rule-3 token", lambda record: record["rule3_token_is_numeric"]),
        ("textual Rule-3 token", lambda record: not record["rule3_token_is_numeric"]),
        ("same country", lambda record: record["same_country"]),
        ("same city", lambda record: record["same_city"]),
        ("same state/region", lambda record: record["same_state_region"]),
    ]
    print(f"{'Boolean feature':35} {'TRUE':>10} {'FALSE':>10}")
    for label, predicate in boolean_features:
        true_count, false_count = count_true_false(records, predicate)
        print(f"{label:35} {true_count:10,} {false_count:10,}")

    numeric_features = [
        ("name similarity", lambda record: record["name_similarity"]),
        ("normalized name token overlap", lambda record: record["name_token_overlap"]),
        ("name length difference", lambda record: record["name_length_difference"]),
        ("shared numeric token count", lambda record: len(record["shared_numeric_tokens"])),
        ("shared informative address count", lambda record: record["shared_informative_address_count"]),
        ("Rule-3 token frequency", lambda record: record["rule3_frequency"]),
        ("address similarity", lambda record: record["address_similarity"]),
    ]
    print("\nNumeric feature distributions")
    print(f"{'Numeric feature':35} {'TRUE':65} {'FALSE':65}")
    for label, getter in numeric_features:
        print(
            f"{label:35} {distribution([getter(record) for record in true_records]):65} "
            f"{distribution([getter(record) for record in false_records]):65}"
        )


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
    frequencies = count_global_address_tokens(
        [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
    )
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))
    address_index, entity_tokens = build_address_index(target_lookup)

    candidates = []
    for s1_id in required_s1:
        s1_row = s1_lookup[s1_id]
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup[s1_id].split(",")
            if entity_id.strip()
        }
        current = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        rule_f = rule_f_candidates(s1_row, address_index, entity_tokens, frequencies)
        for candidate_id in sorted((rule_f - current)):
            target_row = target_lookup[candidate_id]
            record = make_record(
                s1_id,
                s1_row,
                candidate_id,
                target_row,
                frequencies,
                candidate_id in true_ids,
            )
            if record["name_similarity"] >= RULE_3_THRESHOLD:
                candidates.append(record)

    true_records = [record for record in candidates if record["true"]]
    false_records = [record for record in candidates if not record["true"]]

    print("\n" + "=" * 100)
    print("RULE 3 FALSE-CASE ANALYSIS")
    print("=" * 100)
    print(f"Rule 3 candidates: {len(candidates):,}")
    print(f"True candidates:   {len(true_records):,}")
    print(f"False candidates:  {len(false_records):,}")
    print_feature_comparison(candidates)

    conditions = {
        "name similarity >= 0.85": lambda record: record["name_similarity"] >= 0.85,
        "name similarity >= 0.90": lambda record: record["name_similarity"] >= 0.90,
        "shared name token": lambda record: bool(record["shared_name_tokens"]),
        "name containment": lambda record: record["name_containment"],
        "textual Rule-3 token": lambda record: not record["rule3_token_is_numeric"],
        "numeric Rule-3 token": lambda record: record["rule3_token_is_numeric"],
        ">=2 shared numeric address tokens": lambda record: len(record["shared_numeric_tokens"]) >= 2,
        "address similarity >= 0.60": lambda record: record["address_similarity"] >= 0.60,
        "address similarity >= 0.70": lambda record: record["address_similarity"] >= 0.70,
        "address similarity >= 0.80": lambda record: record["address_similarity"] >= 0.80,
        "same country + name similarity >= 0.80": lambda record: record["same_country"] and record["name_similarity"] >= 0.80,
        "name similarity >= 0.80 AND textual token": lambda record: record["name_similarity"] >= 0.80 and not record["rule3_token_is_numeric"],
        "name similarity >= 0.85 AND textual token": lambda record: record["name_similarity"] >= 0.85 and not record["rule3_token_is_numeric"],
        "name similarity >= 0.90 AND textual token": lambda record: record["name_similarity"] >= 0.90 and not record["rule3_token_is_numeric"],
    }
    print("\nANALYTICAL CONDITION TESTS")
    print(f"{'Condition':48} {'Retained':>10} {'True':>8} {'False':>8} {'Precision':>10} {'True recall':>12}")
    for label, predicate in conditions.items():
        retained = [record for record in candidates if predicate(record)]
        retained_true = sum(record["true"] for record in retained)
        retained_false = len(retained) - retained_true
        precision = retained_true / len(retained) * 100 if retained else 0
        recall = retained_true / len(true_records) * 100 if true_records else 0
        print(
            f"{label:48} {len(retained):10,} {retained_true:8,} {retained_false:8,} "
            f"{precision:9.2f}% {recall:11.2f}%"
        )

    print("\nFALSE CANDIDATES AND DETECTED COLLISION REASONS")
    reason_counts = Counter(record["collision_reason"] for record in false_records)
    for reason, count in reason_counts.most_common():
        print(f"{reason:45} {count:8,}")

    print("\nALL 16 FALSE RULE-3 CANDIDATES")
    print("=" * 100)
    for number, record in enumerate(sorted(false_records, key=lambda item: (item["s1_id"], item["candidate_id"])), start=1):
        print(f"\n[{number}] {record['collision_reason']}")
        print(f"S1 ID:                 {record['s1_id']}")
        print(f"Candidate ID:          {record['candidate_id']}")
        print(f"Source:                {record['source']}")
        print(f"S1 name:               {record['s1_name']}")
        print(f"Candidate name:        {record['candidate_name']}")
        print(f"S1 address:            {record['s1_address']}")
        print(f"Candidate address:     {record['candidate_address']}")
        print(f"Name similarity:       {record['name_similarity']:.3f}")
        print(f"Shared address token:  {record['rule3_token']}")
        print(f"Token frequency:       {record['rule3_frequency']:,}")
        print(f"Shared name tokens:    {sorted(record['shared_name_tokens'])}")
        print(f"Address similarity:    {record['address_similarity']:.3f}")
        print(f"Collision reason:      {record['collision_reason']}")

    print("\nCONDITION IMPACT TABLE")
    print(f"{'Condition':48} {'False reduced':>14} {'True retained':>14} {'False retained':>15}")
    for label, predicate in conditions.items():
        retained = [record for record in candidates if predicate(record)]
        retained_true = sum(record["true"] for record in retained)
        retained_false = len(retained) - retained_true
        print(f"{label:48} {len(false_records) - retained_false:14,} {retained_true:14,} {retained_false:15,}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()