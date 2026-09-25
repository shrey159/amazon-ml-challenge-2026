"""Compare selective one-address-token rules against the current blocker."""

from collections import defaultdict
from pathlib import Path
import difflib
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import create_name_block_index, get_address_tokens, get_candidates  # noqa: E402
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
FALSE_EXAMPLE_LIMIT = 20


def normalized_name(value):
    return " ".join(str(value).lower().split())


def character_similarity(left, right):
    return difflib.SequenceMatcher(
        None,
        normalized_name(left),
        normalized_name(right),
    ).ratio()


def name_contains(left, right):
    left = normalized_name(left)
    right = normalized_name(right)
    return bool(left and right and (left in right or right in left))


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


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


def candidate_record(s1_id, s1_row, candidate_id, target_lookup, frequencies):
    target_row = target_lookup[candidate_id]
    shared = set(get_address_tokens(s1_row["business_address"])) & set(
        get_address_tokens(target_row["business_address"])
    )
    token = next(iter(shared))
    return {
        "s1_id": s1_id,
        "candidate_id": candidate_id,
        "source": "S2" if candidate_id.startswith("S2-") else "S3",
        "s1_name": s1_row["business_name"],
        "candidate_name": target_row["business_name"],
        "s1_address": s1_row["business_address"],
        "candidate_address": target_row["business_address"],
        "token": token,
        "frequency": frequencies[token],
        "name_similarity": character_similarity(
            s1_row["business_name"], target_row["business_name"]
        ),
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

    baseline_candidates = {}
    rule_f_candidates_by_s1 = {}
    known_true_rule_f = set()
    for s1_id in required_s1:
        s1_row = s1_lookup[s1_id]
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup[s1_id].split(",")
            if entity_id.strip()
        }
        baseline = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        rule_f = rule_f_candidates(s1_row, address_index, entity_tokens, frequencies)
        baseline_candidates[s1_id] = baseline
        rule_f_candidates_by_s1[s1_id] = rule_f
        known_true_rule_f.update((s1_id, entity_id) for entity_id in (rule_f - baseline) & true_ids)

    rules = {
        "RULE 1: similarity >= 0.60": lambda similarity, containment, textual: similarity >= 0.60,
        "RULE 2: similarity >= 0.70": lambda similarity, containment, textual: similarity >= 0.70,
        "RULE 3: similarity >= 0.80": lambda similarity, containment, textual: similarity >= 0.80,
        "RULE 4: name containment": lambda similarity, containment, textual: containment,
        "RULE 5: similarity >= 0.60 OR containment": lambda similarity, containment, textual: similarity >= 0.60 or containment,
        "RULE 6: textual token AND similarity >= 0.60": lambda similarity, containment, textual: textual and similarity >= 0.60,
    }
    results = {}

    for rule_name, qualifies in rules.items():
        additions_by_s1 = defaultdict(set)
        false_examples = []
        added_true = set()
        added_by_source = {"S2": 0, "S3": 0}
        true_by_source = {"S2": 0, "S3": 0}
        false_by_source = {"S2": 0, "S3": 0}

        for s1_id in required_s1:
            s1_row = s1_lookup[s1_id]
            true_ids = {
                entity_id.strip()
                for entity_id in truth_lookup[s1_id].split(",")
                if entity_id.strip()
            }
            baseline = baseline_candidates[s1_id]
            for candidate_id in rule_f_candidates_by_s1[s1_id] - baseline:
                candidate = candidate_record(
                    s1_id, s1_row, candidate_id, target_lookup, frequencies
                )
                textual = not candidate["token"].isdigit()
                containment = name_contains(s1_row["business_name"], candidate["candidate_name"])
                if not qualifies(candidate["name_similarity"], containment, textual):
                    continue

                additions_by_s1[s1_id].add(candidate_id)
                source = candidate["source"]
                added_by_source[source] += 1
                if candidate_id in true_ids:
                    added_true.add((s1_id, candidate_id))
                    true_by_source[source] += 1
                else:
                    false_by_source[source] += 1
                    if len(false_examples) < FALSE_EXAMPLE_LIMIT:
                        false_examples.append(candidate)

        counts = [len(additions_by_s1.get(s1_id, set())) for s1_id in required_s1]
        affected_counts = [count for count in counts if count > 0]
        total_added = sum(counts)
        total_true = len(added_true)
        total_false = total_added - total_true
        results[rule_name] = {
            "total_added": total_added,
            "total_true": total_true,
            "total_false": total_false,
            "precision": total_true / total_added * 100 if total_added else 0,
            "recall": total_true / len(known_true_rule_f) * 100,
            "affected": len(affected_counts),
            "median": percentile(affected_counts, 0.50),
            "p90": percentile(affected_counts, 0.90),
            "maximum": max(affected_counts) if affected_counts else 0,
            "added_by_source": added_by_source,
            "true_by_source": true_by_source,
            "false_by_source": false_by_source,
            "false_examples": sorted(
                false_examples,
                key=lambda item: (item["s1_id"], item["candidate_id"]),
            ),
        }

    print("\n" + "=" * 110)
    print("SELECTIVE ONE-TOKEN RULE COMPARISON")
    print("=" * 110)
    print(f"Known true Rule F recoveries: {len(known_true_rule_f):,}")
    print("Each rule is evaluated independently against the current blocker baseline.")
    print(
        f"{'Rule':42} {'Added':>10} {'True':>8} {'False':>8} {'Precision':>10} "
        f"{'Recall':>9} {'Affected':>10} {'Median':>9} {'P90':>9} {'Max':>8}"
    )
    for rule_name, result in results.items():
        print(
            f"{rule_name:42} {result['total_added']:10,} {result['total_true']:8,} "
            f"{result['total_false']:8,} {result['precision']:9.2f}% "
            f"{result['recall']:8.2f}% {result['affected']:10,} "
            f"{result['median']:9.1f} {result['p90']:9.1f} {result['maximum']:8,}"
        )
        print("  Source      Added       True      False     Precision")
        for source in ("S2", "S3"):
            added = result["added_by_source"][source]
            true_count = result["true_by_source"][source]
            false_count = result["false_by_source"][source]
            precision = true_count / added * 100 if added else 0
            print(
                f"  {source:8} {added:10,} {true_count:10,} {false_count:10,} "
                f"{precision:10.2f}%"
            )

        print("  FALSE ADDITIONAL CANDIDATES (up to 20)")
        for number, candidate in enumerate(result["false_examples"], start=1):
            print(f"  [{number}]")
            print(f"  S1 ID:                 {candidate['s1_id']}")
            print(f"  Candidate ID:          {candidate['candidate_id']}")
            print(f"  Source:                {candidate['source']}")
            print(f"  S1 name:               {candidate['s1_name']}")
            print(f"  Candidate name:        {candidate['candidate_name']}")
            print(f"  S1 address:            {candidate['s1_address']}")
            print(f"  Candidate address:     {candidate['candidate_address']}")
            print(f"  Shared address token:  {candidate['token']}")
            print(f"  Token frequency:       {candidate['frequency']:,}")
            print(f"  Name similarity:       {candidate['name_similarity']:.3f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()