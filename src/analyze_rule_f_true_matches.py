"""Analyze only true matches added by hypothetical Rule F."""

from collections import Counter
import difflib
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import (  # noqa: E402
    COMMON_NAME_TOKENS,
    create_name_block_index,
    get_address_tokens,
    get_candidates,
    get_name_tokens,
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


def normalize_for_similarity(value):
    return " ".join(str(value).lower().split())


def edit_distance(left, right):
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


def name_signals(left, right):
    left_normalized = normalize_for_similarity(left)
    right_normalized = normalize_for_similarity(right)
    maximum_length = max(len(left_normalized), len(right_normalized))
    distance = edit_distance(left_normalized, right_normalized)
    return {
        "exact_normalized_name": left_normalized == right_normalized,
        "shared_name_tokens": set(get_name_tokens(left)) & set(get_name_tokens(right)),
        "character_similarity": difflib.SequenceMatcher(
            None, left_normalized, right_normalized
        ).ratio(),
        "edit_similarity": (
            1 - distance / maximum_length if maximum_length else 1.0
        ),
        "name_contains_other": (
            bool(left_normalized)
            and bool(right_normalized)
            and (left_normalized in right_normalized or right_normalized in left_normalized)
        ),
        "name_token_overlap": (
            len(set(get_name_tokens(left)) & set(get_name_tokens(right)))
            / min(len(set(get_name_tokens(left))), len(set(get_name_tokens(right))))
            if get_name_tokens(left) and get_name_tokens(right)
            else 0.0
        ),
    }


def script_names(value):
    scripts = set()
    for character in str(value):
        if not unicodedata.category(character).startswith(("L", "M")):
            continue
        character_name = unicodedata.name(character, "")
        for script in (
            "LATIN", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA",
            "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "ARABIC",
            "HEBREW", "CYRILLIC", "GREEK",
        ):
            if script in character_name:
                scripts.add(script.title())
                break
    return scripts


def cross_script_signal(s1_row, target_row):
    s1_scripts = script_names(
        f"{s1_row['business_name']} {s1_row['business_address']}"
    )
    target_scripts = script_names(
        f"{target_row['business_name']} {target_row['business_address']}"
    )
    return {
        "s1_scripts": s1_scripts,
        "target_scripts": target_scripts,
        "different_scripts": bool(s1_scripts and target_scripts and s1_scripts.isdisjoint(target_scripts)),
    }


def classify_pattern(record):
    similarity = record["character_similarity"]
    shared_names = record["shared_name_tokens"]
    cross_script = record["different_scripts"]
    name_contains = record["name_contains_other"]
    target_name = record["target_name"].lower()
    domain_name = target_name.endswith((".com", ".net", ".org")) or ".com" in target_name

    if cross_script:
        return "cross-script + 1 address token"
    if record["exact_normalized_name"]:
        return "exact/near-exact name + 1 address token"
    if domain_name and (similarity >= 0.65 or name_contains or shared_names):
        return "domain/DBA name + 1 address token"
    if shared_names and similarity >= 0.70:
        return "shared name token + 1 address token"
    if similarity >= 0.82 and name_contains:
        return "strong name similarity + 1 address token"
    if similarity >= 0.82:
        return "strong name similarity + 1 address token"
    if similarity >= 0.60 or name_contains:
        return "moderate name signal + 1 address token"
    if shared_names:
        return "shared name token + 1 address token"
    return "weak/no name signal"


def build_address_index(records):
    token_index = {}
    entity_tokens = {}
    for entity_id, row in records.items():
        country = str(row["country"]).strip().lower()
        tokens = set(get_address_tokens(row["business_address"]))
        entity_tokens[entity_id] = tokens
        for token in tokens:
            token_index.setdefault((country, token), set()).add(entity_id)
    return token_index, entity_tokens


def rule_f_candidates(s1_row, token_index, entity_tokens, global_frequencies):
    country = str(s1_row["country"]).strip().lower()
    s1_tokens = set(get_address_tokens(s1_row["business_address"]))
    possible = set()
    for token in s1_tokens:
        if global_frequencies[token] <= RULE_F_LIMIT:
            possible.update(token_index.get((country, token), set()))
    return {
        entity_id
        for entity_id in possible
        if len(s1_tokens & entity_tokens[entity_id]) == 1
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
    global_frequencies = count_global_address_tokens(
        [TRAIN_DIR / "train_source2.tsv", TRAIN_DIR / "train_source3.tsv"]
    )
    s2_indexes = create_name_block_index(pd.DataFrame(s2_lookup.values()))
    s3_indexes = create_name_block_index(pd.DataFrame(s3_lookup.values()))
    address_index, entity_tokens = build_address_index(target_lookup)

    recoveries = []
    for s1_id in required_s1:
        s1_row = s1_lookup[s1_id]
        true_ids = {
            entity_id.strip()
            for entity_id in truth_lookup[s1_id].split(",")
            if entity_id.strip()
        }
        current_candidates = set(get_candidates(s1_row, s2_indexes, s3_indexes))
        additional = rule_f_candidates(
            s1_row, address_index, entity_tokens, global_frequencies
        ) - current_candidates
        for target_id in sorted(additional & true_ids):
            target_row = target_lookup[target_id]
            s1_address_tokens = set(get_address_tokens(s1_row["business_address"]))
            target_address_tokens = set(get_address_tokens(target_row["business_address"]))
            shared_address_tokens = s1_address_tokens & target_address_tokens
            shared_numeric_tokens = {
                token for token in shared_address_tokens if token.isdigit()
            }
            names = name_signals(s1_row["business_name"], target_row["business_name"])
            scripts = cross_script_signal(s1_row, target_row)
            record = {
                "s1_id": s1_id,
                "target_id": target_id,
                "source": "S2" if target_id.startswith("S2-") else "S3",
                "s1_name": s1_row["business_name"],
                "target_name": target_row["business_name"],
                "s1_address": s1_row["business_address"],
                "target_address": target_row["business_address"],
                "shared_address_tokens": shared_address_tokens,
                "shared_numeric_tokens": shared_numeric_tokens,
                "shared_informative_address_count": len(shared_address_tokens),
                "rule_f_token": next(iter(shared_address_tokens)),
                "rule_f_frequency": global_frequencies[next(iter(shared_address_tokens))],
                "country": s1_row["country"],
                **names,
                **scripts,
            }
            record["rule_f_token_is_numeric"] = record["rule_f_token"].isdigit()
            record["pattern"] = classify_pattern(record)
            record["combined_name_address_signal"] = (
                "shared name token + address token" if record["shared_name_tokens"]
                else "name containment + address token" if record["name_contains_other"]
                else "name similarity + address token" if record["character_similarity"] >= 0.60
                else "address token only"
            )
            recoveries.append(record)

    recoveries.sort(key=lambda record: (record["s1_id"], record["target_id"]))
    pattern_counts = Counter(record["pattern"] for record in recoveries)

    print("\n" + "=" * 88)
    print("RULE F TRUE-RECOVERY GROUPED ANALYSIS")
    print("=" * 88)
    print(f"True Rule F recoveries: {len(recoveries):,}")
    print("\nPATTERN COUNTS")
    print(f"{'Pattern':55} {'Count':>8} {'Percentage':>12}")
    for pattern, count in pattern_counts.most_common():
        print(f"{pattern:55} {count:8,} {count / len(recoveries) * 100:11.2f}%")

    def count_and_percent(label, predicate):
        count = sum(predicate(record) for record in recoveries)
        print(f"{label:55} {count:8,} {count / len(recoveries) * 100:11.2f}%")

    print("\nADDITIONAL SIGNAL COUNTS")
    print(f"{'Signal':55} {'Count':>8} {'Percentage':>12}")
    count_and_percent("exact normalized name", lambda record: record["exact_normalized_name"])
    count_and_percent("shared useful name token", lambda record: bool(record["shared_name_tokens"]))
    count_and_percent("name containment", lambda record: record["name_contains_other"])
    count_and_percent("character similarity >= 0.82", lambda record: record["character_similarity"] >= 0.82)
    count_and_percent("character similarity >= 0.60", lambda record: record["character_similarity"] >= 0.60)
    count_and_percent("shared numeric address token", lambda record: bool(record["shared_numeric_tokens"]))
    count_and_percent("Rule F token is textual", lambda record: not record["rule_f_token_is_numeric"])
    count_and_percent("same country", lambda record: True)
    count_and_percent("different detected scripts", lambda record: record["different_scripts"])
    count_and_percent("shared name token + address token", lambda record: bool(record["shared_name_tokens"]))
    count_and_percent("name containment + address token", lambda record: record["name_contains_other"])
    count_and_percent("name similarity + address token", lambda record: record["character_similarity"] >= 0.60)
    count_and_percent("address token only", lambda record: record["character_similarity"] < 0.60 and not record["shared_name_tokens"] and not record["name_contains_other"])

    print("\n" + "=" * 88)
    print("ALL TRUE RULE F RECOVERIES")
    print("=" * 88)
    for number, record in enumerate(recoveries, start=1):
        print(f"\n[{number}] {record['pattern']}")
        print(f"S1 ID:                 {record['s1_id']}")
        print(f"True candidate ID:     {record['target_id']}")
        print(f"Source:                {record['source']}")
        print(f"S1 name:               {record['s1_name']}")
        print(f"Candidate name:        {record['target_name']}")
        print(f"S1 address:            {record['s1_address']}")
        print(f"Candidate address:     {record['target_address']}")
        print(f"Shared address token:  {record['rule_f_token']}")
        print(f"Token frequency:       {record['rule_f_frequency']:,}")
        print(f"Name similarity:       {record['character_similarity']:.3f}")
        print(f"Edit similarity:       {record['edit_similarity']:.3f}")
        print(f"Shared name tokens:    {sorted(record['shared_name_tokens'])}")
        print(f"Shared numeric tokens: {sorted(record['shared_numeric_tokens'])}")
        print(f"Country:               {record['country']}")
        print(f"S1 scripts:            {sorted(record['s1_scripts'])}")
        print(f"Candidate scripts:     {sorted(record['target_scripts'])}")
        print(f"Different scripts:     {record['different_scripts']}")
        print(f"Pattern:               {record['pattern']}")
        print(f"Combined signal:       {record['combined_name_address_signal']}")

    print("\n" + "=" * 88)
    print("FACTUAL SIGNAL SUMMARY")
    print("=" * 88)
    for label, predicate in (
        ("exact normalized name", lambda record: record["exact_normalized_name"]),
        ("shared useful name token", lambda record: bool(record["shared_name_tokens"])),
        ("name containment", lambda record: record["name_contains_other"]),
        ("character similarity >= 0.82", lambda record: record["character_similarity"] >= 0.82),
        ("shared numeric address token", lambda record: bool(record["shared_numeric_tokens"])),
        ("different detected scripts", lambda record: record["different_scripts"]),
        ("address token only", lambda record: record["combined_name_address_signal"] == "address token only"),
    ):
        count = sum(predicate(record) for record in recoveries)
        print(f"{label:35} {count:3,}/{len(recoveries):,} ({count / len(recoveries) * 100:.2f}%)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()