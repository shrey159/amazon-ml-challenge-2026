"""Cached pairwise features for post-blocking entity matching."""

import difflib
from functools import lru_cache

import numpy as np
import pandas as pd

from blocking import get_address_tokens, get_name_tokens
from normalize import normalize_address, normalize_name


FEATURE_NAMES = [
    "exact_name",
    "name_character_similarity",
    "name_edit_similarity",
    "name_token_overlap_count",
    "name_token_jaccard",
    "name_length_difference",
    "name_contains",
    "exact_address",
    "address_character_similarity",
    "address_token_overlap_count",
    "address_token_jaccard",
    "numeric_token_overlap_count",
    "numeric_token_jaccard",
    "address_component_count",
    "address_contains",
    "same_country",
    "name_address_similarity_product",
    "strong_name_weak_address",
    "weak_name_strong_address",
    "strong_name_same_country",
]


def _edit_distance(left, right):
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


@lru_cache(maxsize=250_000)
def _similarity(left, right):
    return difflib.SequenceMatcher(None, left, right).ratio()


@lru_cache(maxsize=250_000)
def _edit_similarity(left, right):
    maximum_length = max(len(left), len(right))
    if not maximum_length:
        return 1.0
    return 1.0 - _edit_distance(left, right) / maximum_length


def prepare_record(row):
    """Precompute normalized strings and token sets once per source record."""
    name = row["business_name"]
    address = row["business_address"]
    name_tokens = frozenset(get_name_tokens(name))
    address_tokens = frozenset(get_address_tokens(address))
    numeric_tokens = frozenset(token for token in address_tokens if token.isdigit())
    return {
        "entity_id": row["entity_id"],
        "name": normalize_name(name),
        "address": normalize_address(address),
        "name_tokens": name_tokens,
        "address_tokens": address_tokens,
        "numeric_tokens": numeric_tokens,
        "country": str(row["country"]).strip().lower(),
    }


def pair_feature_vector(left, right):
    """Return numeric post-blocking features for one prepared pair."""
    shared_name = left["name_tokens"] & right["name_tokens"]
    shared_address = left["address_tokens"] & right["address_tokens"]
    shared_numeric = left["numeric_tokens"] & right["numeric_tokens"]
    name_union = left["name_tokens"] | right["name_tokens"]
    address_union = left["address_tokens"] | right["address_tokens"]
    numeric_union = left["numeric_tokens"] | right["numeric_tokens"]
    name_similarity = _similarity(left["name"], right["name"])
    address_similarity = _similarity(left["address"], right["address"])
    name_edit_similarity = _edit_similarity(left["name"], right["name"])
    same_country = left["country"] == right["country"]

    values = [
        float(left["name"] == right["name"]),
        name_similarity,
        name_edit_similarity,
        float(len(shared_name)),
        len(shared_name) / len(name_union) if name_union else 1.0,
        float(abs(len(left["name"]) - len(right["name"]))),
        float(bool(left["name"] and right["name"] and (left["name"] in right["name"] or right["name"] in left["name"]))),
        float(left["address"] == right["address"]),
        address_similarity,
        float(len(shared_address)),
        len(shared_address) / len(address_union) if address_union else 1.0,
        float(len(shared_numeric)),
        len(shared_numeric) / len(numeric_union) if numeric_union else 1.0,
        float(len(shared_address)),
        float(bool(left["address"] and right["address"] and (left["address"] in right["address"] or right["address"] in left["address"]))),
        float(same_country),
        name_similarity * address_similarity,
        float(name_similarity >= 0.85 and address_similarity < 0.60),
        float(name_similarity < 0.60 and address_similarity >= 0.80),
        float(name_similarity >= 0.85 and same_country),
    ]
    return np.asarray(values, dtype=np.float32)


def feature_frame(candidate_pairs, s1_records, s2_records, s3_records):
    """Bulk-join candidate IDs to records, then calculate pair features."""
    required_columns = {"s1_id", "candidate_id", "source"}
    missing = required_columns - set(candidate_pairs.columns)
    if missing:
        raise ValueError(f"Candidate file is missing columns: {sorted(missing)}")

    s1_columns = ["entity_id", "business_name", "business_address", "country"]
    target_columns = s1_columns
    left_records = s1_records[s1_columns].rename(
        columns={column: f"s1_{column}" for column in s1_columns}
    )
    frames = []
    for source, target_records in (("S2", s2_records), ("S3", s3_records)):
        source_pairs = candidate_pairs[
            candidate_pairs["source"].astype(str).str.upper() == source
        ].copy()
        if source_pairs.empty:
            continue
        target_frame = target_records[target_columns].rename(
            columns={column: f"target_{column}" for column in target_columns}
        )
        merged = source_pairs.merge(
            left_records,
            left_on="s1_id",
            right_on="s1_entity_id",
            how="left",
            validate="many_to_one",
        ).merge(
            target_frame,
            left_on="candidate_id",
            right_on="target_entity_id",
            how="left",
            validate="many_to_one",
        )
        if merged[["s1_entity_id", "target_entity_id"]].isna().any().any():
            raise ValueError(f"Unresolved {source} IDs in candidate pairs")
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "source", *FEATURE_NAMES])

    merged = pd.concat(frames, ignore_index=True)
    vectors = []
    prepared_left = {}
    prepared_right = {}
    for row in merged.itertuples(index=False):
        left = prepared_left.get(row.s1_entity_id)
        if left is None:
            left = prepare_record(
                {
                    "entity_id": row.s1_entity_id,
                    "business_name": row.s1_business_name,
                    "business_address": row.s1_business_address,
                    "country": row.s1_country,
                }
            )
            prepared_left[row.s1_entity_id] = left
        right = prepared_right.get(row.target_entity_id)
        if right is None:
            right = prepare_record(
                {
                    "entity_id": row.target_entity_id,
                    "business_name": row.target_business_name,
                    "business_address": row.target_business_address,
                    "country": row.target_country,
                }
            )
            prepared_right[row.target_entity_id] = right
        vectors.append(pair_feature_vector(left, right))
    feature_values = np.asarray(vectors, dtype=np.float32)
    result = merged[["s1_id", "candidate_id", "source"]].copy()
    for index, feature_name in enumerate(FEATURE_NAMES):
        result[feature_name] = feature_values[:, index]
    return result