"""Candidate-pair and source-record IO helpers."""

from pathlib import Path

import pandas as pd


def candidate_ids(path, chunksize=100_000):
    s1_ids = set()
    s2_ids = set()
    s3_ids = set()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["s1_id", "candidate_id", "source"],
        chunksize=chunksize,
    ):
        s1_ids.update(chunk["s1_id"])
        s2_ids.update(chunk.loc[chunk["source"].str.upper() == "S2", "candidate_id"])
        s3_ids.update(chunk.loc[chunk["source"].str.upper() == "S3", "candidate_id"])
    return s1_ids, s2_ids, s3_ids


def load_required_records(path, required_ids, chunksize=100_000):
    found = {}
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        matches = chunk[chunk["entity_id"].isin(required_ids)]
        for row in matches.itertuples(index=False):
            found[row.entity_id] = row._asdict()
        if len(found) == len(required_ids):
            break
    missing = required_ids - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing):,} records from {path}")
    return pd.DataFrame(found.values())