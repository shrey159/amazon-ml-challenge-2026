"""Train a post-blocking matcher from an existing candidate-pairs TSV."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))

from matching_features import FEATURE_NAMES, feature_frame  # noqa: E402
from candidate_io import candidate_ids, load_required_records  # noqa: E402


DEFAULT_TRAIN_DIR = Path("dataset/train")
DEFAULT_CANDIDATE_PATH = Path("candidate_pairs.tsv")
DEFAULT_MODEL_PATH = Path("output/matcher.joblib")
CHUNK_SIZE = 100_000
NEGATIVE_RESERVOIR_SIZE = 600_000
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pairs", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def load_truth(path):
    ground_truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        row.source1_entity_id: {
            entity_id.strip()
            for entity_id in row.matched_entity_ids.split(",")
            if entity_id.strip()
        }
        for row in ground_truth.itertuples(index=False)
    }


def collect_pairs(args, s1_records, s2_records, s3_records, truth):
    """Label all rows first, then feature only positives and sampled negatives."""
    rng = random.Random(RANDOM_STATE)
    positive_pairs = []
    negative_pairs = []
    negative_seen = 0
    total_candidates = 0
    source_counts = defaultdict(int)

    for candidates in pd.read_csv(
        args.candidate_pairs,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
    ):
        total_candidates += len(candidates)
        for row in candidates.itertuples(index=False):
            source = str(row.source).upper()
            source_counts[source] += 1
            pair = (row.s1_id, row.candidate_id, source)
            if row.candidate_id in truth.get(row.s1_id, set()):
                positive_pairs.append(pair)
                continue
            negative_seen += 1
            if len(negative_pairs) < NEGATIVE_RESERVOIR_SIZE:
                negative_pairs.append(pair)
            else:
                replacement = rng.randrange(negative_seen)
                if replacement < NEGATIVE_RESERVOIR_SIZE:
                    negative_pairs[replacement] = pair

    selected_pairs = positive_pairs + negative_pairs
    selected_keys = {
        f"{s1_id}|{candidate_id}|{source}"
        for s1_id, candidate_id, source in selected_pairs
    }
    feature_chunks = []
    label_chunks = []
    for candidates in pd.read_csv(
        args.candidate_pairs,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
    ):
        row_keys = (
            candidates["s1_id"]
            + "|"
            + candidates["candidate_id"]
            + "|"
            + candidates["source"].str.upper()
        )
        selected = candidates[row_keys.isin(selected_keys)]
        if selected.empty:
            continue
        for source in ("S2", "S3"):
            source_selected = selected[selected["source"].str.upper() == source]
            if source_selected.empty:
                continue
            frame = feature_frame(source_selected, s1_records, s2_records, s3_records)
            feature_chunks.append(frame[FEATURE_NAMES].to_numpy(dtype=np.float32))
            label_chunks.append(
                np.asarray(
                    [
                        candidate_id in truth.get(s1_id, set())
                        for s1_id, candidate_id in zip(frame.s1_id, frame.candidate_id)
                    ],
                    dtype=np.int8,
                )
            )

    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    negative_count = len(labels) - int(labels.sum())
    negative_weight = negative_count / negative_seen if negative_seen else 1.0
    weights = np.ones(len(labels), dtype=np.float32)
    weights[labels == 0] = negative_weight
    return (
        features,
        labels,
        weights,
        total_candidates,
        source_counts,
        int(labels.sum()),
        negative_seen,
    )


def main():
    args = parse_args()
    if not args.candidate_pairs.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {args.candidate_pairs}. "
            "Run the existing blocker/export step first."
        )

    print("Reading candidate IDs and loading only referenced source records once...")
    s1_ids, s2_ids, s3_ids = candidate_ids(args.candidate_pairs, args.chunksize)
    s1_records = load_required_records(args.train_dir / "train_source1.tsv", s1_ids, args.chunksize)
    s2_records = load_required_records(args.train_dir / "train_source2.tsv", s2_ids, args.chunksize)
    s3_records = load_required_records(args.train_dir / "train_source3.tsv", s3_ids, args.chunksize)
    truth = load_truth(args.train_dir / "train_ground_truth.tsv")

    print("Extracting labels and features from candidate pairs in chunks...")
    (
        features,
        labels,
        weights,
        total_candidates,
        source_counts,
        positive_count,
        negative_seen,
    ) = collect_pairs(args, s1_records, s2_records, s3_records, truth)
    print(f"Candidate pairs processed: {total_candidates:,}")
    print(f"S2 candidate pairs: {source_counts['S2']:,}")
    print(f"S3 candidate pairs: {source_counts['S3']:,}")
    print(f"Positive pairs: {positive_count:,}")
    print(f"Negative pairs seen: {negative_seen:,}")
    print(f"Negative reservoir used: {len(labels) - positive_count:,}")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(features, labels, model__sample_weight=weights)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "candidate_columns": ["s1_id", "candidate_id", "source"],
        },
        args.model_out,
    )
    print(f"Model written: {args.model_out}")


if __name__ == "__main__":
    main()