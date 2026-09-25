"""Stream candidate scores, evaluate entity-level F0.5, and write matches."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from candidate_io import candidate_ids, load_required_records  # noqa: E402
from matching_features import FEATURE_NAMES, feature_frame  # noqa: E402


THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("output/matcher.joblib"))
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--s3", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def load_truth(path, sample_size, random_state, selected_ids=None):
    if path is None:
        return None
    ground_truth = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if selected_ids is not None:
        ground_truth = ground_truth[
            ground_truth["source1_entity_id"].isin(selected_ids)
        ]
    elif sample_size:
        ground_truth = ground_truth.sample(sample_size, random_state=random_state)
    return {
        row.source1_entity_id: {
            entity_id.strip()
            for entity_id in row.matched_entity_ids.split(",")
            if entity_id.strip()
        }
        for row in ground_truth.itertuples(index=False)
    }


def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def score_chunk(candidates, model, s1_records, s2_records, s3_records):
    scores_by_s1 = defaultdict(list)
    for source in ("S2", "S3"):
        source_candidates = candidates[
            candidates["source"].str.upper() == source
        ]
        if source_candidates.empty:
            continue
        frame = feature_frame(source_candidates, s1_records, s2_records, s3_records)
        scores = model.predict_proba(
            frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
        )[:, 1]
        for s1_id, candidate_id, score in zip(frame.s1_id, frame.candidate_id, scores):
            scores_by_s1[s1_id].append((candidate_id, float(score)))
    return scores_by_s1


def selected_candidate_ids(path, selected_s1_ids, chunksize):
    selected_s2_ids = set()
    selected_s3_ids = set()
    for candidates in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        selected = candidates[candidates["s1_id"].isin(selected_s1_ids)]
        selected_s2_ids.update(
            selected.loc[selected["source"].str.upper() == "S2", "candidate_id"]
        )
        selected_s3_ids.update(
            selected.loc[selected["source"].str.upper() == "S3", "candidate_id"]
        )
    return selected_s2_ids, selected_s3_ids


def filter_candidate_file(path, selected_s1_ids, output_path, chunksize):
    """Materialize only selected S1 rows before any feature extraction."""
    selected_count = 0
    wrote_header = False
    for candidates in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        selected = candidates[candidates["s1_id"].isin(selected_s1_ids)]
        if selected.empty:
            continue
        selected.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
        )
        wrote_header = True
        selected_count += len(selected)
    if not wrote_header:
        pd.DataFrame(columns=["s1_id", "candidate_id", "source"]).to_csv(
            output_path, sep="\t", index=False
        )
    print(f"Fast validation candidate pairs retained: {selected_count:,}")


def entity_counts(scored_candidates, actual, threshold):
    predicted = {
        candidate_id
        for candidate_id, score in scored_candidates
        if score >= threshold
    }
    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    false_negative = len(actual - predicted)
    precision = true_positive / len(predicted) if predicted else (1.0 if not actual else 0.0)
    recall = true_positive / len(actual) if actual else (1.0 if not predicted else 0.0)
    return true_positive, false_positive, false_negative, f05(precision, recall), len(predicted)


def stream_entities(path, model, s1_records, s2_records, s3_records, chunksize, allowed_ids=None):
    """Yield grouped S1 scores while retaining only a cross-chunk final group."""
    pending_id = None
    pending_scores = []
    for candidates in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        if allowed_ids is not None:
            candidates = candidates[candidates["s1_id"].isin(allowed_ids)]
            if candidates.empty:
                continue
        chunk_scores = score_chunk(candidates, model, s1_records, s2_records, s3_records)
        last_id = str(candidates.iloc[-1]["s1_id"])
        for s1_id, scores in chunk_scores.items():
            if pending_id is not None and s1_id != pending_id:
                yield pending_id, pending_scores
                pending_id = None
                pending_scores = []
            if pending_id is None:
                pending_id = s1_id
            pending_scores.extend(scores)
        if pending_id is not None and pending_id != last_id:
            yield pending_id, pending_scores
            pending_id = None
            pending_scores = []
    if pending_id is not None:
        yield pending_id, pending_scores


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature names do not match current feature implementation")

    truth = load_truth(args.ground_truth, args.sample_size, args.random_state)
    temporary_directory = None
    if args.sample_size:
        if truth is not None:
            evaluation_ids = set(truth)
        else:
            candidate_s1_ids, _, _ = candidate_ids(args.candidate_pairs, args.chunksize)
            rng = random.Random(args.random_state)
            evaluation_ids = set(
                rng.sample(sorted(candidate_s1_ids), min(args.sample_size, len(candidate_s1_ids)))
            )
        temporary_directory = tempfile.TemporaryDirectory(prefix="matcher_fast_")
        candidate_path = Path(temporary_directory.name) / "candidate_pairs.tsv"
        filter_candidate_file(
            args.candidate_pairs, evaluation_ids, candidate_path, args.chunksize
        )
    else:
        candidate_path = args.candidate_pairs
        candidate_s1_ids, _, _ = candidate_ids(args.candidate_pairs, args.chunksize)
        evaluation_ids = set(candidate_s1_ids)
    _, candidate_s2_ids, candidate_s3_ids = candidate_ids(candidate_path, args.chunksize)
    if args.sample_size or truth is not None:
        s1_records = load_required_records(args.s1, evaluation_ids, args.chunksize)
    else:
        s1_records = pd.read_csv(args.s1, sep="\t", dtype=str, keep_default_na=False)
    s2_records = load_required_records(args.s2, candidate_s2_ids, args.chunksize)
    s3_records = load_required_records(args.s3, candidate_s3_ids, args.chunksize)

    candidate_count = 0
    for candidate_chunk in pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
    ):
        candidate_count += len(candidate_chunk)
    print(f"Test S1 rows: {len(s1_records):,}")
    print(f"Final candidate pairs: {candidate_count:,}")

    metrics = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "f05_sum": 0.0, "predicted": 0, "count": 0}
        for threshold in THRESHOLDS
    }
    seen_ids = set()
    print("Streaming candidate scores for threshold evaluation...")
    for s1_id, scores in stream_entities(
        candidate_path, model, s1_records, s2_records, s3_records, args.chunksize
    ):
        seen_ids.add(s1_id)
        actual = truth.get(s1_id, set()) if truth is not None else set()
        for threshold in THRESHOLDS:
            tp, fp, fn, score, predicted_count = entity_counts(scores, actual, threshold)
            result = metrics[threshold]
            result["tp"] += tp
            result["fp"] += fp
            result["fn"] += fn
            result["f05_sum"] += score
            result["predicted"] += bool(predicted_count)
            result["count"] += 1

    if truth is not None:
        for missing_id in evaluation_ids - seen_ids:
            for threshold in THRESHOLDS:
                tp, fp, fn, score, predicted_count = entity_counts([], truth[missing_id], threshold)
                result = metrics[threshold]
                result["tp"] += tp
                result["fp"] += fp
                result["fn"] += fn
                result["f05_sum"] += score
                result["predicted"] += bool(predicted_count)
                result["count"] += 1

        print("threshold\tTP\tFP\tFN\tprecision\trecall\tF0.5\tpredicted_S1\tavg_matches_per_S1")
        results = []
        for threshold in THRESHOLDS:
            result = metrics[threshold]
            precision = result["tp"] / (result["tp"] + result["fp"]) if result["tp"] + result["fp"] else 0.0
            recall = result["tp"] / (result["tp"] + result["fn"]) if result["tp"] + result["fn"] else 0.0
            average = (result["tp"] + result["fp"]) / result["count"] if result["count"] else 0.0
            row = {
                "threshold": threshold,
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "precision": precision,
                "recall": recall,
                "f05": result["f05_sum"] / result["count"] if result["count"] else 0.0,
                "predicted": result["predicted"],
                "average": average,
            }
            results.append(row)
            print(
                f"{threshold:.2f}\t{row['tp']}\t{row['fp']}\t{row['fn']}\t"
                f"{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f05']:.4f}\t"
                f"{row['predicted']}\t{row['average']:.4f}"
            )
        selected = args.threshold if args.threshold is not None else max(results, key=lambda row: row["f05"])["threshold"]
    else:
        selected = args.threshold if args.threshold is not None else 0.70
        print(f"No ground truth supplied; using threshold: {selected:.2f}")

    predictions = defaultdict(list)
    for s1_id, scores in stream_entities(
        candidate_path, model, s1_records, s2_records, s3_records, args.chunksize
    ):
        predictions[s1_id] = sorted(
            candidate_id for candidate_id, score in scores if score >= selected
        )

    output_rows = [
        {
            "s1_entity_id": s1_id,
            "matched_entity_ids": ",".join(predictions.get(s1_id, [])),
        }
        for s1_id in sorted(evaluation_ids)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, sep="\t", index=False)
    output_frame = pd.DataFrame(output_rows)
    predicted_pairs = [
        (row.s1_entity_id, candidate_id)
        for row in output_frame.itertuples(index=False)
        for candidate_id in row.matched_entity_ids.split(",")
        if candidate_id
    ]
    predicted_pair_set = set(predicted_pairs)
    for candidate_chunk in pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.chunksize,
    ):
        candidate_keys = set(zip(candidate_chunk["s1_id"], candidate_chunk["candidate_id"]))
        predicted_pair_set.difference_update(candidate_keys)
    invalid_predictions = predicted_pair_set
    empty_predictions = output_frame["matched_entity_ids"].isna().sum()
    duplicate_s1 = output_frame["s1_entity_id"].duplicated().sum()
    print(f"Candidates processed: {candidate_count:,}")
    print(f"Predicted matches: {len(predicted_pairs):,}")
    predicted_s1_count = sum(bool(value) for value in output_frame["matched_entity_ids"])
    print(f"S1 entities with predictions: {predicted_s1_count:,}")
    print(f"S1 entities with zero predictions: {sum(not bool(value) for value in output_frame['matched_entity_ids']):,}")
    print(f"Average predicted matches per S1: {len(predicted_pairs) / len(output_frame):.4f}")
    print("Verification:")
    print(f"  output row count matches test S1: {len(output_frame) == len(s1_records)}")
    print(f"  duplicate source1_entity_id count: {duplicate_s1}")
    print(f"  predictions outside candidate set: {len(invalid_predictions)}")
    print(f"  empty predictions containing NaN: {empty_predictions}")
    print(f"  TSV columns correct: {list(output_frame.columns) == ['s1_entity_id', 'matched_entity_ids']}")
    print(f"Selected validation threshold: {selected:.2f}")
    print(f"Matching results written: {args.output}")


if __name__ == "__main__":
    main()
