"""Export candidate pairs using the existing blocker without changing it."""

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from blocking import create_name_block_index, get_candidates  # noqa: E402
from candidate_io import load_required_records  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "dataset" / "train"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=TRAIN_DIR / "train_ground_truth.tsv")
    parser.add_argument("--s1", type=Path, default=TRAIN_DIR / "train_source1.tsv")
    parser.add_argument("--s2", type=Path, default=TRAIN_DIR / "train_source2.tsv")
    parser.add_argument("--s3", type=Path, default=TRAIN_DIR / "train_source3.tsv")
    parser.add_argument("--output", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_size:
        ground_truth = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        ground_truth = ground_truth.sample(args.sample_size, random_state=args.random_state)
        required_s1 = set(ground_truth["source1_entity_id"])
        required_s2 = set()
        required_s3 = set()
        for matched_ids in ground_truth["matched_entity_ids"]:
            for entity_id in matched_ids.split(","):
                entity_id = entity_id.strip()
                if entity_id.startswith("S2-"):
                    required_s2.add(entity_id)
                elif entity_id.startswith("S3-"):
                    required_s3.add(entity_id)
        s1_records = load_required_records(args.s1, required_s1)
        s2_records = load_required_records(args.s2, required_s2)
        s3_records = load_required_records(args.s3, required_s3)
    else:
        print("Loading full test sources for candidate generation...")
        s1_records = pd.read_csv(args.s1, sep="\t", dtype=str, keep_default_na=False)
        s2_records = pd.read_csv(args.s2, sep="\t", dtype=str, keep_default_na=False)
        s3_records = pd.read_csv(args.s3, sep="\t", dtype=str, keep_default_na=False)
    s2_indexes = create_name_block_index(s2_records)
    s3_indexes = create_name_block_index(s3_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        output.write("s1_id\tcandidate_id\tsource\n")
        pair_count = 0
        for row in s1_records.to_dict("records"):
            for candidate_id in get_candidates(row, s2_indexes, s3_indexes):
                source = "S2" if candidate_id.startswith("S2-") else "S3"
                output.write(f"{row['entity_id']}\t{candidate_id}\t{source}\n")
                pair_count += 1
    print(f"Candidate pairs written: {pair_count:,}")
    print(f"Candidate file: {args.output}")


if __name__ == "__main__":
    main()