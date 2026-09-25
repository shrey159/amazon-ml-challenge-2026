import pandas as pd

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

def count_matches(x):
    if not x.strip():
        return 0
    return len(x.split(","))

gt["match_count"] = gt["matched_entity_ids"].apply(count_matches)

print("Total S1:", len(gt))

print("\nMatch count distribution:")
print(gt["match_count"].value_counts().sort_index())

print("\nBasic statistics:")
print(gt["match_count"].describe())

print("\nS1 with 0 matches:", (gt["match_count"] == 0).sum())
print("S1 with 1 match :", (gt["match_count"] == 1).sum())
print("S1 with 2+ matches:", (gt["match_count"] >= 2).sum())

print("\nTop match counts:")
print(gt["match_count"].value_counts().sort_index().tail(20))