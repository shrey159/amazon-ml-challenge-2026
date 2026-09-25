import pandas as pd

base = "dataset/train/"

s1 = pd.read_csv(
    base + "train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    base + "train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    base + "train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = pd.read_csv(
    base + "train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Pick 5 S1 records that actually have matches
samples = gt[gt["matched_entity_ids"] != ""].head(5)

for _, row in samples.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"].split(",")

    s1_row = s1[s1["entity_id"] == s1_id].iloc[0]

    print("\n" + "=" * 80)
    print("S1")
    print("=" * 80)
    print(s1_row.to_string())

    print("\nMATCHES")
    print("-" * 80)

    for entity_id in matched_ids:

        if entity_id.startswith("S2-"):
            result = s2[s2["entity_id"] == entity_id]
        else:
            result = s3[s3["entity_id"] == entity_id]

        if len(result) > 0:
            print(result.iloc[0].to_string())
            print("-" * 80)