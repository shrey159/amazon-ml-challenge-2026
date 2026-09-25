import pandas as pd

from blocking import (
    create_name_block_index,
    get_candidates
)


# --------------------------------------------------
# 1. Load S1 sample
# --------------------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5000
)


# --------------------------------------------------
# 2. Load S2 + S3 sample
# --------------------------------------------------

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=100000
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=100000
)


# --------------------------------------------------
# 3. Create blocking indexes
# --------------------------------------------------

s2_indexes = create_name_block_index(s2)
s3_indexes = create_name_block_index(s3)


# --------------------------------------------------
# 4. Count candidates for each S1
# --------------------------------------------------

candidate_counts = []


for _, row in s1.iterrows():

    candidates = get_candidates(
        row,
        s2_indexes,
        s3_indexes
    )

    candidate_counts.append(
        len(candidates)
    )


# --------------------------------------------------
# 5. Five-number summary
# --------------------------------------------------

series = pd.Series(candidate_counts)

print("\n========== CANDIDATE COUNT SUMMARY ==========")

print(
    "Min:",
    series.min()
)

print(
    "Q1:",
    series.quantile(0.25)
)

print(
    "Median:",
    series.median()
)

print(
    "Q3:",
    series.quantile(0.75)
)

print(
    "Max:",
    series.max()
)

print("==============================================")