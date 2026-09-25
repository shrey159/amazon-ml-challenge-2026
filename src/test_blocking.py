import pandas as pd

from normalize import normalize_name
from blocking import create_name_block_index, get_candidates


# Load only a small sample
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=1000
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=1000
)


# Create indexes
s2_index = create_name_block_index(s2)
s3_index = create_name_block_index(s3)


# Test first S1 record
row = s1.iloc[0]

print("Original name:")
print(row["business_name"])

print("\nNormalized name:")
print(normalize_name(row["business_name"]))

print("\nCountry:")
print(row["country"])


# Find candidates
candidates = get_candidates(
    row,
    s2_index,
    s3_index
)

print("\nCandidates:")
print(candidates)

print("\nNumber of candidates:")
print(len(candidates))