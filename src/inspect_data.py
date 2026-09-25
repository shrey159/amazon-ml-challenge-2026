import pandas as pd
from pathlib import Path

BASE = Path("dataset")

files = [
    BASE / "train" / "train_source1.tsv",
    BASE / "train" / "train_source2.tsv",
    BASE / "train" / "train_source3.tsv",
    BASE / "train" / "train_ground_truth.tsv",
    BASE / "test" / "test_source1.tsv",
    BASE / "test" / "test_source2.tsv",
    BASE / "test" / "test_source3.tsv",
]

for file in files:
    df = pd.read_csv(file, sep="\t")

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))