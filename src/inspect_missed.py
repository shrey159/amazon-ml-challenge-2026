import pandas as pd


TARGET_S1 = "S1-966027321"
TARGET_S3 = "S3-472664232"


# --------------------------------------------------
# Load S1 record
# --------------------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)


# --------------------------------------------------
# Load S3 record
# --------------------------------------------------

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)


s1_row = s1[
    s1["entity_id"] == TARGET_S1
].iloc[0]


s3_row = s3[
    s3["entity_id"] == TARGET_S3
].iloc[0]


# --------------------------------------------------
# Print comparison
# --------------------------------------------------

print("\n========== S1 RECORD ==========")

print("Entity ID :", s1_row["entity_id"])
print("Name      :", s1_row["business_name"])
print("Address   :", s1_row["business_address"])
print("Country   :", s1_row["country"])


print("\n========== TRUE S3 MATCH ==========")

print("Entity ID :", s3_row["entity_id"])
print("Name      :", s3_row["business_name"])
print("Address   :", s3_row["business_address"])
print("Country   :", s3_row["country"])


print("\n========== COMPARISON ==========")

print("S1 Name    :", s1_row["business_name"])
print("S3 Name    :", s3_row["business_name"])

print()

print("S1 Address :", s1_row["business_address"])
print("S3 Address :", s3_row["business_address"])

print()

print("S1 Country :", s1_row["country"])
print("S3 Country :", s3_row["country"])