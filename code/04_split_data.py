import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# PROJECT PATH
# ==================================================

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

results_path = os.path.join(
    project_path,
    "results"
)


# ==================================================
# LOAD 3000-IMAGE DATASET
# ==================================================

input_file = os.path.join(
    results_path,
    "small_dataset.csv"
)

df = pd.read_csv(input_file)

print("=" * 60)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print("\nTotal images:", len(df))


# ==================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ==================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["dx"],
    random_state=42
)


# ==================================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ==================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["dx"],
    random_state=42
)


# ==================================================
# DISPLAY SIZES
# ==================================================

print("\nDataset split:")

print("Training   :", len(train_df))
print("Validation :", len(validation_df))
print("Testing    :", len(test_df))


# ==================================================
# CHECK CLASS DISTRIBUTION
# ==================================================

print("\nTraining classes:")
print(train_df["dx"].value_counts())

print("\nValidation classes:")
print(validation_df["dx"].value_counts())

print("\nTesting classes:")
print(test_df["dx"].value_counts())


# ==================================================
# SAVE
# ==================================================

train_df.to_csv(
    os.path.join(
        results_path,
        "train.csv"
    ),
    index=False
)

validation_df.to_csv(
    os.path.join(
        results_path,
        "validation.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        results_path,
        "test.csv"
    ),
    index=False
)


# ==================================================
# FINAL
# ==================================================

print("\nFiles saved successfully:")

print("results/train.csv")
print("results/validation.csv")
print("results/test.csv")

print("\n" + "=" * 60)
print("DATA SPLITTING COMPLETED")
print("=" * 60)