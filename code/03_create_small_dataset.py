import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================
# PROJECT PATH
# ==================================================

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

metadata_path = os.path.join(
    project_path,
    "HAM10000_metadata.csv"
)

results_path = os.path.join(
    project_path,
    "results"
)

os.makedirs(results_path, exist_ok=True)


# ==================================================
# SETTINGS
# ==================================================

TARGET_IMAGES = 3000
RANDOM_STATE = 42


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 60)
print("CREATING 3000-IMAGE HAM10000 DATASET")
print("=" * 60)

df = pd.read_csv(metadata_path)

print("\nOriginal dataset:", len(df))


# ==================================================
# ORIGINAL DISTRIBUTION
# ==================================================

print("\nOriginal class distribution:")
print(df["dx"].value_counts())


# ==================================================
# STRATIFIED SELECTION
# ==================================================

small_df, _ = train_test_split(
    df,
    train_size=TARGET_IMAGES,
    stratify=df["dx"],
    random_state=RANDOM_STATE
)


# ==================================================
# SHUFFLE
# ==================================================

small_df = small_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ==================================================
# CHECK
# ==================================================

print("\nSelected images:", len(small_df))

print("\nSelected class distribution:")
print(small_df["dx"].value_counts())

print("\nClass percentages:")
print(
    (small_df["dx"].value_counts(normalize=True) * 100).round(2)
)


# ==================================================
# CHECK 7 CLASSES
# ==================================================

expected_classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

if sorted(small_df["dx"].unique()) == sorted(expected_classes):
    print("\nAll 7 classes are present.")
else:
    print("\nWARNING: Not all 7 classes are present.")


# ==================================================
# SAVE
# ==================================================

output_file = os.path.join(
    results_path,
    "small_dataset.csv"
)

small_df.to_csv(
    output_file,
    index=False
)

print("\nSaved successfully:")
print(output_file)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)