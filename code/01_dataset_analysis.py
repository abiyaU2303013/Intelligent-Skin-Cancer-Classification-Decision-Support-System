import pandas as pd
import os

# Current project folder
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Metadata file
metadata_path = os.path.join(project_path, "HAM10000_metadata.csv")

# Check file
if not os.path.exists(metadata_path):
    print("ERROR: Metadata file not found!")
    print("Path checked:", metadata_path)
    exit()

# Load metadata
df = pd.read_csv(metadata_path)

print("Dataset loaded successfully!")

print("\nNumber of records:", len(df))

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["dx"].value_counts())