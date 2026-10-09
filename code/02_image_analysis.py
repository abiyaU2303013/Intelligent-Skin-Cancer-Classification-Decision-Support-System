import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Project folder
project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Paths
metadata_path = os.path.join(
    project_path,
    "HAM10000_metadata.csv"
)

image_folder_1 = os.path.join(
    project_path,
    "HAM10000_images_part_1"
)

image_folder_2 = os.path.join(
    project_path,
    "HAM10000_images_part_2"
)

# Load metadata
df = pd.read_csv(metadata_path)

print("=" * 50)
print("HAM10000 IMAGE ANALYSIS")
print("=" * 50)

# --------------------------------------------------
# 1. Dataset information
# --------------------------------------------------

print("\nTotal records:", len(df))

print("\nClasses:")
print(df["dx"].unique())

print("\nNumber of classes:", df["dx"].nunique())

# --------------------------------------------------
# 2. Class distribution
# --------------------------------------------------

class_counts = df["dx"].value_counts()

print("\nClass distribution:")
print(class_counts)

# --------------------------------------------------
# 3. Check image folders
# --------------------------------------------------

images_1 = [
    f for f in os.listdir(image_folder_1)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

images_2 = [
    f for f in os.listdir(image_folder_2)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("\nImages in part 1:", len(images_1))
print("Images in part 2:", len(images_2))
print("Total image files:", len(images_1) + len(images_2))

# --------------------------------------------------
# 4. Check whether metadata images exist
# --------------------------------------------------

image_ids = set(
    os.path.splitext(f)[0]
    for f in images_1 + images_2
)

metadata_ids = set(df["image_id"])

missing_images = metadata_ids - image_ids

print("\nImages in metadata:", len(metadata_ids))
print("Images found:", len(metadata_ids & image_ids))
print("Missing images:", len(missing_images))

# --------------------------------------------------
# 5. Plot class distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

class_counts.plot(kind="bar")

plt.title("HAM10000 Class Distribution")
plt.xlabel("Skin Lesion Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()

# --------------------------------------------------
# 6. Display one sample image from each class
# --------------------------------------------------

classes = sorted(df["dx"].unique())

plt.figure(figsize=(14, 8))

for i, class_name in enumerate(classes):

    sample = df[df["dx"] == class_name].iloc[0]

    image_id = sample["image_id"]

    image_path_1 = os.path.join(
        image_folder_1,
        image_id + ".jpg"
    )

    image_path_2 = os.path.join(
        image_folder_2,
        image_id + ".jpg"
    )

    if os.path.exists(image_path_1):
        image_path = image_path_1
    elif os.path.exists(image_path_2):
        image_path = image_path_2
    else:
        continue

    image = Image.open(image_path)

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()