"""
Shows preprocessing results for ONE skin lesion image:
  1) Prints a summary of the preprocessing steps applied.
  2) Saves a small before/after PNG (original vs resized+augmented).

Place this file in the SAME folder as preprocessing.py and run:
    python preprocessing_results.py
"""

import os
import random

import pandas as pd
from PIL import Image
from torchvision import transforms


# ==================================================
# 1. PATHS (same convention as preprocessing.py)
# ==================================================

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(project_path, "results")

train_df = pd.read_csv(os.path.join(results_path, "train.csv"))


# ==================================================
# 2. SETTINGS (match preprocessing.py)
# ==================================================

IMAGE_SIZE = 224

class_mapping = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6
}
inverse_mapping = {v: k for k, v in class_mapping.items()}

# The CSV only stores the "dx" text label (e.g. "nv"), not the numeric
# "label" column — that's created in-memory inside preprocessing.py, so we
# recreate it here the same way.
train_df["label"] = train_df["dx"].map(class_mapping)

# Same augmentation pipeline used for TRAINING images
# (Normalize/ToTensor skipped here so the image stays viewable)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
])


# ==================================================
# 3. FIND IMAGE PATH (same logic as preprocessing.py)
# ==================================================

def find_image_path(image_id):
    part1 = os.path.join(project_path, "HAM10000_images_part_1", image_id + ".jpg")
    part2 = os.path.join(project_path, "HAM10000_images_part_2", image_id + ".jpg")
    if os.path.exists(part1):
        return part1
    if os.path.exists(part2):
        return part2
    return None


# ==================================================
# 4. PRINT PREPROCESSING STEPS APPLIED
# ==================================================

def print_preprocessing_steps(row, original, after):
    print("=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)

    print(f"\nImage ID       : {row['image_id']}")
    print(f"Original size  : {original.size[0]} x {original.size[1]} px")
    print(f"Original mode  : {original.mode}")

    print("\nSteps applied:")
    print(f"  1. Resize                 -> {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("  2. Random Horizontal Flip  -> probability 0.5")
    print("  3. Random Rotation         -> up to 10 degrees")
    print("  4. Random Resized Crop     -> scale 0.9 - 1.0")
    print("  6. Normalize               -> mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]")
    print(f"\nFinal size     : {after.size[0]} x {after.size[1]} px")
    print("=" * 60)


# ==================================================
# 5. PICK ONE IMAGE, PROCESS, AND DISPLAY
# ==================================================

def show_one_before_after(seed=None, save_image=True):
    if seed is not None:
        random.seed(seed)

    row = train_df.sample(n=1).iloc[0]
    image_path = find_image_path(row["image_id"])

    original = Image.open(image_path).convert("RGB")
    after = train_transform(original)

    print_preprocessing_steps(row, original, after)

    if save_image:
        # Small side-by-side thumbnail, not a full matplotlib figure
        thumb_size = 150
        original_thumb = original.resize((thumb_size, thumb_size))
        after_thumb = after.resize((thumb_size, thumb_size))

        combined = Image.new("RGB", (thumb_size * 2 + 10, thumb_size), "white")
        combined.paste(original_thumb, (0, 0))
        combined.paste(after_thumb, (thumb_size + 10, 0))

        save_path = os.path.join(results_path, "before_after_small.png")
        combined.save(save_path)
        print(f"\nSmall before/after image saved to: {save_path}")


if __name__ == "__main__":
    show_one_before_after()