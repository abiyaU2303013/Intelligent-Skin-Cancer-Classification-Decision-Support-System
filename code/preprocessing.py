import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight


# ==================================================
# 1. PROJECT PATH
# ==================================================

project_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

results_path = os.path.join(
    project_path,
    "results"
)


# ==================================================
# 2. LOAD TRAIN / VALIDATION / TEST DATA
# ==================================================

train_df = pd.read_csv(
    os.path.join(results_path, "train.csv")
)

validation_df = pd.read_csv(
    os.path.join(results_path, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(results_path, "test.csv")
)


print("=" * 60)
print("HAM10000 PYTORCH PREPROCESSING")
print("=" * 60)

print("\nDataset sizes:")
print("Training   :", len(train_df))
print("Validation :", len(validation_df))
print("Testing    :", len(test_df))


# ==================================================
# 3. CREATE NUMERIC LABELS
# ==================================================

class_mapping = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

train_df["label"] = train_df["dx"].map(class_mapping)
validation_df["label"] = validation_df["dx"].map(class_mapping)
test_df["label"] = test_df["dx"].map(class_mapping)


# Check for missing labels

if (
    train_df["label"].isnull().any()
    or validation_df["label"].isnull().any()
    or test_df["label"].isnull().any()
):

    print("\nERROR: Unknown class found.")

    print(
        train_df["dx"].unique()
    )

    raise ValueError(
        "One or more diagnosis classes could not be mapped."
    )


# Convert labels to integer

train_df["label"] = train_df["label"].astype(int)
validation_df["label"] = validation_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)


print("\nClass mapping:")

for name, number in class_mapping.items():

    print(
        f"{name} -> {number}"
    )


# ==================================================
# 4. SETTINGS
# ==================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 7


# ==================================================
# 5. IMAGE TRANSFORMS
# ==================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.9, 1.0)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# 6. FIND IMAGE PATH
# ==================================================

def find_image_path(image_id):

    part1 = os.path.join(
        project_path,
        "HAM10000_images_part_1",
        image_id + ".jpg"
    )

    part2 = os.path.join(
        project_path,
        "HAM10000_images_part_2",
        image_id + ".jpg"
    )

    if os.path.exists(part1):
        return part1

    if os.path.exists(part2):
        return part2

    return None


# Create image paths

train_df["image_path"] = train_df["image_id"].apply(
    find_image_path
)

validation_df["image_path"] = validation_df["image_id"].apply(
    find_image_path
)

test_df["image_path"] = test_df["image_id"].apply(
    find_image_path
)


# ==================================================
# 7. CHECK IMAGE PATHS
# ==================================================

missing_train = train_df["image_path"].isnull().sum()
missing_validation = validation_df["image_path"].isnull().sum()
missing_test = test_df["image_path"].isnull().sum()

print("\nMissing images:")

print(
    "Training   :",
    missing_train
)

print(
    "Validation :",
    missing_validation
)

print(
    "Testing    :",
    missing_test
)


if (
    missing_train > 0
    or missing_validation > 0
    or missing_test > 0
):

    raise FileNotFoundError(
        "Some HAM10000 images could not be found."
    )


# ==================================================
# 8. CUSTOM DATASET
# ==================================================

class HAM10000Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]

        label = int(row["label"])

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# ==================================================
# 9. CREATE DATASETS
# ==================================================

print("\nCreating datasets...")

train_dataset = HAM10000Dataset(
    train_df,
    transform=train_transform
)

validation_dataset = HAM10000Dataset(
    validation_df,
    transform=test_transform
)

test_dataset = HAM10000Dataset(
    test_df,
    transform=test_transform
)


# ==================================================
# 10. CREATE DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 11. CLASS WEIGHTS
# ==================================================

classes = np.arange(
    NUM_CLASSES
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].values
)

class_weights = torch.tensor(
    class_weights_array,
    dtype=torch.float32
)


print("\nClass weights:")
print("-" * 30)

for class_id, weight in enumerate(
    class_weights
):

    print(
        f"Class {class_id}: "
        f"{weight.item():.4f}"
    )


# ==================================================
# 12. SAVE CLASS WEIGHTS
# ==================================================

class_weights_df = pd.DataFrame({

    "label": classes,

    "weight": class_weights.numpy()

})

class_weights_df.to_csv(
    os.path.join(
        results_path,
        "class_weights.csv"
    ),
    index=False
)


# ==================================================
# 13. CHECK ONE BATCH
# ==================================================

print("\nChecking training batch...")

images, labels = next(
    iter(train_loader)
)

print(
    "Image batch shape:",
    images.shape
)

print(
    "Label batch shape:",
    labels.shape
)


# ==================================================
# 14. DEVICE
# ==================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU available:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device(
        "cpu"
    )

    print(
        "\nGPU not available."
    )

    print(
        "Training will use CPU."
    )


# ==================================================
# 15. FINAL STATUS
# ==================================================

print("\n" + "=" * 60)

print(
    "PREPROCESSING COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print("\nCompleted:")

print("1. 3000-image dataset loaded")
print("2. Train/validation/test data loaded")
print("3. 7 classes converted to numeric labels")
print("4. Images located")
print("5. Images resized to 224 x 224")
print("6. Images normalized")
print("7. Training augmentation applied")
print("8. Class weights calculated")
print("9. PyTorch DataLoaders created")

print(
    "\nReady for ResNet50 training."
)