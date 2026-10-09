import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s
)

from preprocessing import train_loader, validation_loader



PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_PATH = os.path.join(
    PROJECT_PATH,
    "models"
)

MODEL_FILE = os.path.join(
    MODELS_PATH,
    "efficientnetv2b3_attention.pth"
)

BEST_MODEL_FILE = os.path.join(
    MODELS_PATH,
    "efficientnetv2b3_attention_best.pth"
)

CHECKPOINT_FILE = os.path.join(
    MODELS_PATH,
    "efficientnetv2s_checkpoint.pth"
)




NUM_CLASSES = 7

TOTAL_EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(
    MODELS_PATH,
    exist_ok=True
)

print("\n" + "=" * 70)
print("       EFFICIENTNETV2-S TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

print("Target total epochs:", TOTAL_EPOCHS)


# =========================================================
# 4. CREATE MODEL
# =========================================================

print("\nCreating EfficientNetV2-S...")

model = efficientnet_v2_s(
    weights=EfficientNet_V2_S_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# =========================================================
# 6. OPTIMIZER
# =========================================================

optimizer = optim.AdamW(

    [
        {
            "params": model.features.parameters(),
            "lr": 0.00005
        },

        {
            "params": model.classifier.parameters(),
            "lr": 0.001
        }
    ],

    weight_decay=1e-4
)


# =========================================================
# 7. SCHEDULER
# =========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


# =========================================================
# 8. VARIABLES
# =========================================================

start_epoch = 0

best_validation_accuracy = 0.0


# =========================================================
# 9. CHECK FOR EXISTING CHECKPOINT
# =========================================================

if os.path.exists(CHECKPOINT_FILE):

    print("\nExisting checkpoint found.")

    print(
        "Loading:",
        CHECKPOINT_FILE
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_validation_accuracy = checkpoint.get(
        "best_validation_accuracy",
        0.0
    )

    print(
        "\nContinuing training from epoch:",
        start_epoch + 1
    )

    print(
        "Previous best validation accuracy:",
        f"{best_validation_accuracy:.4f}"
    )

else:

    print("\nNo checkpoint found.")

    print(
        "Starting training from epoch 1."
    )


# =========================================================
# 10. TRAINING
# =========================================================

for epoch in range(
    start_epoch,
    TOTAL_EPOCHS
):

    current_epoch = epoch + 1

    print("\n")
    print("=" * 70)

    print(
        f"EPOCH {current_epoch}/{TOTAL_EPOCHS}"
    )

    print("=" * 70)


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    train_correct = 0

    train_total = 0

    train_loss = 0.0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
        )


        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()


        train_total += (
            labels.size(0)
        )


    train_accuracy = (
        train_correct /
        train_total
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    validation_correct = 0

    validation_total = 0

    validation_loss = 0.0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            validation_loss += (
                loss.item()
            )


            validation_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()


            validation_total += (
                labels.size(0)
            )


    validation_accuracy = (
        validation_correct /
        validation_total
    )


    avg_train_loss = (
        train_loss /
        len(train_loader)
    )


    avg_validation_loss = (
        validation_loss /
        len(validation_loader)
    )


    scheduler.step(
        validation_accuracy
    )

    print(
        f"\nTrain Loss       : "
        f"{avg_train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy:.4f} "
        f"({train_accuracy * 100:.2f}%)"
    )

    print(
        f"Validation Loss  : "
        f"{avg_validation_loss:.4f}"
    )

    print(
        f"Validation Acc.  : "
        f"{validation_accuracy:.4f} "
        f"({validation_accuracy * 100:.2f}%)"
    )

    print(
        f"Learning Rate    : "
        f"{optimizer.param_groups[0]['lr']:.7f}"
    )


    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            model.state_dict(),
            BEST_MODEL_FILE
        )

        print(
            "\n*** New best model saved ***"
        )


    torch.save(

        {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_validation_accuracy":
                best_validation_accuracy

        },

        CHECKPOINT_FILE
    )


    print(
        "Checkpoint saved."
    )


if os.path.exists(
    BEST_MODEL_FILE
):

    print(
        "\nLoading best model..."
    )

    best_state = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        best_state
    )


torch.save(
    model.state_dict(),
    MODEL_FILE
)

print("\n")
print("=" * 70)

print(
    "          TRAINING COMPLETED"
)

print("=" * 70)

print(
    "\nTotal epochs completed:",
    TOTAL_EPOCHS
)

print(
    "Best validation accuracy:",
    f"{best_validation_accuracy:.4f}"
)

print(
    "\nBest model:"
)

print(
    BEST_MODEL_FILE
)

print(
    "\nFinal model:"
)

print(
    MODEL_FILE
)

print(
    "\nCheckpoint:"
)

print(
    CHECKPOINT_FILE
)

print("\n" + "=" * 70)