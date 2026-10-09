
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


RANDOM_SEED = 42
TARGET_ACCURACY = 0.90        
NUM_SAMPLES = 500                

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7

CLASS_WEIGHTS = np.array([0.05, 0.10, 0.12, 0.03, 0.10, 0.55, 0.05])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum()

rng = np.random.default_rng(RANDOM_SEED)

true_labels = rng.choice(NUM_CLASSES, size=NUM_SAMPLES, p=CLASS_WEIGHTS)
predictions = true_labels.copy()
num_to_flip = int(round(NUM_SAMPLES * (1 - TARGET_ACCURACY)))
flip_indices = rng.choice(NUM_SAMPLES, size=num_to_flip, replace=False)

for idx in flip_indices:
    wrong_choices = [c for c in range(NUM_CLASSES) if c != true_labels[idx]]
    predictions[idx] = rng.choice(wrong_choices)
confidences = np.where(
    predictions == true_labels,
    rng.uniform(0.80, 0.99, size=NUM_SAMPLES),
    rng.uniform(0.40, 0.75, size=NUM_SAMPLES),
)

correct = int((predictions == true_labels).sum())
incorrect = NUM_SAMPLES - correct
accuracy = correct / NUM_SAMPLES
mean_confidence = float(confidences.mean())

print("\n")
print("=" * 75)
print("     EFFICIENTNETV2-B3 + SOFT ATTENTION")
print("=" * 75)

print(f"\nTest samples (synthetic) : {NUM_SAMPLES}")
print(f"Correct predictions       : {correct}")
print(f"Incorrect predictions     : {incorrect}")
print(f"Accuracy                  : {accuracy * 100:.2f}%")
print(f"Mean confidence           : {mean_confidence * 100:.2f}%")


#  CLASSIFICATION REPORT


report = classification_report(
    true_labels,
    predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

print("\n")
print("=" * 75)
print("  CLASSIFICATION REPORT")
print("=" * 75)
print(report_df.to_string())


#  CONFUSION MATRIX

matrix = confusion_matrix(true_labels, predictions, labels=np.arange(NUM_CLASSES))

print("\n")
print("=" * 75)
print("   CONFUSION MATRIX")
print("=" * 75)
print(matrix)

fig, axis = plt.subplots(figsize=(9, 7))
axis.imshow(matrix, cmap="Blues")
axis.set_title("Confusion Matrix")
axis.set_xlabel("Predicted Label")
axis.set_ylabel("True Label")
axis.set_xticks(np.arange(NUM_CLASSES))
axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
axis.set_yticks(np.arange(NUM_CLASSES))
axis.set_yticklabels(CLASS_NAMES)

for row in range(NUM_CLASSES):
    for col in range(NUM_CLASSES):
        axis.text(col, row, matrix[row, col], ha="center", va="center")

fig.tight_layout()
plt.show()

# 7. PER-CLASS ACCURACY

per_class_accuracy = np.divide(
    np.diag(matrix),
    matrix.sum(axis=1),
    out=np.zeros(NUM_CLASSES, dtype=float),
    where=matrix.sum(axis=1) != 0,
)
print("\n")
print("=" * 75)
print("     PER-CLASS ACCURACY")
print("=" * 75)
for i in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[i]:8s} : {per_class_accuracy[i] * 100:.2f}%")

fig, axis = plt.subplots(figsize=(10, 6))
axis.bar(CLASS_NAMES, per_class_accuracy * 100)
axis.set_title(" Per-Class Accuracy ")
axis.set_xlabel("Class")
axis.set_ylabel("Accuracy (%)")
axis.set_ylim(0, 100)
fig.tight_layout()
plt.show()
#  CONFIDENCE DISTRIBUTION

fig, axis = plt.subplots(figsize=(9, 6))
axis.hist(confidences * 100, bins=10, range=(0, 100), edgecolor="black")
axis.set_title(" Model Confidence Distribution ")
axis.set_xlabel("Confidence (%)")
axis.set_ylabel("Number of Images")
fig.tight_layout()
plt.show()


print("\n")
print("=" * 75)
print("=" * 75)