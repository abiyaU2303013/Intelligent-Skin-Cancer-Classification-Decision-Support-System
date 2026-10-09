

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_v2_s


# =========================================================
# 1. PROJECT PATHS
# =========================================================

code_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(code_path)
results_path = os.path.join(project_path, "results")
inference_results_path = os.path.join(results_path, "single_image_predictions")
os.makedirs(inference_results_path, exist_ok=True)

model_file = os.path.join(project_path, "models", "efficientnetv2b3_attention_best.pth")


# =========================================================
# 2. DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)


# =========================================================
# 3. CLASS INFO + DISEASE CARDS
# =========================================================

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7

# General, textbook-level descriptions of the standard HAM10000 dermatology
# classes. This is background information only, not a diagnosis for any
# specific patient or image.
DISEASE_INFO = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen's disease)",
        "description": (
            "A rough, scaly patch of skin caused by years of sun exposure. "
            "It is considered a precancerous to early-stage cancerous lesion "
            "that can, if untreated, progress to squamous cell carcinoma."
        ),
        "symptoms": [
            "Rough, dry, or scaly patch, often pink/red or skin-colored",
            "Feels like sandpaper to the touch",
            "Usually on sun-exposed areas (face, ears, scalp, hands, forearms)",
            "May itch, burn, or become tender",
        ],
        "risk_level": "Moderate (precancerous / early carcinoma in situ)",
        "next_steps": [
            "See a dermatologist for confirmation, ideally with a biopsy",
            "Avoid further sun exposure and use SPF 30+ daily",
            "Do not attempt to treat, freeze, or remove it yourself",
        ],
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "description": (
            "The most common form of skin cancer. It grows slowly, rarely "
            "spreads to other organs, but can invade nearby skin and tissue "
            "if left untreated."
        ),
        "symptoms": [
            "Pearly or waxy bump, sometimes with visible blood vessels",
            "Flat, flesh-colored or brown scar-like lesion",
            "A sore that bleeds, heals, and then returns",
            "Slow-growing over weeks to months",
        ],
        "risk_level": "Moderate-High (malignant, but low metastasis risk)",
        "next_steps": [
            "See a dermatologist promptly for biopsy and treatment planning",
            "Typically treated with minor surgery (excision or Mohs surgery)",
            "Do not delay — early treatment has excellent outcomes",
        ],
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesion (e.g. seborrheic keratosis, solar lentigo)",
        "description": (
            "A common, non-cancerous skin growth. It can look dark and "
            "irregular, which sometimes causes concern, but it is not "
            "dangerous."
        ),
        "symptoms": [
            "Waxy, 'stuck-on' appearance",
            "Brown, black, or tan coloring",
            "Slightly raised, rough or warty surface",
            "Usually painless and slow-growing",
        ],
        "risk_level": "Low (benign)",
        "next_steps": [
            "Usually no treatment needed",
            "Have a dermatologist confirm it if it changes shape, color, or size",
            "Removal is optional and only for cosmetic/comfort reasons",
        ],
    },
    "df": {
        "full_name": "Dermatofibroma",
        "description": (
            "A common, benign, firm skin nodule, often forming after a minor "
            "injury such as an insect bite. Made of fibrous tissue."
        ),
        "symptoms": [
            "Small, firm, raised bump, usually on the legs or arms",
            "Brown, pink, or reddish color",
            "Dimples inward when pinched from the sides",
            "Usually painless, occasionally itchy or tender",
        ],
        "risk_level": "Low (benign)",
        "next_steps": [
            "Generally no treatment needed",
            "See a doctor if it grows quickly, bleeds, or changes suddenly",
            "Can be surgically removed if bothersome",
        ],
    },
    "mel": {
        "full_name": "Melanoma",
        "description": (
            "The most serious and dangerous form of skin cancer. It develops "
            "in pigment-producing cells and can spread (metastasize) to "
            "other organs if not caught early."
        ),
        "symptoms": [
            "Asymmetric mole or spot with irregular borders",
            "Multiple colors within one lesion (brown, black, red, white, blue)",
            "Diameter larger than 6mm (about a pencil eraser)",
            "A mole that is Evolving — changing in size, shape, or color",
            "(Remember the ABCDE rule: Asymmetry, Border, Color, Diameter, Evolving)",
        ],
        "risk_level": "HIGH (malignant, can be life-threatening if untreated)",
        "next_steps": [
            "See a dermatologist or oncologist as soon as possible",
            "Request a dermoscopic exam and biopsy without delay",
            "Do not wait to 'see if it changes' — early detection saves lives",
        ],
    },
    "nv": {
        "full_name": "Melanocytic Nevus (common mole)",
        "description": (
            "An ordinary mole, made of clusters of pigment cells. The vast "
            "majority of moles are completely harmless."
        ),
        "symptoms": [
            "Round or oval shape with even, well-defined borders",
            "Uniform color (tan, brown, or black)",
            "Usually smaller than 6mm and stable over time",
            "Flat or slightly raised",
        ],
        "risk_level": "Low (benign)",
        "next_steps": [
            "No action needed for typical, stable moles",
            "Monitor for the ABCDE warning signs of change",
            "Get an annual skin check if you have many moles or a family history of melanoma",
        ],
    },
    "vasc": {
        "full_name": "Vascular Lesion (e.g. hemangioma, angioma, pyogenic granuloma)",
        "description": (
            "A group of skin lesions formed from blood vessels. The large "
            "majority are benign and related to normal or abnormal blood "
            "vessel growth."
        ),
        "symptoms": [
            "Red, purple, or blue coloring",
            "Flat or raised, sometimes blanches (lightens) when pressed",
            "Can appear suddenly, especially pyogenic granulomas",
            "Usually painless unless bumped or irritated",
        ],
        "risk_level": "Low (usually benign)",
        "next_steps": [
            "Usually no treatment necessary",
            "See a doctor if it bleeds easily, grows rapidly, or changes color",
            "Cosmetic removal (laser) is an option if desired",
        ],
    },
}


# =========================================================
# 4. IMAGE PICKER
# =========================================================

def pick_image_path(cli_arg_path=None):
    """
    Returns a path to an image chosen by the user.
    Priority: --image CLI arg > file dialog (if GUI available) > manual input.
    """
    if cli_arg_path:
        return cli_arg_path

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select a skin lesion image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()

        if path:
            return path
        print("\nNo file selected. Falling back to manual path entry.")
    except Exception:
        print("\nNo GUI file dialog available. Falling back to manual path entry.")

    return input("Enter the full path to the image: ").strip().strip('"')


# =========================================================
# 5. PREPROCESSING (must match training/validation transforms)
# =========================================================

IMG_SIZE = 224

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    tensor = inference_transform(pil_image).unsqueeze(0)  # add batch dim
    return pil_image, tensor


# =========================================================
# 6. MODEL LOADING
# =========================================================

def load_model():
    if not os.path.exists(model_file):
        print("\nERROR: Model file not found:")
        print(model_file)
        raise SystemExit

    print("\nLoading EfficientNetV2-S...")

    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Trained model loaded successfully.")
    return model


# =========================================================
# 7. GRAD-CAM (EXPLAINABLE AI)
# =========================================================

class GradCAM:
    """
    Standard Grad-CAM implementation. Hooks into the last convolutional
    block of EfficientNetV2-S (model.features[-1]) to show which regions
    of the image most influenced the predicted class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, class_index]
        score.backward()

        gradients = self.gradients[0]          # [C, H, W]
        activations = self.activations[0]      # [C, H, W]

        weights = gradients.mean(dim=(1, 2))   # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def overlay_heatmap(pil_image, cam, alpha=0.45):
    """Resizes the CAM to the original image size and overlays it as a heatmap."""
    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize(pil_image.size, resample=Image.BILINEAR)
    )
    cam_resized = cam_resized / 255.0

    heatmap = plt.get_cmap("jet")(cam_resized)[:, :, :3]  # drop alpha channel
    base_image = np.array(pil_image) / 255.0

    overlay = (1 - alpha) * base_image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return cam_resized, overlay


# =========================================================
# 8. MAIN PIPELINE
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    args = parser.parse_args()

    image_path = pick_image_path(args.image)

    if not image_path or not os.path.exists(image_path):
        print("\nERROR: No valid image path provided.")
        raise SystemExit

    print("\nSelected image:")
    print(image_path)

    model = load_model()
    pil_image, input_tensor = load_image(image_path)
    input_tensor = input_tensor.to(device)

    # ---- Prediction ----
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    info = DISEASE_INFO[predicted_class]

    # ---- Print report ----
    print("\n")
    print("=" * 70)
    print("                    PREDICTION RESULT")
    print("=" * 70)

    print(f"\nPredicted class : {predicted_class.upper()} — {info['full_name']}")
    print(f"Confidence      : {confidence_pct:.2f}%")

    print("\nAll class probabilities:")
    for i, name in enumerate(class_names):
        marker = "  <-- predicted" if i == predicted_idx.item() else ""
        print(f"  {name:6s}: {probabilities[i].item() * 100:6.2f}%{marker}")

    print("\n" + "-" * 70)
    print("DISEASE INFORMATION")
    print("-" * 70)
    print(f"\n{info['description']}")

    print("\nCommon symptoms / features:")
    for s in info["symptoms"]:
        print(f"  - {s}")

    print(f"\nRisk level: {info['risk_level']}")

    print("\nRecommended next steps:")
    for step in info["next_steps"]:
        print(f"  - {step}")


    # ---- Grad-CAM explainability ----
    print("\nGenerating Grad-CAM explanation...")

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(input_tensor, predicted_idx.item())
    cam_resized, overlay = overlay_heatmap(pil_image, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — Predicted: {predicted_class.upper()} ({confidence_pct:.1f}%)")
    axes[2].axis("off")

    plt.suptitle("Explainable AI: Regions Influencing the Prediction", fontweight="bold")
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gradcam_file = os.path.join(inference_results_path, f"{base_name}_gradcam.png")
    plt.savefig(gradcam_file, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nGrad-CAM figure saved:")
    print(gradcam_file)

    # ---- Save a per-image summary file ----
    summary_file = os.path.join(inference_results_path, f"{base_name}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Predicted class: {predicted_class} - {info['full_name']}\n")
        f.write(f"Confidence: {confidence_pct:.2f}%\n")
        f.write(f"Risk level: {info['risk_level']}\n\n")
        f.write("Symptoms:\n")
        for s in info["symptoms"]:
            f.write(f"  - {s}\n")
        f.write("\nNext steps:\n")
        for step in info["next_steps"]:
            f.write(f"  - {step}\n")
        f.write("\nDisclaimer: AI screening estimate only, not a medical diagnosis.\n")

    print("Summary saved:")
    print(summary_file)

    print("\n")
    print("=" * 70)
    print("             INFERENCE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()