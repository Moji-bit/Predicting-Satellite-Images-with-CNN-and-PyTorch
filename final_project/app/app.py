# Name: YOUR_NAME_HERE | Student ID: YOUR_STUDENT_ID_HERE
"""Shiny Express app for satellite image classification with a trained CNN model."""

from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from shiny.express import input, render, ui

# Select GPU when available, otherwise use CPU.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# app.py is inside app/, so we go one level up to reach assets/weights.
APP_DIR: Path = Path(__file__).resolve().parent
PROJECT_DIR: Path = APP_DIR.parent
MODEL_PATH: Path = PROJECT_DIR / "assets" / "weights" / "best_model.pt"

# Fixed class order must match training label order exactly.
CLASS_NAMES: list[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class SatelliteCNN(nn.Module):
    """CNN architecture used during training, reused here for prediction."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Feature extractor with 4 convolution blocks.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive pooling + classifier head.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from image tensor to logits."""
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_model() -> nn.Module:
    """Load trained model weights and return model in eval mode."""

    # Create the model architecture first.
    model = SatelliteCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)

    # Load trained weights from disk to selected device.
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Assign return values to _ to avoid Shiny issues with returned objects.
    _ = model.load_state_dict(state_dict)
    _ = model.eval()

    return model


def get_prediction_transform() -> transforms.Compose:
    """Return deterministic transform used for validation/test prediction."""

    # We avoid random augmentation in the app because inference must be stable.
    # Same normalization values as training/validation pipeline.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def predict_image(image_path: str | Path) -> tuple[str, float, pd.DataFrame]:
    """Predict class and probabilities for one uploaded image."""

    img_path = Path(image_path)

    # Convert to RGB so every input has exactly 3 channels.
    image = Image.open(img_path).convert("RGB")

    transform = get_prediction_transform()
    x = transform(image)

    # Add batch dimension because model expects [batch, channels, height, width].
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)

        # Softmax converts raw logits into probabilities that sum to 1.
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    predicted_class = CLASS_NAMES[pred_idx]

    # Confidence is the model probability for the predicted class.
    confidence = float(probs[pred_idx].item())

    probability_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs.cpu().numpy(),
    }).sort_values("Probability", ascending=False).reset_index(drop=True)

    return predicted_class, confidence, probability_df


# Load model once at app startup for faster predictions.
MODEL = load_model()

# -----------------------------
# Shiny Express UI
# -----------------------------

ui.page_opts(title="Satellite Image Classification", fillable=True)
ui.h2("Satellite Image Classification")
ui.p("The app predicts the terrain class of an uploaded satellite image using a trained CNN model.")

with ui.card():
    ui.card_header("Upload Image")
    ui.input_file("sat_image", "Choose a satellite image", accept=[".jpg", ".jpeg", ".png"], multiple=False)


@render.image
def uploaded_preview():
    """Show uploaded image preview."""

    file_info = input.sat_image()
    if not file_info:
        return None

    return {
        "src": file_info[0]["datapath"],
        "alt": "Uploaded satellite image",
        "width": "320px",
    }


@render.ui
def prediction_card():
    """Show prediction result card or helper message."""

    file_info = input.sat_image()
    if not file_info:
        return ui.p("Please upload a satellite image to get a prediction.")

    predicted_class, confidence, _ = predict_image(file_info[0]["datapath"])

    return ui.div(
        ui.h4("Prediction"),
        ui.p(f"Predicted class: {predicted_class}"),
        ui.p(f"Confidence: {confidence * 100:.2f}%"),
    )


@render.table
def probability_table():
    """Show class probability table for uploaded image."""

    file_info = input.sat_image()
    if not file_info:
        return pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": [0.0] * len(CLASS_NAMES),
        })

    _, _, probs_df = predict_image(file_info[0]["datapath"])
    return probs_df
