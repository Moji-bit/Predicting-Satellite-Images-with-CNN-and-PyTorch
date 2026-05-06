# Mojtaba Akhundzadah - Student ID: K12443705

from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from shiny.express import input, render, ui


# ------------------------------------------------------------
# Paths and device
# ------------------------------------------------------------

APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = APP_DIR / "best_model.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_DIR / "assets" / "weights" / "best_model.pth"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# Class names
# This order must match the sorted class-folder order in the notebook.
# ------------------------------------------------------------

CLASS_NAMES = [
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


# ------------------------------------------------------------
# Model architecture
# This must match the notebook model exactly.
# ------------------------------------------------------------

class SatelliteCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# Prediction transform
# Same as validation/test transform in the notebook.
# ------------------------------------------------------------

prediction_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = SatelliteCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------

ui.page_opts(title="Satellite Image Classifier", fillable=True)

ui.h2("Satellite Image Classifier")
ui.p("Upload a 64x64 RGB satellite image. The CNN model predicts the terrain class.")

ui.input_file(
    "image_file",
    "Select satellite image",
    accept=[".jpg", ".jpeg", ".png"],
    multiple=False
)

ui.hr()


@render.image
def uploaded_image():
    file_info = input.image_file()

    if not file_info:
        return None

    image_path = file_info[0]["datapath"]

    return {
        "src": image_path,
        "width": "300px",
        "height": "300px",
        "alt": "Uploaded satellite image"
    }


def predict_uploaded_image():
    file_info = input.image_file()

    if not file_info:
        return None

    image_path = file_info[0]["datapath"]

    image = Image.open(image_path).convert("RGB")
    image_tensor = prediction_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    probabilities_cpu = probabilities.cpu().numpy()
    predicted_index = int(probabilities.argmax().item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    probability_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probabilities_cpu
    })

    probability_df["Probability"] = probability_df["Probability"].round(4)
    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    ).reset_index(drop=True)

    return predicted_class, confidence, probability_df


@render.ui
def prediction_card():
    result = predict_uploaded_image()

    if result is None:
        return ui.div("Please upload an image first.")

    predicted_class, confidence, _ = result

    return ui.div(
        ui.h4("Prediction"),
        ui.p(f"Predicted class: {predicted_class}"),
        ui.p(f"Confidence: {confidence * 100:.2f}%")
    )


@render.data_frame
def probability_table():
    result = predict_uploaded_image()

    if result is None:
        return pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": [0.0] * len(CLASS_NAMES)
        })

    _, _, probability_df = result
    return probability_df
