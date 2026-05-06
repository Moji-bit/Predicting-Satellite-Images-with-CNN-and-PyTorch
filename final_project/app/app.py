# Student: K12443705
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from shiny.express import input, render, ui

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path(__file__).resolve().parent / 'best_model.pth'

IDX_TO_CLASS = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake',
}

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = SimpleCNN(num_classes=10).to(DEVICE)
if MODEL_PATH.exists():
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    _ = model.load_state_dict(state_dict)
_ = model.eval()

ui.h3('Satellite Image Classifier')
ui.input_file('file1', 'Upload image', accept=['.jpg', '.jpeg', '.png'])

@render.image
def uploaded_image():
    f = input.file1()
    if not f:
        return None
    return {'src': f[0]['datapath'], 'alt': 'Uploaded image', 'width': '256px'}

@render.text
def prediction_text():
    f = input.file1()
    if not f:
        return 'Upload an image to get prediction.'

    img = Image.open(f[0]['datapath']).convert('RGB')
    x = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx].item()) * 100
    return f'Predicted class: {pred_class} | Confidence: {confidence:.2f}%'

@render.table
def probability_table():
    f = input.file1()
    if not f:
        return pd.DataFrame({'Class': [], 'Probability': []})

    img = Image.open(f[0]['datapath']).convert('RGB')
    x = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    df = pd.DataFrame({
        'Class': [IDX_TO_CLASS[i] for i in range(10)],
        'Probability': probs,
    }).sort_values('Probability', ascending=False).reset_index(drop=True)
    return df
