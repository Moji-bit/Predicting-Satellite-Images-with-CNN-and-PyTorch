# Predicting Satellite Images with CNN and PyTorch — Simple Overview

## 1) What this project does
This project builds a complete image-classification pipeline for EuroSAT satellite patches.
Each input image is **64×64 RGB**, and the model predicts one of **10 land-use classes**.

Classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake
- Industrial

---

## 2) End-to-end goal
Build the full workflow:

1. Load and index data
2. Preprocess labels and split train/validation
3. Run EDA and save plots
4. Train a **custom** CNN in PyTorch (no pretrained model)
5. Evaluate with curves + confusion matrix + misclassifications
6. Predict on unlabeled test images
7. Generate `submission.csv` (no header, class names as text)
8. Build a Shiny Express app for single-image upload + prediction

---

## 3) Minimum required functions

### Data preprocessing
- `preprocess(data_folder: str) -> tuple[pd.DataFrame, dict]`
  - Return a dataframe with file paths and integer labels.
  - Return an index-to-class mapping dictionary.
- `preprocess_test(data_folder: str) -> pd.DataFrame`
  - Return test image file list (no labels).

Use an **80/20 stratified split** with a fixed random seed.

### EDA
Save all figures under `assets/plots/`:
- `show_samples(df, num_samples=5)` → `random_samples.png`
- `average_pixel_plot(df)` → `average_pixel_distribution.png`
- `average_brightness_per_class(df)` → `average_brightness.png`

### CNN training
Implement:
- `SatelliteDataset(Dataset)`
- Train/validation transforms
- DataLoaders (`shuffle=True` for train; `False` for val/test)
- Custom CNN (not transfer learning)
- Training loop with loss, accuracy, validation, and best-model checkpointing

Save best model to:
- `assets/weights/best_model.pth`

### Evaluation outputs
Save:
- `training_curves.png`
- `confusion_matrix.png`
- `misclassified_samples.png`

### Test prediction output
Create `submission.csv` in format:

```text
273.jpg,HerbaceousVegetation
1418.jpg,SeaLake
33.jpg,Industrial
```

No header row.

---

## 4) Shiny Express app requirements
The app should:
- Upload one image
- Display image preview
- Display predicted class
- Display confidence score
- Display probability table for all classes

Load model safely in app runtime:

```python
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()
```

Use the **same normalization/transforms** as validation/inference.

---

## 5) Suggested implementation order
1. Folder structure
2. Preprocessing and stratified split
3. EDA plots
4. Dataset + transforms + loaders
5. CNN model
6. Training and checkpointing
7. Evaluation plots
8. Test predictions and `submission.csv`
9. Shiny app
10. Full notebook rerun from top to bottom

---

## 6) Common mistakes to avoid
- Using pretrained models
- Non-stratified split
- Missing random seed
- Header in `submission.csv`
- Numeric labels in `submission.csv`
- Different app transforms vs training
- Missing saved plots/weights
- Notebook cells failing on clean rerun

---

## 7) Deliverables checklist
Zip file should contain:
- `k[studentID].ipynb`
- `app/` directory (with `app.py` and required files)

Zip name format:
- `k[studentID]_final_project.zip`

`submission.csv` is uploaded separately to the challenge server.
