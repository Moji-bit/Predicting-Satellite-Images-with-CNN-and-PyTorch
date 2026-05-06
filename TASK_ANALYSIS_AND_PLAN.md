# Task Analysis and Execution Plan (K12443705)

## What the assignment requires

### Dataset and labels
- 10,000 RGB images, each 64x64.
- 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Pasture, PermanentCrop, Residential, River, SeaLake, Industrial.
- Training data is class-folder structured.

### 1) Data handling & preprocessing (4 pts)
Implement:
- `preprocess(data_folder: str) -> tuple[pd.DataFrame, dict]`

Output expectations:
- DataFrame with at least: `folder`, `file_name`, numeric `label`.
- Label dictionary mapping class names to numeric IDs (or inverse mapping, as long as used consistently).
- Stratified train/validation split (e.g., 80/20) with fixed random seed.

### 2) EDA (9 pts)
Implement and save plots to `assets/plots/`:
1. `show_samples(df, num_samples=5)` -> `random_samples.png`
2. `average_pixel_plot(df)` -> `average_pixel_distribution.png`
3. `average_brightness_per_class(df)` -> `average_brightness.png`

### 3) CNN implementation & training (20 pts)
Required components:
- Custom `Dataset` class using DataFrame rows.
- Train/val transforms (augmentation for train only).
- DataLoaders with sensible batch size and shuffle behavior.
- **Custom CNN only** (no transfer learning / pretrained backbone).
- Training + validation loops with:
  - `CrossEntropyLoss`
  - Optimizer (Adam/SGD)
  - metric tracking (loss + accuracy)
  - checkpoint best model to `assets/weights/best_model.pth`
  - optional scheduler / early stopping

### 4) Model evaluation & analysis (7 pts)
Create:
- Training vs validation loss curve + validation accuracy curve.
- Confusion matrix on validation set.
- 5 misclassified sample visualizations (true vs predicted labels).

### 5) Test prediction + submission format
- Test folder has images only (no class subfolders).
- Implement `preprocess_test(data_folder: str) -> pd.DataFrame`.
- Predict class labels for all test images.
- Create `submission.csv` with:
  - no header
  - `file_name,predicted_class_text`
  - class names as text, not IDs.

### 6) Shiny Express app (10 pts)
- Use **Shiny Express** (not Shiny Core).
- User uploads image, app shows:
  - uploaded image preview
  - predicted class
  - confidence score
  - full class probability table
- Load model with safe return-value assignment:
  - `_ = model.load_state_dict(state_dict)`
  - `_ = model.eval()`
- Apply same normalization/inference transforms as model training pipeline.

## Key grading/risk points
- Notebook must run top-to-bottom without crashing.
- No external datasets and no dataset modification.
- Challenge server scoring has limited attempts; submissions must be validated locally first.

## Step-by-step implementation order (how we will proceed)
1. Preprocessing + label maps + stratified split.
2. EDA functions + saved plots.
3. Dataset/transforms/dataloaders.
4. CNN model definition.
5. Training utilities + checkpointing.
6. Evaluation plots + misclassification analysis.
7. Test preprocessing + submission generation.
8. Shiny app inference UI.
9. End-to-end rerun and packaging.

## Next action
Start with Step 1 only: implement `preprocess()` and train/validation split with reproducibility.
