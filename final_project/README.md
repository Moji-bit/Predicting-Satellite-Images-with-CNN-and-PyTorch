# Satellite Image Classification with CNN and PyTorch

This project trains a custom CNN to classify 64x64 RGB satellite images into 10 terrain classes.

## Main files and folders
- `k12443705.ipynb`
- `app/app.py`
- `assets/plots`
- `assets/weights/best_model.pt`
- `submission.csv`

## How to run the notebook
Open `k12443705.ipynb` and run all cells from top to bottom.

## How to run the Shiny app
```bash
shiny run --reload --launch-browser app/app.py
```

## Notes
- No external data is used.
- The model is trained from scratch.
- The best model is saved based on validation accuracy.
- `submission.csv` is generated separately for the challenge server.
- The ZIP file is created manually by the student.
