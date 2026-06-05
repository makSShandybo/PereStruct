# Training the Classifiers

This guide covers training the two-level article linking models: Plain-Text and Title-Plain.

## Prerequisites

- Dataset prepared in `perestruct_benchmark/` with annotated page images
- All dependencies installed (`pip install -r requirements.txt`)
- YOLO weights available (e.g., `soviet_yolo.pt`)

## Training Workflow

### 1. Copy Data

Copy the benchmark dataset to the project's data folder:

```bash
cp -r perestruct_benchmark/* data/
```

Expected structure:
```
data/
├── dataset_reindexed.json    # Main annotations
└── images/                     # Page images (.jpg)
```

### 2. Prepare Datasets

Generate pair datasets with TF-IDF and YOLO embeddings. This creates cached splits (`splits.pkl`) and pair files for both models:

```bash
python prepare_dataset.py --workers 8 --device 0
```

**Outputs:**
- `processed/splits.pkl` (train/val/test split indices)
- `processed/pairs_soviet.pkl` (Plain Text model dataset)
- `processed/pairs_soviet_title.pkl` (Title model dataset)

*Note: This step is memory and GPU intensive. Adjust `--workers` based on your GPU memory.*

### 3. Train Plain Text Model

Train the baseline model for Plain-Text ↔ Plain-Text block linking:

```bash
python train_text.py
```

**Inputs:** `processed/pairs_soviet.pkl`  
**Outputs:**
- `models/best_soviet_model.pkl`
- `models/best_soviet_vectorizer.pkl`
- `models/best_soviet_model_metadata.json`

Training uses TF-IDF (char n-grams 2-4) + Geometric features + YOLO embeddings (576-dim).

### 4. Train Title-Plain Model

Train the specialized model handling Title ↔ Plain Text and Title ↔ Title pairs:

```bash
python train_title.py
```

**Inputs:** `processed/pairs_soviet_title.pkl`  
**Outputs:**
- `models/best_soviet_title_model.pkl`
- `models/best_soviet_title_vectorizer.pkl`
- `models/best_soviet_title_model_metadata.json`

This model adds binary type features (is_title flags) to the feature set.

You should have 4 files total (2 models + 2 vectorizers) before running prediction.