# CP-DMGC-CWT
Cross-Patient Seizure Prediction via Dynamic Multi-Graph Convolution and Channel-Weighted Transformer

## Quick Start

```bash
git clone https://github.com/Mj-Heidari/CP-DMGC-CWT.git
cd CP-DMGC-CWT
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training System

This repository includes a full training system for seizure prediction models with nested cross-validation, experiment tracking, checkpointing, undersampling, and analysis.

### 🔧 **Core Features**

* **Argument Parsing**: Command-line arguments + configuration file support
* **Experiment Tracking**: Organized run directories with timestamps and IDs
* **Model Checkpointing**: Save only the best models based on validation AUC
* **Undersampling**: Random undersampling of interictal data each epoch
* **Moving Average Predictions**: Smooth predictions with configurable window size
* **Comprehensive Logging**: Structured logging in multiple formats

### 📁 **Directory Structure**

```
runs/
├── run1_20231215_120000/
│   ├── config.json              # Run configuration
│   ├── results.json             # Final results
│   ├── detailed_results.pkl     # Detailed training logs
│   ├── checkpoints/             # Best model checkpoints
│   │   ├── best_model_outer1_inner1.pth
│   │   └── ...
│   ├── logs/
│   │   └── training.log         # Training logs
│   ├── summary_results.csv      # Results summary
│   ├── configuration.csv        # Configuration table
│   ├── report.html              # HTML report
│   └── *.png                    # Generated plots
└── run2_20231215_130000/
    └── ...
```


---

## Usage

### 1. Basic Training

```bash
# Simple run with default parameters
python train.py \
    --dataset_dir "data/BIDS_CHB-MIT" \
    --subject_id "01" \
    --model "CE-stSENet" \
    --epochs 20

# Custom parameters
python train.py \
    --dataset_dir "data/BIDS_CHB-MIT" \
    --subject_id "02" \
    --model "EEGNet" \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-4 \
    --moving_avg_window 5
```

### 2. Using Configuration Files

```bash
# Create configuration template
python config_template.py

# Run with config file
python train.py --config config_basic.json

# Override specific parameters
python train.py --config config_basic.json --epochs 50 --lr 5e-4
```

### 3. Cross-Validation Options

```bash
# Leave-one-preictal-group-out CV (default)
python train.py \
    --outer_cv_mode "leave_one_preictal" \
    --outer_cv_method "balanced_shuffled" \
    --inner_cv_mode "leave_one_preictal" \
    --inner_cv_method "balanced_shuffled"

# Stratified K-fold CV
python train.py \
    --outer_cv_mode "stratified" \
    --inner_cv_mode "stratified" \
    --n_fold 5

# Mixed approaches
python train.py \
    --outer_cv_mode "leave_one_preictal" \
    --inner_cv_mode "stratified" \
    --n_fold 3
```

### 4. Results Analysis

```bash
# Analyze most recent run
python analyze_results.py

# Analyze specific run
python analyze_results.py --run_dir runs/run1_20231215_120000

# Compare all runs
python analyze_results.py --compare_all

# Compare specific runs
python analyze_results.py --run_ids run1_20231215_120000 run2_20231215_130000
```

---

## Command Line Arguments

### Dataset Parameters

* `--dataset_dir`: Path to dataset directory (default: "data/BIDS_CHB-MIT")
* `--subject_id`: Subject ID (default: "01")
* `--suffix`: Dataset suffix (default: "None_F_F")
* `--use_uint16`: Use uint16 format (flag)

### Model Parameters

* `--model`: Model name (default: "CE-stSENet")

### Training Parameters

* `--batch_size`: Batch size (default: 64)
* `--lr`: Learning rate (default: 1e-3)
* `--epochs`: Number of epochs (default: 20)

### Cross-Validation Parameters

* `--outer_cv_mode`: Outer CV mode ("leave_one_preictal" or "stratified")
* `--outer_cv_method`: Method for leave-one-preictal ("balanced", "balanced_shuffled", "nearest")
* `--inner_cv_mode`: Inner CV mode ("leave_one_preictal" or "stratified")
* `--inner_cv_method`: Method for inner CV
* `--n_fold`: Number of folds for stratified CV (default: 5)

### Other Parameters

* `--random_state`: Random seed (default: 42)
* `--moving_avg_window`: Moving average window size (default: 3)

---

## Key Improvements

1. **Undersampling Strategy**

   * Random undersampling of interictal data each epoch
   * Balanced training without weighted losses

2. **Best Model Selection**

   * Validation AUC tracked per fold
   * Only best checkpoint saved and used for testing

3. **Moving Average Predictions**

   * Configurable window size (default: 3)
   * Raw vs. smoothed metrics comparison

4. **Comprehensive Logging**

   * JSON configs for reproducibility
   * Training curves + visual reports
   * HTML reports with visualizations

---

## Output Metrics

* **AUC**: Area Under ROC Curve
* **Sensitivity**: Detection rate
* **FPR/hour**: False positives per hour
* Results available **raw** and **smoothed**
* Per-fold + aggregated (mean ± std)

---

## Example Workflows

### 1. Single Subject Analysis

```bash
python train.py --subject_id "01" --model "CE-stSENet" --epochs 30
python analyze_results.py
```

### 2. Model Comparison

```bash
python train.py --model "CE-stSENet" --subject_id "01"
python train.py --model "EEGNet" --subject_id "01"
python analyze_results.py --compare_all
```

### 3. Hyperparameter Sweep

```bash
python train.py --lr 1e-3 --subject_id "01"
python train.py --lr 5e-4 --subject_id "01"
python train.py --lr 1e-4 --subject_id "01"
python analyze_results.py --compare_all
```

---

## Troubleshooting

* **Import Errors**: Ensure `transforms.signal.normalize` and `models.provider` exist
* **CUDA Issues**: Falls back to CPU automatically
* **Memory Issues**: Lower batch size or enable gradient checkpointing
* **File Permissions**: Check write access for `runs/` directory

Logs, configs, and checkpoints are stored in `runs/runX_timestamp/`.

---

## Advanced Usage

### Custom Data Transforms

```python
from transforms.signal.your_transform import YourTransform

dataset = CHBMITDataset(
    ...,
    online_transforms=[YourTransform(), InstanceNormTransform()],
    ...
)
```

### Custom Models

```python
from models.provider import get_builder
builder = get_builder(model="YourModelName")
```
