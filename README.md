# Introduction
A Flexible Framework for Testing and Developing Deep Learning Models for Seizure Prediction and Detection Task using CHB-MIT dataset.

## Quick Start

```bash
git clone https://github.com/mehrdad-anvari/seizure-prediction.git
cd seizure-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training & Analysis System

This repository includes a comprehensive training and analysis pipeline for seizure prediction models with:
- **Nested cross-validation** for robust evaluation
- **Separation of training and analysis** for flexibility and reproducibility
- **Multiple calibration methods** and post-processing options
- **Comprehensive metrics and visualizations**

### 🔧 **Core Features**

#### Training (`train.py`)
* **Argument Parsing**: Command-line arguments + configuration file support
* **Experiment Tracking**: Organized run directories with timestamps and IDs
* **Model Checkpointing**: Save only the best models based on validation AUC
* **Undersampling**: Random undersampling of interictal data each epoch
* **Raw Predictions**: Saves unprocessed predictions for flexible post-analysis
* **Comprehensive Logging**: Structured logging with training metrics

#### Analysis (`analyze_results2.py`)
* **Multiple Calibration Methods**: Percentile, Beta, Isotonic, Temperature scaling
* **Moving Average Smoothing**: Configurable window sizes (1, 3, 5, 7, 10, ...)
* **Threshold Exploration**: Test multiple classification thresholds
* **Comprehensive Metrics**: AUC, sensitivity, FPR/hour, precision, recall, F1, and more
* **Rich Visualizations**: ROC curves, PR curves, confusion matrices, Pareto frontiers
* **Modular Design**: Easy to extend with new metrics and visualizations

### 📁 **Directory Structure**

```
runs/
├── run1_20231215_120000/
│   ├── config.json                      # Run configuration
│   ├── raw_predictions.pkl              # Raw predictions from training
│   ├── checkpoints/                     # Best model checkpoints
│   │   ├── best_model_outer1_inner1.pth
│   │   └── ...
│   ├── logs/
│   │   └── training.log                 # Training logs
│   ├── variant_summary.csv              # ALL variants with metrics
│   ├── best_variants_auc.csv            # Top 10 by AUC
│   ├── best_variants_sensitivity.csv    # Top 10 by sensitivity
│   ├── best_variants_f1.csv             # Top 10 by F1
│   ├── calibration_comparison.csv       # Compare calibration methods
│   ├── ma_window_comparison.csv         # Compare MA windows
│   ├── threshold_comparison.csv         # Compare thresholds
│   ├── pareto_optimal_variants.csv      # Pareto frontier variants
│   └── visualizations/
│       ├── roc_*.png
│       ├── pr_*.png
│       ├── prob_dist_*.png
│       ├── cm_*.png
│       ├── comparison_*.png
│       ├── threshold_analysis_*.png
│       ├── ma_comparison_*.png
│       ├── calibration_comparison_*.png
│       └── pareto_frontier.png
└── run2_20231215_130000/
    └── ...
```

---

## Usage

### 1. Training Models

Train models and save raw predictions (no post-processing):

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
    --lr 5e-4
```

### 2. Analyzing Results

Analyze trained models with comprehensive post-processing:

```bash
# Analyze most recent run with default parameters
python analyze_results2.py

# Analyze specific run
python analyze_results2.py --run_dir runs/run1_20231215_120000

# Customize analysis parameters
python analyze_results2.py \
    --calibration_methods none percentile beta \
    --ma_windows 1 3 5 7 10 \
    --thresholds 0.3 0.4 0.5 0.6 0.7 \
    --percentiles 5 10 15 20

# Minimal analysis (faster)
python analyze_results2.py \
    --calibration_methods none percentile \
    --ma_windows 1 3 5 \
    --thresholds 0.5
```

### 3. Using Configuration Files

```bash
# Create configuration template
python config_template.py

# Run training with config file
python train.py --config config_basic.json

# Override specific parameters
python train.py --config config_basic.json --epochs 50 --lr 5e-4
```

### 4. Complete Workflow

```bash
# Step 1: Train model
python train.py \
    --subject_id "01" \
    --model "CE-stSENet" \
    --epochs 30

# Step 2: Comprehensive analysis
python analyze_results2.py

# Step 3: Review results
# - Check variant_summary.csv for all combinations
# - Review best_variants_*.csv for top performers
# - Examine visualizations/ folder for plots
# - Check pareto_optimal_variants.csv for optimal tradeoffs
```

---

## Command Line Arguments

### Training (`train.py`)

#### Dataset Parameters
* `--dataset_dir`: Path to dataset directory (default: "data/BIDS_CHB-MIT")
* `--subject_id`: Subject ID (default: "01")
* `--suffix`: Dataset suffix (default: "None_F_F")
* `--use_uint16`: Use uint16 format (flag)
* `--apply_normalization`: Apply InstanceNormTransform (flag)

#### Model Parameters
* `--model`: Model name (default: "CE-stSENet")
  - Options: CE-stSENet, EEGNet, FBMSNet, EEGBandClassifier, etc.

#### Training Parameters
* `--batch_size`: Batch size (default: 64)
* `--lr`: Learning rate (default: 1e-3)
* `--epochs`: Number of epochs (default: 20)

#### Cross-Validation Parameters
* `--outer_cv_mode`: Outer CV mode (default: "leave_one_preictal")
  - Options: "leave_one_preictal", "stratified"
* `--outer_cv_method`: Method for leave-one-preictal (default: "balanced_shuffled")
  - Options: "balanced", "balanced_shuffled", "nearest"
* `--inner_cv_mode`: Inner CV mode (default: "leave_one_preictal")
* `--inner_cv_method`: Method for inner CV (default: "balanced_shuffled")
* `--n_fold`: Number of folds for stratified CV (default: 5)
* `--inner_cv_shuffle`: Shuffle flag for inner CV (flag)
* `--outer_cv_shuffle`: Shuffle flag for outer CV (flag)

#### Other Parameters
* `--random_state`: Random seed (default: 42)
* `--config`: Path to JSON config file

### Analysis (`analyze_results2.py`)

#### Basic Options
* `--run_dir`: Path to specific run directory (default: most recent)
* `--runs_dir`: Path to runs directory (default: "runs")

#### Analysis Parameters
* `--calibration_methods`: Calibration methods to analyze
  - Options: none, percentile, beta, isotonic, temperature
  - Default: All methods
* `--ma_windows`: Moving average windows to test (default: 1, 3, 5, 7, 10)
* `--thresholds`: Classification thresholds to test (default: 0.3, 0.4, 0.5, 0.6, 0.7)
* `--percentiles`: Percentiles for percentile calibration (default: 5, 10, 15, 20)

---

## Cross-Validation Strategies

### 1. Leave-One-Preictal-Out (Default)

Best for seizure prediction as it tests generalization to unseen seizures:

```bash
python train.py \
    --outer_cv_mode "leave_one_preictal" \
    --outer_cv_method "balanced_shuffled" \
    --inner_cv_mode "leave_one_preictal" \
    --inner_cv_method "balanced_shuffled"
```

**Methods:**
- `balanced`: Balance interictal samples across folds
- `balanced_shuffled`: Balanced + randomized selection
- `nearest`: Use temporally nearest interictal samples

### 2. Stratified K-Fold

Standard K-fold with stratification:

```bash
python train.py \
    --outer_cv_mode "stratified" \
    --inner_cv_mode "stratified" \
    --n_fold 5
```

### 3. Mixed Approaches

Combine different strategies:

```bash
python train.py \
    --outer_cv_mode "leave_one_preictal" \
    --inner_cv_mode "stratified" \
    --n_fold 3
```

---

## Output Metrics

### Basic Metrics
* **Accuracy**: Overall classification accuracy
* **Precision**: Positive predictive value
* **Recall**: True positive rate
* **F1 Score**: Harmonic mean of precision and recall
* **AUC**: Area Under ROC Curve
* **AP**: Average Precision (area under PR curve)

### Seizure-Specific Metrics
* **Sensitivity**: At least one preictal sample detected per seizure
* **FPR/hour**: False positives per hour
* **Time to Detection**: Time from first preictal to first detection (seconds)
* **Specificity**: True negative rate
* **Confusion Matrix**: TP, TN, FP, FN counts

### Variants Tested

The analysis automatically generates results for combinations of:
- **5 calibration methods** (none, percentile, beta, isotonic, temperature)
- **5 MA windows** (1, 3, 5, 7, 10)
- **5 thresholds** (0.3, 0.4, 0.5, 0.6, 0.7)
- **4 percentiles** for percentile calibration (5, 10, 15, 20)

This creates **hundreds of variant combinations** to find optimal performance.

---

## Calibration Methods

### 1. No Calibration (`none`)
Raw ensemble predictions without calibration.

### 2. Percentile-Based (`percentile`)
Shifts probability distribution so that top N% of validation preictal samples are above threshold 0.5.

```bash
# Test different percentiles
python analyze_results2.py --percentiles 5 10 15 20 25 30
```

**Use case**: Reduce false positives by only detecting highest-confidence predictions.

### 3. Beta Distribution (`beta`)
Fits Beta distributions to preictal and interictal validation probabilities, then uses likelihood ratio.

**Use case**: When class distributions have clear parametric forms.

### 4. Isotonic Regression (`isotonic`)
Non-parametric calibration using sklearn's isotonic regression.

**Use case**: General-purpose calibration, no distribution assumptions.

### 5. Temperature Scaling (`temperature`)
Scales logits by learned temperature parameter.

**Use case**: When model is already well-calibrated but needs fine-tuning.

---

## Visualization Outputs

### Per-Variant Plots
- **ROC Curves**: True positive rate vs false positive rate
- **Precision-Recall Curves**: Precision vs recall tradeoff
- **Probability Distributions**: Histograms for preictal vs interictal
- **Confusion Matrices**: Classification results per fold

### Comparison Plots
- **Metric Comparisons**: Bar charts of top 20 variants by metric
- **Threshold Sensitivity**: How metrics change with threshold
- **MA Window Comparison**: Performance across window sizes
- **Calibration Method Comparison**: Boxplots comparing methods
- **Pareto Frontier**: Optimal sensitivity vs FPR tradeoff

---

## Example Workflows

### 1. Single Subject Analysis

```bash
# Train
python train.py --subject_id "01" --model "CE-stSENet" --epochs 30

# Analyze
python analyze_results2.py

# Review
# - Check variant_summary.csv
# - Look at pareto_optimal_variants.csv
# - Review visualizations/
```

### 2. Model Comparison

```bash
# Train multiple models
python train.py --model "CE-stSENet" --subject_id "01"
python train.py --model "EEGNet" --subject_id "01"
python train.py --model "FBMSNet" --subject_id "01"

# Analyze each
python analyze_results2.py --run_dir runs/run1_*
python analyze_results2.py --run_dir runs/run2_*
python analyze_results2.py --run_dir runs/run3_*

# Compare best_variants_auc.csv across runs
```

### 3. Hyperparameter Sweep

```bash
# Sweep learning rates
for lr in 1e-3 5e-4 1e-4; do
    python train.py --lr $lr --subject_id "01" --epochs 30
done

# Analyze all
for run_dir in runs/run*; do
    python analyze_results2.py --run_dir $run_dir
done
```

### 4. Optimization for Specific Metric

```bash
# Train once
python train.py --subject_id "01" --epochs 30

# Analyze
python analyze_results2.py

# Review best_variants_sensitivity.csv if you want high sensitivity
# Review best_variants_f1.csv if you want balanced performance
# Review pareto_optimal_variants.csv for optimal tradeoffs
```

### 5. Quick Analysis

```bash
# Fast analysis with fewer variants
python analyze_results2.py \
    --calibration_methods none percentile \
    --ma_windows 1 5 \
    --thresholds 0.5
```

---

## Key Improvements Over Previous System

### 1. Separation of Concerns
* **Training** focuses only on model training and raw predictions
* **Analysis** handles ALL post-processing, metrics, and visualization
* Re-analyze without retraining
* Test new calibration methods on existing predictions

### 2. Comprehensive Variant Testing
* Automatically tests hundreds of combinations
* Identifies optimal configurations per metric
* Pareto frontier analysis for tradeoff decisions

### 3. Modular & Extensible Design
* Easy to add new metrics (3 lines of code)
* Easy to add new visualizations (1 function)
* Easy to add new calibration methods (2 functions)
* Easy to add new post-processing

### 4. Better Results Organization
* All variants in one CSV for easy filtering
* Separate tables for best performers per metric
* Comparison tables for each parameter
* Rich visualizations for understanding

### 5. Reproducibility
* Raw predictions saved for exact reproduction
* Configuration saved with every run
* All analysis parameters logged
* Deterministic random seeds

---

## Extending the System

### Adding a New Metric

```python
# In analyze_results2.py, add to MetricsCalculator
@staticmethod
def compute_all_metrics(y_true, y_pred, y_probs):
    basic = MetricsCalculator.compute_basic_metrics(...)
    seizure = MetricsCalculator.compute_seizure_specific_metrics(...)
    
    # Add your metric
    custom = {
        'my_metric': compute_my_metric(y_true, y_pred)
    }
    
    return {**basic, **seizure, **custom}
```

The metric automatically appears in all summaries and tables!

### Adding a New Visualization

```python
# In Visualizer class
def plot_my_visualization(self, processed_results, variant_name):
    plt.figure(figsize=(10, 8))
    # Your plotting code
    plt.savefig(self.viz_dir / 'my_viz.png')
    plt.close()

# Call it in analyze_run()
visualizer.plot_my_visualization(processed_results, best_variant)
```

### Adding a New Calibration Method

```python
# In ProbabilityCalibrator class
def _fit_my_method(self, val_probs, val_labels):
    self.my_param = learn_param(val_probs, val_labels)

def _transform_my_method(self, probs):
    return calibrate(probs, self.my_param)

# Add to dispatch in fit() and transform() methods
```

See the included **Analysis System Guide** document for detailed extension instructions.

---

## Troubleshooting

### Training Issues
* **Import Errors**: Ensure `transforms.signal.normalize` and `models.provider` exist
* **CUDA Issues**: Falls back to CPU automatically
* **Memory Issues**: Lower batch size or reduce epochs
* **File Permissions**: Check write access for `runs/` directory

### Analysis Issues
* **No raw_predictions.pkl**: Train with the new `train.py` version
* **Missing calibration methods**: Install scipy and sklearn
* **Out of memory**: Reduce variant combinations with CLI arguments
* **Slow analysis**: Use fewer calibration methods, MA windows, or thresholds

### Result Interpretation
* **High FPR**: Try higher thresholds or percentile calibration
* **Low sensitivity**: Try lower thresholds or remove calibration
* **Check Pareto frontier**: For optimal sensitivity/FPR tradeoff
* **Check variant_summary.csv**: Sort by your preferred metric

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

### Custom Post-Processing

Modify `ResultsProcessor._process_outer_fold()` in `analyze_results2.py` to add custom post-processing steps.

---

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Cross-Patient Seizure Prediction via Dynamic Multi-Graph Convolution and Channel-Weighted Transformer},
  author={Your Names},
  journal={Your Journal},
  year={2024}
}
```

---

## License

[Your License Here]

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For questions or issues, please open a GitHub issue.