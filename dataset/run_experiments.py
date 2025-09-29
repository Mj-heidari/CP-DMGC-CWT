import subprocess
import sys
from itertools import product

# Configuration
DATASET_DIR = "data/BIDS_CHB-MIT"
SUBJECT_ID = "01"
MODEL = "CE-stSENet"
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3

# Define preprocessing options
NORMALIZATION_METHODS = ["none", "zscore", "robust"]
ICA_OPTIONS = [False, True]
FILTER_OPTIONS = [False, True]


def get_suffix(norm_method, apply_ica, apply_filter):
    """Generate suffix based on preprocessing configuration."""
    # Normalization
    if norm_method == "none":
        norm_str = "None"
    elif norm_method == "zscore":
        norm_str = "Z"
    elif norm_method == "robust":
        norm_str = "R"
    else:
        norm_str = norm_method
    
    # ICA
    ica_str = "T" if apply_ica else "F"
    
    # Filter
    filter_str = "T" if apply_filter else "F"
    
    return f"{norm_str}_{ica_str}_{filter_str}"


def run_preprocessing(dataset_dir, subject_id, norm_method, apply_ica, apply_filter):
    """Run preprocessing script."""
    cmd = [
        "python", "preprocess_chbmit.py",
        "--dataset_dir", dataset_dir,
        "--subjects", subject_id,
        "--save_uint16",
        "--normalization_method", norm_method,
        "--plot_psd"
    ]
    
    if apply_ica:
        cmd.append("--apply_ica")
    
    if apply_filter:
        cmd.append("--apply_filter")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_training(dataset_dir, subject_id, model, epochs, batch_size, lr, suffix, apply_normalization):
    """Run training script."""
    cmd = [
        "python", "train.py",
        "--dataset_dir", dataset_dir,
        "--subject_id", subject_id,
        "--model", model,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--suffix", suffix
    ]
    
    if apply_normalization:
        cmd.append("--apply_normalization")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    """Run all preprocessing and training experiments."""
    total_experiments = len(NORMALIZATION_METHODS) * len(ICA_OPTIONS) * len(FILTER_OPTIONS)
    current_experiment = 0
    
    print("=" * 60)
    print(f"Starting {total_experiments} experiments")
    print("=" * 60)
    
    for norm_method, apply_ica, apply_filter in product(NORMALIZATION_METHODS, ICA_OPTIONS, FILTER_OPTIONS):
        current_experiment += 1
        
        suffix = get_suffix(norm_method, apply_ica, apply_filter)
        
        print("\n" + "=" * 60)
        print(f"Experiment {current_experiment}/{total_experiments}")
        print(f"Configuration:")
        print(f"  Normalization: {norm_method}")
        print(f"  ICA: {apply_ica}")
        print(f"  Filter: {apply_filter}")
        print(f"  Suffix: {suffix}")
        print("=" * 60)
        
        # Run preprocessing
        print(f"\n[{current_experiment}/{total_experiments}] Running preprocessing...")
        preprocess_success = run_preprocessing(
            DATASET_DIR,
            SUBJECT_ID,
            norm_method,
            apply_ica,
            apply_filter
        )
        
        if not preprocess_success:
            print(f"ERROR: Preprocessing failed for configuration {suffix}")
            print("Skipping training for this configuration.")
            continue
        
        print(f"✓ Preprocessing completed successfully")
        
        # Determine if normalization should be applied during training
        # Apply normalization in training only if NOT applied during preprocessing
        apply_normalization_in_training = (norm_method == "none")
        
        # Run training
        print(f"\n[{current_experiment}/{total_experiments}] Running training...")
        training_success = run_training(
            DATASET_DIR,
            SUBJECT_ID,
            MODEL,
            EPOCHS,
            BATCH_SIZE,
            LR,
            suffix,
            apply_normalization_in_training
        )
        
        if not training_success:
            print(f"ERROR: Training failed for configuration {suffix}")
            continue
        
        print(f"✓ Training completed successfully")
        print(f"✓ Experiment {suffix} completed!")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()