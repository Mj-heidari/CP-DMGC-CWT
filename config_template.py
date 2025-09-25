import json
import argparse
from pathlib import Path

def load_config_from_file(config_path):
    """Load configuration from JSON file and merge with command line arguments"""
    parser = argparse.ArgumentParser(description='Load config from file')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Parse only the config argument first
    args, remaining = parser.parse_known_args()
    
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Now parse all arguments with defaults from config
    parser = argparse.ArgumentParser(description='Nested Cross-Validation for Seizure Prediction')
    
    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, 
                       default=config.get('dataset_dir', 'data/BIDS_CHB-MIT'),
                       help='Path to dataset directory')
    parser.add_argument('--subject_id', type=str, 
                       default=config.get('subject_id', '01'),
                       help='Subject ID to process')
    parser.add_argument('--suffix', type=str, 
                       default=config.get('suffix', 'None_F_F'),
                       help='Dataset suffix')
    parser.add_argument('--use_uint16', action='store_true',
                       default=config.get('use_uint16', False),
                       help='Use uint16 data format')
    
    # Model parameters
    parser.add_argument('--model', type=str, 
                       default=config.get('model', 'CE-stSENet'),
                       help='Model name')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, 
                       default=config.get('batch_size', 64),
                       help='Batch size')
    parser.add_argument('--lr', type=float, 
                       default=config.get('lr', 1e-3),
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                       default=config.get('epochs', 20),
                       help='Number of epochs')
    
    # Cross-validation parameters
    parser.add_argument('--outer_cv_mode', type=str, 
                       default=config.get('outer_cv_mode', 'leave_one_preictal'),
                       choices=['leave_one_preictal', 'stratified'],
                       help='Outer CV mode')
    parser.add_argument('--outer_cv_method', type=str, 
                       default=config.get('outer_cv_method', 'balanced_shuffled'),
                       choices=['balanced', 'balanced_shuffled', 'nearest'],
                       help='Outer CV method (for leave_one_preictal)')
    parser.add_argument('--inner_cv_mode', type=str, 
                       default=config.get('inner_cv_mode', 'leave_one_preictal'),
                       choices=['leave_one_preictal', 'stratified'],
                       help='Inner CV mode')
    parser.add_argument('--inner_cv_method', type=str, 
                       default=config.get('inner_cv_method', 'balanced_shuffled'),
                       choices=['balanced', 'balanced_shuffled', 'nearest'],
                       help='Inner CV method (for leave_one_preictal)')
    parser.add_argument('--n_fold', type=int, 
                       default=config.get('n_fold', 5),
                       help='Number of folds (for stratified CV)')
    
    # Other parameters
    parser.add_argument('--random_state', type=int, 
                       default=config.get('random_state', 42),
                       help='Random state for reproducibility')
    parser.add_argument('--moving_avg_window', type=int, 
                       default=config.get('moving_avg_window', 3),
                       help='Moving average window size')
    
    return parser.parse_args()

# Example configuration files

# config_basic.json
basic_config = {
    "dataset_dir": "data/BIDS_CHB-MIT",
    "subject_id": "01",
    "model": "CE-stSENet",
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    "outer_cv_mode": "leave_one_preictal",
    "outer_cv_method": "balanced_shuffled",
    "inner_cv_mode": "leave_one_preictal",
    "inner_cv_method": "balanced_shuffled",
    "random_state": 42,
    "moving_avg_window": 3
}

# config_eegnet.json
eegnet_config = {
    "dataset_dir": "data/BIDS_CHB-MIT",
    "subject_id": "02",
    "model": "EEGNet",
    "epochs": 50,
    "batch_size": 32,
    "lr": 5e-4,
    "outer_cv_mode": "leave_one_preictal",
    "outer_cv_method": "nearest",
    "inner_cv_mode": "stratified",
    "n_fold": 5,
    "random_state": 123,
    "moving_avg_window": 5,
    "use_uint16": True
}

# config_stratified.json
stratified_config = {
    "dataset_dir": "data/BIDS_CHB-MIT",
    "subject_id": "*",
    "model": "CE-stSENet",
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "outer_cv_mode": "stratified",
    "inner_cv_mode": "stratified",
    "n_fold": 5,
    "random_state": 42,
    "moving_avg_window": 3
}

if __name__ == "__main__":
    # Create example config files
    configs = {
        "config_basic.json": basic_config,
        "config_eegnet.json": eegnet_config,
        "config_stratified.json": stratified_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created {filename}")
    
    print("\nTo use a config file, run:")
    print("python train.py --config config_basic.json")
    print("\nYou can still override individual parameters:")
    print("python train.py --config config_basic.json --epochs 50 --lr 5e-4")