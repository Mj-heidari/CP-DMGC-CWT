import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from dataset.dataset import CHBMITDataset, UnderSampledDataLoader, make_cv_splitter

import numpy as np
import warnings
import argparse
import os
import json
import logging
from datetime import datetime
import pickle

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", run_dir=None, use_mil=False, bag_size=1, bag_step=1, max_size=8):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # No weighted loss
        self.run_dir = run_dir
        self.best_val_auc = -1
        self.best_model_path = None

        self.use_mil = use_mil
        self.max_size = max_size
        self.bag_size = max(bag_size,max_size)
        self.bag_step = bag_step

    def aggregate_outputs(self, outputs, labels, bag_size=8, aggregator_1="max", aggregator_0="mean"):
        """
        Group samples in the batch by their label (0 or 1) and aggregate outputs
        in smaller bags (e.g., size 8). The last bag can be smaller if not divisible.
        Returns aggregated predictions and labels for loss computation.
        """
        aggregated = []
        aggregated_labels = []

        if bag_size == 1:
            return outputs, labels
        
        for label in [0, 1]:
            mask = labels == label
            if not torch.any(mask):
                continue
            
            aggregator = aggregator_1 if label == 1 else aggregator_0

            group_outputs = outputs[mask]  # (n_samples_in_group, num_classes)
            n = group_outputs.size(0)

            # Split group into smaller bags
            for i in range(0, n, bag_size):
                bag = group_outputs[i:i + bag_size]

                if aggregator == "max":
                    agg_out, _ = torch.max(bag, dim=0)
                elif aggregator == "mean":
                    agg_out = torch.mean(bag, dim=0)

                aggregated.append(agg_out.unsqueeze(0))
                aggregated_labels.append(torch.tensor([label], device=labels.device))

        if len(aggregated) == 0:
            # Fallback to original outputs if no aggregation was done
            return outputs, labels

        aggregated = torch.cat(aggregated, dim=0)
        aggregated_labels = torch.cat(aggregated_labels, dim=0)
        return aggregated, aggregated_labels


    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            if self.model.__class__.__name__ == 'CE_stSENet':
                X = X.unsqueeze(2)
            elif self.model.__class__.__name__ == 'EEGNet' or self.model.__class__.__name__ == 'TSception':
                X = X.unsqueeze(1)
            outputs = self.model(X)

            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            
            if self.use_mil:
                outputs, y = self.aggregate_outputs(outputs, y, bag_size=self.bag_size)

            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / (len(train_loader) * train_loader.batch_size)
        acc = accuracy_score(all_labels, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel().tolist()
        TRP = tp / (tp + fn)
        FPR = fp / (fp + tn)
        self.bag_size = max(self.bag_size + self.bag_step, self.max_size)
        return avg_loss, acc, TRP, FPR

    def evaluate(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        with torch.no_grad():
            for X, y in tqdm(loader, desc="Evaluating", leave=False):
                X, y = X.to(self.device), y.to(self.device)
        
                if self.model.__class__.__name__ == 'CE_stSENet':
                    X = X.unsqueeze(2)
                elif self.model.__class__.__name__ == 'EEGNet' or self.model.__class__.__name__ == 'TSception':
                    X = X.unsqueeze(1)
                outputs = self.model(X)

                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(0)
                
                
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * X.size(0)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")
        report = classification_report(all_labels, all_preds, digits=4)
        return avg_loss, acc, auc, report, np.array(all_probs), np.array(all_preds), np.array(all_labels)

    def save_checkpoint(self, epoch, metric, outer_fold, inner_fold):
        """Save model checkpoint if it's the best so far"""
        if metric > self.best_val_auc:
            self.best_val_auc = metric
            if self.run_dir:
                checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.best_model_path = os.path.join(
                    checkpoint_dir, f'best_model_outer{outer_fold}_inner{inner_fold}.pth'
                )
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info(f"Saved best model at epoch {epoch} with metric={metric:.4f}")
    
    def load_best_model(self):
        """Load the best saved model"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            logging.info(f"Loaded best model from {self.best_model_path}")


def moving_average_predictions(probs, window_size=3):
    """Apply moving average to predictions"""
    if len(probs) < window_size:
        return probs
    
    smoothed_probs = np.copy(probs).astype(float)
    
    # For the first samples, use available data
    for i in range(len(probs)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        smoothed_probs[i] = np.mean(probs[start_idx:end_idx])
    
    return smoothed_probs


def setup_run_directory():
    """Create a new run directory with timestamp"""
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    # Find next run number or use timestamp
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith('run')]
    if existing_runs:
        run_numbers = []
        for run_dir in existing_runs:
            try:
                num = int(run_dir.split('_')[0].replace('run', ''))
                run_numbers.append(num)
            except ValueError:
                continue
        next_run = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_run = 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"run{next_run}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def setup_logging(run_dir):
    """Setup logging configuration"""
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def save_run_config(run_dir, args, model_name):
    """Save run configuration and parameters"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'arguments': vars(args),
        'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    }
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Saved run configuration to {config_path}")


def run_nested_cv(
    dataset,
    model_builder,
    batch_size=64,
    lr=1e-3,
    epochs=20,
    outer_cv_params=None,
    inner_cv_params=None,
    run_dir=None,
    moving_avg_window=3,
):
    """
    Perform nested cross-validation with improvements:
    - Uses best model instead of last epoch
    - Saves model checkpoints
    - Applies moving average to predictions
    - Undersamples interictal data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []
    detailed_results = []

    # Default configs
    if outer_cv_params is None:
        outer_cv_params = {"mode": "leave_one_preictal", "method": "balanced", "random_state": 0}
    if inner_cv_params is None:
        inner_cv_params = {"mode": "stratified", "n_fold": 5, "shuffle": False, "random_state": 0}

    logging.info(f"Starting nested CV with {len(list(make_cv_splitter(dataset, **outer_cv_params)))} outer folds")

    # Outer CV
    for fold, (train_val_dataset, test_dataset) in enumerate(make_cv_splitter(dataset, **outer_cv_params)):
        logging.info(f"\n===== Outer Fold {fold+1} =====")

        test_probs_ensemble = []
        y_test = dataset.y[test_dataset.indices]
        
        fold_results = {
            'outer_fold': fold + 1,
            'inner_fold_results': [],
            'test_indices': test_dataset.indices,
        }

        # Inner CV
        inner_splits = list(make_cv_splitter(train_val_dataset, **inner_cv_params))
        logging.info(f"Inner CV: {len(inner_splits)} folds")
        
        for inner_fold, (train_dataset, val_dataset) in enumerate(inner_splits):
            logging.info(f"  Inner Fold {inner_fold+1}")

            # Create undersampled train loader
            train_loader = UnderSampledDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model & trainer
            model = model_builder()
            trainer = Trainer(model, device=device, run_dir=run_dir, use_mil=True, bag_size=8, bag_step=0)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Training loop
            inner_fold_log = []
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc, TPR, FPR = trainer.train_one_epoch(train_loader, optimizer)
                val_loss, val_acc, val_auc, _, _, all_preds, all_labels = trainer.evaluate(val_loader)
                
                tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel().tolist()
                TPR_val = tp / (tp + fn)
                FPR_val = fp / (fp + tn)
                metric = TPR_val/(FPR_val+10e-5)
                # Save best model checkpoint
                trainer.save_checkpoint(epoch, metric, fold+1, inner_fold+1)
                
                scheduler.step()
                
                epoch_info = {
                    'epoch': epoch,
                    'train_loss': tr_loss,
                    'train_acc': tr_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_auc': val_auc
                }
                inner_fold_log.append(epoch_info)
                
                logging.info(f"Epoch {epoch:02d} | "
                          f"Train {tr_loss:.4f}/{TPR:.4f}/{FPR:.4f}/{TPR/(FPR+10e-5):.4f} | "
                          f"Val {val_loss:.4f}/{val_auc:.4f}/{TPR_val:.4f}/{FPR_val:.4f}/{TPR_val/(FPR_val+10e-5):.4f}")

            # Load best model for test prediction
            trainer.load_best_model()
            
            # Predict test set for ensemble
            _, _, _, _, test_probs, _, _ = trainer.evaluate(test_loader)
            test_probs_ensemble.append(test_probs)
            
            fold_results['inner_fold_results'].append({
                'inner_fold': inner_fold + 1,
                'best_val_auc': trainer.best_val_auc,
                'epoch_logs': inner_fold_log,
                'model_path': trainer.best_model_path
            })

        # Ensemble (mean across inner models)
        final_probs = np.mean(test_probs_ensemble, axis=0)
        
        # Apply moving average
        final_probs_ma = moving_average_predictions(final_probs, moving_avg_window)
        
        # Predictions
        final_preds = (final_probs >= 0.5).astype(int)
        final_preds_ma = (final_probs_ma >= 0.5).astype(int)

        # === Metrics on outer test ===
        auc = roc_auc_score(y_test, final_probs)
        auc_ma = roc_auc_score(y_test, final_probs_ma)

        # Classification reports
        report = classification_report(y_test, final_preds, digits=4)
        report_ma = classification_report(y_test, final_preds_ma, digits=4)

        # Sensitivity: at least one preictal detected
        has_preictal = np.any(y_test == 1)
        if has_preictal:
            detected = np.any((y_test == 1) & (final_preds == 1))
            detected_ma = np.any((y_test == 1) & (final_preds_ma == 1))
            sensitivity = 1 if detected else 0
            sensitivity_ma = 1 if detected_ma else 0
        else:
            sensitivity = np.nan
            sensitivity_ma = np.nan

        # FPR/h
        false_positives = np.sum((y_test == 0) & (final_preds == 1))
        false_positives_ma = np.sum((y_test == 0) & (final_preds_ma == 1))
        hours = (len(y_test) * 5) / 3600.0
        fpr_per_hour = false_positives / hours if hours > 0 else np.nan
        fpr_per_hour_ma = false_positives_ma / hours if hours > 0 else np.nan

        logging.info(f"\n==> Outer Fold {fold+1}")
        logging.info(f"Raw: AUC={auc:.4f}, Sensitivity={sensitivity}, FPR/h={fpr_per_hour:.4f}")
        logging.info(f"MA:  AUC={auc_ma:.4f}, Sensitivity={sensitivity_ma}, FPR/h={fpr_per_hour_ma:.4f}")
        logging.info(f"Raw Classification Report:\n{report}")
        logging.info(f"MA Classification Report:\n{report_ma}")

        fold_result = {
            "fold": fold + 1,
            "raw": {
                "auc": auc,
                "sensitivity": sensitivity,
                "fpr_per_hour": fpr_per_hour,
                "report": report
            },
            "moving_average": {
                "auc": auc_ma,
                "sensitivity": sensitivity_ma,
                "fpr_per_hour": fpr_per_hour_ma,
                "report": report_ma
            },
            "predictions": {
                "final_probs": final_probs.tolist(),
                "final_probs_ma": final_probs_ma.tolist(),
                "y_test": y_test.tolist()
            }
        }
        all_results.append(fold_result)
        detailed_results.append(fold_results)

    # Save detailed results
    if run_dir:
        results_path = os.path.join(run_dir, 'detailed_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        
        results_json_path = os.path.join(run_dir, 'results.json')
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Summary statistics
    raw_aucs = [r["raw"]["auc"] for r in all_results]
    ma_aucs = [r["moving_average"]["auc"] for r in all_results]
    
    raw_sens = [r["raw"]["sensitivity"] for r in all_results if not np.isnan(r["raw"]["sensitivity"])]
    ma_sens = [r["moving_average"]["sensitivity"] for r in all_results if not np.isnan(r["moving_average"]["sensitivity"])]
    
    raw_fprs = [r["raw"]["fpr_per_hour"] for r in all_results]
    ma_fprs = [r["moving_average"]["fpr_per_hour"] for r in all_results]

    logging.info("\n==== Final Results ====")
    logging.info("Raw Results:")
    logging.info(f"  Mean AUC={np.mean(raw_aucs):.4f} ± {np.std(raw_aucs):.4f}")
    logging.info(f"  Mean Sensitivity={np.mean(raw_sens):.4f} ± {np.std(raw_sens):.4f}")
    logging.info(f"  Mean FPR/h={np.mean(raw_fprs):.4f} ± {np.std(raw_fprs):.4f}")
    
    logging.info("Moving Average Results:")
    logging.info(f"  Mean AUC={np.mean(ma_aucs):.4f} ± {np.std(ma_aucs):.4f}")
    logging.info(f"  Mean Sensitivity={np.mean(ma_sens):.4f} ± {np.std(ma_sens):.4f}")
    logging.info(f"  Mean FPR/h={np.mean(ma_fprs):.4f} ± {np.std(ma_fprs):.4f}")

    return all_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Nested Cross-Validation for Seizure Prediction')

    # Dataset parameters
    parser.add_argument('--dataset_dir', type=str, default='data/BIDS_CHB-MIT',
                       help='Path to dataset directory')
    parser.add_argument('--subject_id', type=str, default='01',
                       help='Subject ID to process')
    parser.add_argument('--suffix', type=str, default='None_F_F',
                       help='Dataset suffix')
    parser.add_argument('--use_uint16', action='store_true',
                       help='Use uint16 data format')
    parser.add_argument('--apply_normalization', action='store_true',
                       help='Apply InstanceNormTransform')
    # Model parameters
    parser.add_argument('--model', type=str, default='CE-stSENet',
                       help='Model name')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    
    # Cross-validation parameters
    parser.add_argument('--outer_cv_mode', type=str, default='leave_one_preictal',
                       choices=['leave_one_preictal', 'stratified'],
                       help='Outer CV mode')
    parser.add_argument('--outer_cv_method', type=str, default='balanced_shuffled',
                       choices=['balanced', 'balanced_shuffled', 'nearest'],
                       help='Outer CV method (for leave_one_preictal)')
    parser.add_argument('--inner_cv_mode', type=str, default='leave_one_preictal',
                       choices=['leave_one_preictal', 'stratified'],
                       help='Inner CV mode')
    parser.add_argument('--inner_cv_method', type=str, default='balanced_shuffled',
                       choices=['balanced', 'balanced_shuffled', 'nearest'],
                       help='Inner CV method (for leave_one_preictal)')
    parser.add_argument('--n_fold', type=int, default=5,
                       help='Number of folds (for stratified CV)')
    
    # Other parameters
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--moving_avg_window', type=int, default=3,
                       help='Moving average window size')
    
    # Config 
    parser.add_argument('--config', type=str, default="",
                        help='Path to json config file')
    
    args = parser.parse_args()
    if args.config:
        try:
            with open(args.config, 'r') as file:
                data: dict = json.load(file)
            for key, val in data.items():
                if hasattr(args, key):
                    setattr(args, key, val)
        except Exception as e:
            print("An exception occurred while opening config file. Error:", e)
        
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    # Setup run directory and logging
    run_dir = setup_run_directory()
    setup_logging(run_dir)
    
    logging.info(f"Starting new run in directory: {run_dir}")
    
    # Save configuration
    save_run_config(run_dir, args, args.model)
    
    # Load dataset
    from transforms.signal.normalize import InstanceNormTransform
    from models.provider import get_builder
    
    if args.apply_normalization:
        online_transforms = [InstanceNormTransform()]
    else:
        online_transforms = []

    dataset = CHBMITDataset(
        args.dataset_dir,
        use_uint16=args.use_uint16,
        offline_transforms=[],
        online_transforms=online_transforms,
        suffix=args.suffix,
        subject_id=args.subject_id
    )
    
    logging.info(f"Loaded dataset with {len(dataset)} samples")
    logging.info(f"Class distribution: {np.bincount(dataset.y)}")
    
    # Get model builder
    builder = get_builder(model=args.model)
    
    # Setup CV parameters
    outer_cv_params = {
        "mode": args.outer_cv_mode,
        "random_state": args.random_state,
    }
    if args.outer_cv_mode == "leave_one_preictal":
        outer_cv_params["method"] = args.outer_cv_method
    elif args.outer_cv_mode == "stratified":
        outer_cv_params.update({
            "n_fold": args.n_fold,
            "shuffle": True
        })
    
    inner_cv_params = {
        "mode": args.inner_cv_mode,
        "random_state": args.random_state,
    }
    if args.inner_cv_mode == "leave_one_preictal":
        inner_cv_params["method"] = args.inner_cv_method
    elif args.inner_cv_mode == "stratified":
        inner_cv_params.update({
            "n_fold": args.n_fold,
            "shuffle": False
        })
    
    # Run nested CV
    results = run_nested_cv(
        dataset, builder,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        outer_cv_params=outer_cv_params,
        inner_cv_params=inner_cv_params,
        run_dir=run_dir,
        moving_avg_window=args.moving_avg_window
    )
    
    logging.info(f"Run completed. Results saved in {run_dir}")