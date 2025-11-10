import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from dataset.dataset import CHBMITDataset, MilDataloader, make_cv_splitter
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

import numpy as np
import warnings
import argparse
import os
import json
import logging
from datetime import datetime
import pickle

warnings.filterwarnings("ignore")

def mil_confident_loss(instance_logits, bag_label, T=10):
    weights = torch.softmax(instance_logits[:, 1] / T, dim=0).unsqueeze(-1)  
    bag_logit = (weights * instance_logits).sum(dim=0, keepdim=True)
    bag_loss = F.cross_entropy(bag_logit, bag_label.unsqueeze(0))
    total_loss = bag_loss 
    return total_loss


class Trainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu", run_dir=None, use_mil=False):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # No weighted loss
        self.run_dir = run_dir
        self.best_val_auc = -1
        self.best_model_path = None

        self.use_mil = use_mil


    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            if self.use_mil:
                B, N = X.shape[:2]  # batch_size, bag_size
                X_flat = X.view(B * N, *X.shape[2:])  # (B*N, C, T) or (B*N, features)
            else:
                X_flat = X

            outputs = self.model(X_flat)

            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)

            if self.use_mil:
                outputs = outputs.view(B, N, -1)
                total_bag_loss = 0.0
                bag_outputs = []
                for i in range(B):
                    # use the confident MIL loss
                    bag_loss = mil_confident_loss(outputs[i], y[i])
                    total_bag_loss += bag_loss

                    # store mean output for metrics
                    bag_out = torch.sigmoid(outputs[i]).mean(dim=0)
                    bag_outputs.append(bag_out)

                loss = total_bag_loss / B
                bag_outputs = torch.stack(bag_outputs)
                ys = y
            else:
                bag_outputs = outputs
                ys = y
                loss = self.criterion(bag_outputs, ys)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.shape[0]
            preds = torch.argmax(bag_outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(ys.cpu().numpy())

        avg_loss = total_loss / (len(train_loader) * train_loader.batch_size)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def evaluate(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        with torch.no_grad():
            for X, y in tqdm(loader, desc="Evaluating", leave=False):
                X, y = X.to(self.device), y.to(self.device)

                if self.use_mil:
                    B, N = X.shape[:2]  # batch_size, bag_size
                    X_flat = X.view(B * N, *X.shape[2:])  # (B*N, C, T) or (B*N, features)
                else:
                    X_flat = X
            
                outputs = self.model(X_flat)

                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(0)

                if self.use_mil:
                    outputs = outputs.view(B, N, -1)
                    total_bag_loss = 0.0
                    bag_outputs = []
                    for i in range(B):
                        # use the confident MIL loss
                        bag_loss = mil_confident_loss(outputs[i], y[i])
                        total_bag_loss += bag_loss

                        # store mean output for metrics
                        bag_out = torch.sigmoid(outputs[i]).mean(dim=0)
                        bag_outputs.append(bag_out)

                    loss = total_bag_loss / B
                    bag_outputs = torch.stack(bag_outputs)
                    ys = y
                else:
                    bag_outputs = outputs
                    ys = y
                    loss = self.criterion(bag_outputs, ys)


                if len(bag_outputs.shape) == 1:
                    bag_outputs = bag_outputs.unsqueeze(0)
 
                total_loss += loss.item() * X.shape[0]
                probs = torch.softmax(bag_outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(bag_outputs, dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(ys.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")
        return avg_loss, acc, auc, np.array(all_probs), np.array(all_preds), np.array(all_labels)

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
):
    """
    Perform nested cross-validation with improvements:
    - Uses best model instead of last epoch
    - Saves model checkpoints
    - Undersamples interictal data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default configs
    if outer_cv_params is None:
        outer_cv_params = {"mode": "leave_one_preictal", "method": "balanced", "random_state": 0}
    if inner_cv_params is None:
        inner_cv_params = {"mode": "stratified", "n_fold": 5, "shuffle": False, "random_state": 0}

    logging.info(f"Starting nested CV with {len(list(make_cv_splitter(dataset, **outer_cv_params)))} outer folds")

    # Store all results for later analysis
    cv_results = {
        'outer_folds': []
    }

    # Outer CV
    for fold, (train_val_dataset, test_dataset) in enumerate(make_cv_splitter(dataset, **outer_cv_params)):
        logging.info(f"\n===== Outer Fold {fold+1} =====")

        y_test = dataset.y[test_dataset.indices]
        
        fold_data = {
            'outer_fold': fold + 1,
            'test_indices': test_dataset.indices,
            'y_test': y_test.tolist(),
            'inner_folds': []
        }

        # Inner CV
        inner_splits = list(make_cv_splitter(train_val_dataset, **inner_cv_params))
        logging.info(f"Inner CV: {len(inner_splits)} folds")
        for inner_fold, (train_dataset, val_dataset) in enumerate(inner_splits):
            logging.info(f"  Inner Fold {inner_fold+1}")

            # Create undersampled train loader
            train_loader = MilDataloader(train_dataset, batch_size=batch_size, shuffle=True, bag_size=4)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model & trainer
            model = model_builder()
            trainer = Trainer(model, device=device, run_dir=run_dir, use_mil=True)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Training loop
            training_log = []

            for epoch in range(1, epochs + 1):
                trainer.use_mil = True
                tr_loss, tr_acc = trainer.train_one_epoch(train_loader, optimizer)
                trainer.use_mil = False
                val_loss, val_acc, val_auc, vprobs, vpreds, vlabels = trainer.evaluate(val_loader)
                
                # Compute Wasserstein distance between positive and negative probabilities
                pos_probs = vprobs[vlabels == 1]
                neg_probs = vprobs[vlabels == 0]
                val_wdist = wasserstein_distance(pos_probs, neg_probs)
             
                # Use Wasserstein distance as selection metric
                metric = val_wdist
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
                training_log.append(epoch_info)
                
                logging.info(f"Epoch {epoch:02d} | "
                          f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
                          f"Val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}")
                                

            # Load best model for test prediction
            trainer.load_best_model()
            
            # Get FINAL validation predictions from best model (for calibration)
            val_loss, val_acc, val_auc, val_probs, val_preds, val_labels = trainer.evaluate(val_loader)
            
            # Get test predictions
            test_loss, test_acc, test_auc, test_probs, test_preds, test_labels = trainer.evaluate(test_loader)
            
            # Store inner fold results
            inner_fold_data = {
                'inner_fold': inner_fold + 1,
                'best_val_auc': trainer.best_val_auc,
                'model_path': trainer.best_model_path,
                'training_log': training_log,
                'val_indices': val_dataset.indices,
                'val_probs': val_probs.tolist(),
                'val_labels': val_labels.tolist(),
                'test_probs': test_probs.tolist(),
                'test_labels': test_labels.tolist(),
            }
            
            fold_data['inner_folds'].append(inner_fold_data)
        
        cv_results['outer_folds'].append(fold_data)

    # Save raw results
    if run_dir:
        results_path = os.path.join(run_dir, 'raw_predictions.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(cv_results, f)
        logging.info(f"Saved raw predictions to {results_path}")
   
    return cv_results


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

    
    offline_transforms = []
    if args.model == 'EEGBandClassifier':
        from transforms.signal.filterbank import FilterBank
        filter_bank = FilterBank(
            band_dict={
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 14),
                "beta": (14, 30),
                "gamma": (30, 48),
            },
            sampling_rate=128,
            normalize_by_lowbands=False,
        )
        offline_transforms = [filter_bank]
    
    if args.model == 'EEGWaveNet':
        from transforms.signal.wavletfilterbank import WaveletFilterBank
        filter_bank = WaveletFilterBank(
            fs=128,
            combine_mode="concat_time",
        )
        offline_transforms = [filter_bank]

    dataset = CHBMITDataset(
        args.dataset_dir,
        use_uint16=args.use_uint16,
        offline_transforms=offline_transforms,
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
    )
    
    logging.info(f"Run completed. Results saved in {run_dir}")