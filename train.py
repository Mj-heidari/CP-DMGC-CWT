import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
from dataset.utils import *
from dataset.dataset import CHBMITDataset, leave_one_preictal_group_out, cross_validation

from models.EEGNet import EEGNet
from models.CE_stSENet.CE_stSENet import CE_stSENet

import numpy as np
import warnings
warnings.filterwarnings("ignore")



class Trainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = None  # will set dynamically per fold

    def set_loss_weights(self, y_train):
        """Set class weights dynamically based on training set distribution."""
        class_counts = np.bincount(y_train)
        # inverse frequency weighting
        weights = class_counts.sum() / (len(class_counts) * class_counts)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            if self.model.__class__.__name__ == 'CE_stSENet':
                X = X.unsqueeze(2)
            elif self.model.__class__.__name__ == 'EEGNet':
                X = X.unsqueeze(1)
            outputs = self.model(X)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def evaluate(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

        with torch.no_grad():
            for X, y in tqdm(loader, desc="Evaluating", leave=False):
                X, y = X.to(self.device), y.to(self.device)

                # X: (batch, channels, data_points)
                mean = X.mean(dim=(0, 2), keepdim=True)   # mean per channel
                std = X.std(dim=(0, 2), keepdim=True)     # std per channel

                X = (X - mean) / (std + 1e-6)  # avoid div by 0
        
                if self.model.__class__.__name__ == 'CE_stSENet':
                    X = X.unsqueeze(2)
                elif self.model.__class__.__name__ == 'EEGNet':
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


def run_nested_cv(dataset, model_builder, 
                  batch_size=64, lr=1e-3, epochs=20, split = 'cross validation'):
    """
    model_builder: function that returns a *new model object*
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []



    for fold, (train_val_dataset, test_dataset) in enumerate(leave_one_preictal_group_out(dataset, shuffle=False)):
        print(f"\n===== Outer Fold {fold+1} =====")        

        test_probs_ensemble = []
        y_test = dataset.y[test_dataset.indices]


        if split == 'cross validation':
            split_method = cross_validation(train_val_dataset)
        else:
            split_method = leave_one_preictal_group_out(train_val_dataset, shuffle=False)
        
        for inner_fold, (train_dataset, val_dataset) in enumerate(split_method):
            print(f"\n  --- Inner Fold {inner_fold+1} ---")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model & trainer
            model = model_builder()
            trainer = Trainer(model, device=device)
            trainer.set_loss_weights(dataset.y[train_dataset.indices])

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Training loop
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = trainer.train_one_epoch(train_loader, optimizer)
                val_loss, val_acc, val_auc, _, _, _, _ = trainer.evaluate(val_loader)
                scheduler.step()
                print(f"Epoch {epoch:02d} | "
                      f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
                      f"Val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}")

            # Predict test set for ensemble
            _, _, _, _, test_probs, _, _ = trainer.evaluate(test_loader)
            test_probs_ensemble.append(test_probs)

        # Ensemble (mean across inner models)
        final_probs = np.mean(test_probs_ensemble, axis=0)

        final_preds = (final_probs >= 0.5).astype(int)

        # === Metrics on outer test ===
        auc = roc_auc_score(y_test, final_probs)

        # Classification report
        report = classification_report(y_test, final_preds, digits=4)

        # Sensitivity: at least one preictal detected
        has_preictal = np.any(y_test == 1)
        if has_preictal:
            detected = np.any((y_test == 1) & (final_preds == 1))
            sensitivity = 1 if detected else 0
        else:
            sensitivity = np.nan  # no seizures in test

        # FPR/h
        false_positives = np.sum((y_test == 0) & (final_preds == 1))
        hours = (len(y_test) * 5) / 3600.0
        fpr_per_hour = false_positives / hours if hours > 0 else np.nan

        print(f"\n==> Outer Fold {fold+1} "
              f"AUC={auc:.4f}, Sensitivity={sensitivity}, FPR/h={fpr_per_hour:.4f}")
        print(report)

        all_results.append({
            "auc": auc,
            "sensitivity": sensitivity,
            "fpr_per_hour": fpr_per_hour,
            'report': report
        })

    # Summary
    aucs = [r["auc"] for r in all_results]
    sens = [r["sensitivity"] for r in all_results if not np.isnan(r["sensitivity"])]
    fprs = [r["fpr_per_hour"] for r in all_results]

    print("\n==== Final Results ====")
    print("Per-fold:", all_results)
    print(f"Mean AUC={np.mean(aucs):.4f}, "
          f"Mean Sensitivity={np.mean(sens):.4f}, "
          f"Mean FPR/h={np.mean(fprs):.4f}")

    return all_results

def model_builder(model_class, **kwargs):
    """
    Returns a function that builds a fresh model instance.
    This avoids weight leakage across folds.

    Args:
        model_class: torch.nn.Module class (e.g., EEGNet)
        kwargs: parameters to initialize the model

    Returns:
        A callable that builds a new model each time it's called
    """
    def build():
        return model_class(**kwargs)
    return build


if __name__ == "__main__":
    dataset_dir = "data/BIDS_CHB-MIT"
    dataset = CHBMITDataset(dataset_dir, use_uint16=True, offline_transforms=[])
    model = 'CE-stSENet'

    if model == 'EEGNet':
        builder = model_builder(
            EEGNet,
            chunk_size=640,
            num_electrodes=18,
            dropout=0.5,
            kernel_1=64,
            kernel_2=16,
            F1=8,
            F2=16,
            D=2,
            num_classes=2,
        )
    elif model == 'CE-stSENet':
        builder = model_builder(
            CE_stSENet,
            inc=18,
            class_num=2,
            si=128,
        )

    # Run nested CV for this subject
    results = run_nested_cv(
        dataset=dataset,
        model_builder=builder,
        batch_size=64,
        lr=1e-3,
        epochs=5,
    )

    # Final summary
    print("\n\n==== Overall Results Across Subjects ====")
    for subj_id, results in results.items():
        print(f"{subj_id}: mean AUC={np.mean(results):.4f} | per-fold={results}")
