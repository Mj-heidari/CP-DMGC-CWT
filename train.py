import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dataset.utils import *
from dataset.dataset import CHBMITDataset
from models.EEGNet import EEGNet
import os
import glob

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
            X = X.unsqueeze(1)
            outputs = self.model(X)
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
                X = X.unsqueeze(1)
                outputs = self.model(X)
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


def run_nested_cv(X, y, group_ids, model_builder,
                  n_inner_folds=5, batch_size=64, lr=1e-3, epochs=20):
    """
    model_builder: function that returns a *new model object*
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CHBMITDataset(X, y, group_ids)
    all_results = []

    sample_sec = 5.0  # each sample = 5 seconds

    for outer_idx in range(dataset.N):
        print(f"\n===== Outer Fold {outer_idx+1}/{dataset.N} =====")
        dataset.set_leave_out(outer_idx)

        # Split into test set
        dataset.set_split("test")
        X_test, y_test = dataset.X, dataset.y

        # Train pool
        X_train, y_train = dataset.X_train, dataset.y_train

        # Inner 5-fold CV
        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        test_probs_ensemble = []

        for inner_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\n  --- Inner Fold {inner_idx+1}/{n_inner_folds} ---")

            # Build datasets
            train_ds = TensorDataset(
                torch.tensor(X_train[tr_idx], dtype=torch.float32),
                torch.tensor(y_train[tr_idx], dtype=torch.long),
            )
            val_ds = TensorDataset(
                torch.tensor(X_train[val_idx], dtype=torch.float32),
                torch.tensor(y_train[val_idx], dtype=torch.long),
            )
            test_ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Model & trainer
            model = model_builder()
            trainer = Trainer(model, device=device)
            trainer.set_loss_weights(y_train[tr_idx])

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
        hours = (len(y_test) * sample_sec) / 3600.0
        fpr_per_hour = false_positives / hours if hours > 0 else np.nan

        print(f"\n==> Outer Fold {outer_idx+1} "
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
    subject_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sub-*")))

    all_results = {}

    for subj_path in subject_dirs:
        subj_id = os.path.basename(subj_path)
        print(f"\n############################")
        print(f"#### Processing {subj_id} ####")
        print(f"############################")

        # Collect all sessions for this subject
        ses_paths = glob.glob(os.path.join(subj_path, "ses-*", "eeg", "*.npz"))

        subj_X, subj_y, subj_group_ids = [], [], []
        for ses_path in ses_paths:
            data = np.load(ses_path)
            X, scales = data["X"], data["scales"]
            X = invert_uint16_scaling(X, scales)

            subj_X.append(X)
            subj_y.append(data["y"])
            subj_group_ids.append(data["group_ids"])

        # Concatenate sessions of this subject
        X = np.concatenate(subj_X, axis=0)
        y = np.concatenate(subj_y, axis=0)
        group_ids = np.concatenate(subj_group_ids, axis=0)

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

        # Run nested CV for this subject
        results = run_nested_cv(
            X=X,
            y=y,
            group_ids=group_ids,
            # model_class=MB_dMGC_CWTFFNet,
            model_builder=builder,
            n_inner_folds=5,
            batch_size=64,
            lr=1e-3,
            epochs=20,
        )

        all_results[subj_id] = results

    # Final summary
    print("\n\n==== Overall Results Across Subjects ====")
    for subj_id, results in all_results.items():
        print(f"{subj_id}: mean AUC={np.mean(results):.4f} | per-fold={results}")