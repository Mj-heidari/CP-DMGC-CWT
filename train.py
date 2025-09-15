import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
from models.MB_dMGC_CWTFFNet import MB_dMGC_CWTFFNet
from models.EEGNet import EEGNet

import os
import glob
import numpy as np
from dataset.utils import *
from dataset.dataset import CHBMITDataset


class Trainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0], device=device))
        # self.criterion = nn.CrossEntropyLoss()

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
        return avg_loss, acc, auc, report, all_probs


def run_nested_cv(X, y, group_ids, model_class, 
                  n_inner_folds=5, batch_size=64, lr=1e-3, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CHBMITDataset(X, y, group_ids)
    all_results = []

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

            # Build train/val/test datasets
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
            # model = model_class()
            model = model_class(chunk_size=640,
                        num_electrodes=18,
                        dropout=0.5,
                        kernel_1=64,
                        kernel_2=16,
                        F1=8,
                        F2=16,
                        D=2,
                        num_classes=2)

            trainer = Trainer(model, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training loop
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = trainer.train_one_epoch(train_loader, optimizer)
                val_loss, val_acc, val_auc, _ , _= trainer.evaluate(val_loader)
                print(f"Epoch {epoch:02d} | "
                      f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
                      f"Val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}")

            # Predict test set for ensemble
            _, _, _, _, test_probs = trainer.evaluate(test_loader)
            test_probs_ensemble.append(test_probs)

        # Ensemble (mean across inner models)
        final_probs = np.mean(test_probs_ensemble, axis=0)
        outer_auc = roc_auc_score(y_test, final_probs)
        print(f"\n==> Outer Fold {outer_idx+1} AUC={outer_auc:.4f}")
        all_results.append(outer_auc)

    print("\n==== Final Results ====")
    print("Per-fold AUC:", all_results)
    print("Mean AUC:", np.mean(all_results))
    return all_results


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

        # Run nested CV for this subject
        results = run_nested_cv(
            X=X,
            y=y,
            group_ids=group_ids,
            # model_class=MB_dMGC_CWTFFNet,
            model_class=EEGNet,
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
