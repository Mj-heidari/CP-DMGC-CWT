import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
from models.MB_dMGC_CWTFFNet import MB_dMGC_CWTFFNet
import os
import glob
import numpy as np
from dataset.utils import *
from dataset.dataset import CHBMITDataset

class Trainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
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
            auc = float("nan")  # if only one class present
        report = classification_report(all_labels, all_preds, digits=4)
        return avg_loss, acc, auc, report

def run_cross_validation(X, y, group_ids, model_class, n_folds=5, 
                         batch_size=64, lr=1e-3, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_metrics = []

    for fold_idx in range(n_folds):
        print(f"\n===== Fold {fold_idx+1}/{n_folds} =====")
        dataset = CHBMITDataset(X, y, group_ids, fold_idx=fold_idx, n_folds=n_folds)

        # Train loader
        dataset.set_split("train")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Test loader
        dataset.set_split("test")
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Model & trainer
        model = model_class()
        trainer = Trainer(model, device=device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = trainer.train_one_epoch(train_loader, optimizer)
            val_loss, val_acc, val_auc, _ = trainer.evaluate(test_loader)

            print(f"Epoch {epoch:02d}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")

        # Final evaluation
        test_loss, test_acc, test_auc, report = trainer.evaluate(test_loader)
        print("\nClassification Report:\n", report)

        fold_metrics.append({
            "fold": fold_idx,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc,
        })

    return fold_metrics

if __name__ == "__main__":

    dataset_dir = "data/BIDS_CHB-MIT"
    sessions_pathes = glob.glob(os.path.join(dataset_dir, "*", "*", 'eeg', '*'))
    for sessions_path in sessions_pathes:
        data = np.load(sessions_path)
        X, scales = data["X"], data["scales"]

        # Reconstruct float32 EEG
        X = invert_uint16_scaling(X, scales)

    X = X[:]
    y = data['y'][:]
    group_ids = data["group_ids"][:]


    metrics = run_cross_validation(
        X=X,
        y=y,
        group_ids=group_ids,
        model_class=MB_dMGC_CWTFFNet,
        n_folds=5,
        batch_size=64,
        lr=1e-3,
        epochs=20
    )

    print("\n===== Cross-validation results =====")
    for m in metrics:
        print(f"Fold {m['fold']}: "
            f"Acc={m['test_acc']:.4f}, AUC={m['test_auc']:.4f}, Loss={m['test_loss']:.4f}")

    mean_auc = np.nanmean([m["test_auc"] for m in metrics])
    print(f"\nMean AUC across folds: {mean_auc:.4f}")
