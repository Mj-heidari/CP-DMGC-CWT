import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from tqdm import tqdm
from models.MB_dMGC_CWTFFNet import MB_dMGC_CWTFFNet
import numpy as np
from dataset.utils import *
from dataset.dataset import CHBMITDataset, leave_one_preictal_group_out


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
            auc = float("nan")
        report = classification_report(all_labels, all_preds, digits=4)
        return avg_loss, acc, auc, report, all_probs


def run_nested_cv(dataset, model_class, 
                  batch_size=64, lr=1e-3, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = []

    for fold, (train_val_dataset, test_dataset) in enumerate(leave_one_preictal_group_out(dataset)):
        print(f"\n===== Outer Fold {fold+1} =====")        
        test_probs_ensemble = []
        y_test = dataset.y[test_dataset.indices]

        for inner_fold, (train_dataset, val_dataset) in enumerate(leave_one_preictal_group_out(train_val_dataset)):
            print(f"\n  --- Inner Fold {inner_fold+1} ---")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Model & trainer
            model = model_class()
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
        print(f"\n==> Outer Fold {fold+1} AUC={outer_auc:.4f}")
        all_results.append(outer_auc)

    print("\n==== Final Results ====")
    print("Per-fold AUC:", all_results)
    print("Mean AUC:", np.mean(all_results))
    return all_results


if __name__ == "__main__":
    dataset_dir = "data/BIDS_CHB-MIT"
    dataset = CHBMITDataset(dataset_dir, offline_transforms=[])

    # Run nested CV for this subject
    results = run_nested_cv(
        dataset=dataset,
        model_class=MB_dMGC_CWTFFNet,
        batch_size=64,
        lr=1e-3,
        epochs=20,
    )

    # Final summary
    print("\n\n==== Overall Results Across Subjects ====")
    for subj_id, results in results.items():
        print(f"{subj_id}: mean AUC={np.mean(results):.4f} | per-fold={results}")
