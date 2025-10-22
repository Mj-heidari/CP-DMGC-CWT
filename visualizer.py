import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

class Visualizer:
    def __init__(self, run_dir="runs", metric_for_best="auc", only_curves=False, only_best=False):
        self.run_dir = run_dir
        self.metric_for_best = metric_for_best
        self.only_curves = only_curves
        self.only_best = only_best
        self.reset()

    def reset(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "val_f1": [],
            "tpr": [],
            "fpr": [],
        }
        self.best_metric = -np.inf
        self.best_epoch_data = None

    def _compute_tpr_fpr(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        return tpr, fpr

    def update(self, epoch, tr_loss, tr_acc, val_loss, val_acc, vprobs, vpreds, vlabels):
        val_auc = roc_auc_score(vlabels, vprobs)
        val_f1 = f1_score(vlabels, vpreds)
        tpr, fpr = self._compute_tpr_fpr(vlabels, vpreds)

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(tr_loss)
        self.history["train_acc"].append(tr_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_auc"].append(val_auc)
        self.history["val_f1"].append(val_f1)
        self.history["tpr"].append(tpr)
        self.history["fpr"].append(fpr)

        # Track best epoch
        metric_value = val_auc if self.metric_for_best == "auc" else val_f1
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch_data = (vprobs.copy(), vpreds.copy(), vlabels.copy(), epoch)

    def render(self, fold, inner_fold, vprobs, vpreds, vlabels):
        out_dir = os.path.join(self.run_dir, 'folds', f"fold_{fold}")
        os.makedirs(out_dir, exist_ok=True)

        # ---- 1. Learning curves ----
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = self.history["epoch"]
        ax.plot(epochs, self.history["val_acc"], label="Val Accuracy")
        ax.plot(epochs, self.history["val_auc"], label="Val AUC")
        # ax.plot(epochs, self.history["val_f1"], label="Val F1")
        ax.plot(epochs, self.history["tpr"], label="Sensitivity (TPR)")
        ax.plot(epochs, self.history["fpr"], label="False Alarm (FPR)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title(f"Learning Curves (Fold {fold}, Inner {inner_fold})")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"learning_curve_inner_{inner_fold}.png"))
        plt.close(fig)

        if not self.only_curves:
            # ---- 2. Probability series and histogram ----
            if not self.only_best:
                self._plot_probs_and_hist(vprobs, vpreds, vlabels, out_dir, f"last_inner_{inner_fold}")

            # ---- 3. Best epoch plots ----
            if self.best_epoch_data is not None:
                vprobs_b, vpreds_b, vlabels_b, best_ep = self.best_epoch_data
                self._plot_probs_and_hist(vprobs_b, vpreds_b, vlabels_b, out_dir, f"best_inner_{inner_fold}")

    def _plot_probs_and_hist(self, vprobs, vpreds, vlabels, out_dir, tag):
        # -- Probability series for label=1 samples
        idx_pos = np.where(vlabels == 1)[0]
        probs_pos = vprobs[idx_pos]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(probs_pos, color="blue")
        ax.set_title(f"Predicted Probabilities for Label=1 ({tag})")
        ax.set_xlabel("Sample index (label=1 subset)")
        ax.set_ylabel("Predicted Probability")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"probs_{tag}.png"))
        plt.close(fig)

        # -- Histogram for both classes
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vprobs[vlabels == 0], bins=20, alpha=0.6, label="Label=0", density=True)
        ax.hist(vprobs[vlabels == 1], bins=20, alpha=0.6, label="Label=1", density=True)
        ax.set_title(f"Predicted Probability Distribution ({tag})")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"hist_{tag}.png"))
        plt.close(fig)


def compute_prediction_correlation(test_probs_stack, tlabels, save_path=None, show=False):
    """
    Compute Pearson correlation between model predictions for label=1 (preictal) samples
    and plot it alongside the average ± std of predictions across models in time order.

    Args:
        test_probs_stack (np.ndarray): Shape (n_models, n_samples), predicted probabilities per model.
        tlabels (np.ndarray): Shape (n_samples,), true labels.
        save_path (str, optional): Path to save the plot.
        show (bool): If True, displays the plot interactively.

    Returns:
        pearson_corr (np.ndarray): Pearson correlation matrix.
    """
    mask = (tlabels == 1)
    if not np.any(mask):
        print("⚠️ No label=1 samples found in test set — correlation skipped.")
        return None

    probs_preictal = test_probs_stack[:, mask]
    n_models = probs_preictal.shape[0]
    n_samples = probs_preictal.shape[1]

    # --- Pearson correlation (linear)
    pearson_corr = np.corrcoef(probs_preictal)

    # --- Average and std across models for each preictal sample (in order)
    mean_probs = probs_preictal.mean(axis=0)
    std_probs = probs_preictal.std(axis=0)

    # --- Plot both
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: correlation matrix
    im = axes[0].imshow(pearson_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Pearson Correlation Between Models")
    axes[0].set_xlabel("Model Index")
    axes[0].set_ylabel("Model Index")

    for i in range(n_models):
        for j in range(n_models):
            axes[0].text(j, i, f"{pearson_corr[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    # Right: mean ± std plot
    x = np.arange(n_samples)
    axes[1].plot(x, mean_probs, color='blue', label='Mean prediction')
    axes[1].fill_between(x, mean_probs - std_probs, mean_probs + std_probs,
                         color='blue', alpha=0.2, label='±1 STD')
    axes[1].set_title("Preictal Predictions Across Models")
    axes[1].set_xlabel("Preictal Sample Index (time order)")
    axes[1].set_ylabel("Predicted Probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    elif show:
        plt.show()

    return pearson_corr
