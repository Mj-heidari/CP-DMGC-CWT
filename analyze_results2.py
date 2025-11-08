"""
Comprehensive Analysis Script for Seizure Prediction Results

This script performs all post-processing, metrics computation, and visualization
on raw predictions from train.py. It supports:
- Multiple calibration methods
- Multiple moving average windows
- Multiple thresholds
- Ensemble strategies
- Comprehensive metrics and visualizations
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, 
    average_precision_score, confusion_matrix, classification_report, f1_score
)
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
import os
from probability_calibration import calibrate_ensemble
warnings.filterwarnings('ignore')

# ============================================================================
# TRAINING VISUALIZER (from visualizer.py)
# ============================================================================

class TrainingVisualizer:
    """Visualizer for training curves and validation predictions"""
    
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
        metric_value = val_auc if self.metric_for_best == "auc" else -val_loss
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
        ax.plot(epochs, self.history["val_loss"], label="Val Loss")
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


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def moving_average(probs: np.ndarray, y, window_size: int = 3) -> np.ndarray:
    """Apply moving average smoothing to predictions"""
    if len(probs) < window_size or window_size == 1:
        return probs
    
    y = np.array(y)
    pre_mask = y == 1
    inter_mask = ~pre_mask
    
    smoothed_probs = np.zeros_like(probs).astype(float)

    def moving_avg(target_probs, win_size):
        temp = np.zeros_like(target_probs)
        for i in range(win_size, len(target_probs)):
            start_idx = max(0, i - win_size + 1)
            end_idx = i + 1
            temp[i] = np.mean(target_probs[start_idx:end_idx])
        return temp
    
    if len(probs[pre_mask]):
        smoothed_probs[:len(probs[pre_mask])] = moving_avg(probs[pre_mask],window_size)
    if len(probs[inter_mask]):
        smoothed_probs[len(probs[pre_mask]):] = moving_avg(probs[inter_mask],window_size)
    
    return smoothed_probs

def weighted_ensemble(probs_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine predictions using weighted average"""
    return np.tensordot(weights, probs_stack, axes=1)


def apply_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply threshold to get binary predictions"""
    return (probs >= threshold).astype(int)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive metrics for predictions"""
    
    @staticmethod
    def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict:
        """Compute basic classification metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['auc'] = np.nan
        
        try:
            metrics['ap'] = average_precision_score(y_true, y_probs)
        except:
            metrics['ap'] = np.nan
        
        return metrics
    
    @staticmethod
    def compute_seizure_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute seizure-specific metrics"""
        metrics = {}
        
        # Sensitivity: at least one preictal detected
        has_preictal = np.any(y_true == 1)
        if has_preictal:
            detected = np.any((y_true == 1) & (y_pred == 1))
            metrics['sensitivity'] = 1 if detected else 0
        else:
            metrics['sensitivity'] = np.nan
        
        # FPR per hour
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        hours = (len(y_true) * 5) / 3600.0  # 5 seconds per sample
        metrics['fpr_per_hour'] = false_positives / hours if hours > 0 else np.nan
        
        # --- Suppressed prediction version (for FPR_2) ---
        suppression_len = 12 * 5  # 60 samples (5 minutes)
        y_pred_suppressed = y_pred.copy()

        # Process each contiguous region of identical y_true values separately
        regions = []
        start = 0
        for i in range(1, len(y_true)):
            if y_true[i] != y_true[i - 1]:
                regions.append((start, i))
                start = i
        regions.append((start, len(y_true)))  # last region

        for start, end in regions:
            preds = y_pred_suppressed[start:end]
            i = 0
            while i < len(preds):
                if preds[i] == 1:
                    preds[i + 1 : i + 1 + suppression_len] = 0
                    i += suppression_len  # jump ahead to skip suppressed region
                i += 1
            y_pred_suppressed[start:end] = preds

        # --- Compute suppressed FPR per hour ---
        false_positives_supp = np.sum((y_true == 0) & (y_pred_suppressed == 1))
        metrics['fpr_sup'] = false_positives_supp / hours if hours > 0 else np.nan

        # Time to first detection (in samples)
        if has_preictal:
            first_preictal_idx = np.where(y_true == 1)[0][0]
            detection_mask = (y_true == 1) & (y_pred == 1)
            if np.any(detection_mask):
                first_detection_idx = np.where(detection_mask)[0][0]
                metrics['time_to_detection'] = (first_detection_idx - first_preictal_idx) * 5  # seconds
            else:
                metrics['time_to_detection'] = np.nan
        else:
            metrics['time_to_detection'] = np.nan
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        return metrics
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict:
        """Compute all available metrics"""
        basic = MetricsCalculator.compute_basic_metrics(y_true, y_pred, y_probs)
        seizure = MetricsCalculator.compute_seizure_specific_metrics(y_true, y_pred)
        return {**basic, **seizure}


# ============================================================================
# RESULTS PROCESSOR
# ============================================================================

class ResultsProcessor:
    """Process raw predictions and generate all variants"""
    
    def __init__(self, raw_results: Dict, run_dir: Path):
        self.raw_results = raw_results
        self.run_dir = run_dir
        self.processed_results = []
    
    def process_all_variants(
        self,
        calibration_methods: List[str] = ['none', 'percentile', 'beta', 'isotonic', 'temperature'],
        ma_windows: List[int] = [1, 3, 5, 7],
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
        percentiles: List[int] = [5, 10, 15, 20]
    ):
        """Process all combinations of variants"""
        
        for outer_fold in self.raw_results['outer_folds']:
            fold_results = self._process_outer_fold(
                outer_fold,
                calibration_methods,
                ma_windows,
                thresholds,
                percentiles
            )
            self.processed_results.append(fold_results)
        
        return self.processed_results
    
    def _process_outer_fold(
        self,
        outer_fold: Dict,
        calibration_methods: List[str],
        ma_windows: List[int],
        thresholds: List[float],
        percentiles: List[int]
    ) -> Dict:
        """Process one outer fold with all variants"""
        
        fold_num = outer_fold['outer_fold']
        y_test = np.array(outer_fold['y_test'])
        
        # Extract test predictions and validation data from inner folds
        test_probs_stack = []
        val_probs_list = []
        val_labels_list = []
        val_aucs = []
        
        for inner_fold in outer_fold['inner_folds']:
            test_probs_stack.append(np.array(inner_fold['test_probs']))
            val_probs_list.append(np.array(inner_fold['val_probs']))
            val_labels_list.append(np.array(inner_fold['val_labels']))
            val_aucs.append(inner_fold['best_val_auc'])
        
        test_probs_stack = np.stack(test_probs_stack)
        val_aucs = np.array(val_aucs)
        weights = val_aucs / val_aucs.sum()
        
        # Base ensemble (no calibration)
        base_probs = weighted_ensemble(test_probs_stack, weights)
        
        fold_results = {
            'fold': fold_num,
            'y_test': y_test,
            'variants': {}
        }
        
        # Process each calibration method
        for cal_method in calibration_methods:
            if cal_method == 'none':
                # No calibration
                for ma_window in ma_windows:
                    probs = moving_average(base_probs, y_test, ma_window) if ma_window > 1 else base_probs
                    
                    for threshold in thresholds:
                        preds = apply_threshold(probs, threshold)
                        metrics = MetricsCalculator.compute_all_metrics(y_test, preds, probs)
                        
                        variant_name = f"cal_{cal_method}_ma_{ma_window}_thr_{threshold:.2f}"
                        fold_results['variants'][variant_name] = {
                            'probs': probs,
                            'preds': preds,
                            'metrics': metrics,
                            'config': {
                                'calibration': cal_method,
                                'ma_window': ma_window,
                                'threshold': threshold
                            }
                        }
            
            elif cal_method == 'percentile':
                # Percentile calibration with different percentiles
                for percentile in percentiles:
                    cal_probs, _ = calibrate_ensemble(
                        test_probs_stack,
                        val_probs_list,
                        val_labels_list,
                        val_aucs,
                        calibration_method='percentile',
                        target_percentile=percentile
                    )
                    
                    for ma_window in ma_windows:
                        probs = moving_average(cal_probs, y_test, ma_window) if ma_window > 1 else cal_probs
                        
                        for threshold in thresholds:
                            preds = apply_threshold(probs, threshold)
                            metrics = MetricsCalculator.compute_all_metrics(y_test, preds, probs)
                            
                            variant_name = f"cal_{cal_method}_p{percentile}_ma_{ma_window}_thr_{threshold:.2f}"
                            fold_results['variants'][variant_name] = {
                                'probs': probs,
                                'preds': preds,
                                'metrics': metrics,
                                'config': {
                                    'calibration': cal_method,
                                    'percentile': percentile,
                                    'ma_window': ma_window,
                                    'threshold': threshold
                                }
                            }
            
            else:
                # Other calibration methods (beta, isotonic, temperature)
                cal_probs, _ = calibrate_ensemble(
                    test_probs_stack,
                    val_probs_list,
                    val_labels_list,
                    val_aucs,
                    calibration_method=cal_method
                )
                
                for ma_window in ma_windows:
                    probs = moving_average(cal_probs, y_test, ma_window) if ma_window > 1 else cal_probs
                    
                    for threshold in thresholds:
                        preds = apply_threshold(probs, threshold)
                        metrics = MetricsCalculator.compute_all_metrics(y_test, preds, probs)
                        
                        variant_name = f"cal_{cal_method}_ma_{ma_window}_thr_{threshold:.2f}"
                        fold_results['variants'][variant_name] = {
                            'probs': probs,
                            'preds': preds,
                            'metrics': metrics,
                            'config': {
                                'calibration': cal_method,
                                'ma_window': ma_window,
                                'threshold': threshold
                            }
                        }
        
        return fold_results


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Generate all visualizations"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.viz_dir = run_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def plot_roc_curves(self, processed_results: List[Dict], variant_name: str):
        """Plot ROC curves for a specific variant across all folds"""
        plt.figure(figsize=(10, 8))
        
        for fold_result in processed_results:
            if variant_name not in fold_result['variants']:
                continue
            
            variant = fold_result['variants'][variant_name]
            y_test = fold_result['y_test']
            probs = variant['probs']
            
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, alpha=0.7, 
                    label=f"Fold {fold_result['fold']} (AUC={roc_auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {variant_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'roc_{variant_name}.png', bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, processed_results: List[Dict], variant_name: str):
        """Plot precision-recall curves"""
        plt.figure(figsize=(10, 8))
        
        for fold_result in processed_results:
            if variant_name not in fold_result['variants']:
                continue
            
            variant = fold_result['variants'][variant_name]
            y_test = fold_result['y_test']
            probs = variant['probs']
            
            precision, recall, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            
            plt.plot(recall, precision, alpha=0.7,
                    label=f"Fold {fold_result['fold']} (AP={ap:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: {variant_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'pr_{variant_name}.png', bbox_inches='tight')
        plt.close()
    
    def plot_probability_distributions(self, processed_results: List[Dict], variant_name: str):
        """Plot probability distributions for preictal vs interictal"""
        n_folds = len(processed_results)
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 5))
        if n_folds == 1:
            axes = [axes]
        
        for idx, fold_result in enumerate(processed_results):
            if variant_name not in fold_result['variants']:
                continue
            
            variant = fold_result['variants'][variant_name]
            y_test = fold_result['y_test']
            probs = variant['probs']
            
            axes[idx].hist(probs[y_test == 0], bins=30, alpha=0.7, 
                          label='Interictal', density=True, color='green')
            axes[idx].hist(probs[y_test == 1], bins=30, alpha=0.7,
                          label='Preictal', density=True, color='red')
            axes[idx].axvline(0.5, color='black', linestyle='--', alpha=0.7)
            axes[idx].set_xlabel('Probability')
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(f'Fold {fold_result["fold"]}')
            axes[idx].legend()
        
        plt.suptitle(f'Probability Distributions: {variant_name}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'prob_dist_{variant_name}.png', bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, processed_results: List[Dict], variant_name: str):
        """Plot confusion matrices for all folds"""
        n_folds = len(processed_results)
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 5))
        if n_folds == 1:
            axes = [axes]
        
        for idx, fold_result in enumerate(processed_results):
            if variant_name not in fold_result['variants']:
                continue
            
            variant = fold_result['variants'][variant_name]
            y_test = fold_result['y_test']
            preds = variant['preds']
            
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_title(f'Fold {fold_result["fold"]}')
        
        plt.suptitle(f'Confusion Matrices: {variant_name}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'cm_{variant_name}.png', bbox_inches='tight')
        plt.close()
    
    def plot_metric_comparison(self, summary_df: pd.DataFrame, metric: str):
        """Plot comparison of a specific metric across variants"""
        # Select top 20 variants by metric
        top_variants = summary_df.nlargest(20, f'mean_{metric}')
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_variants)), top_variants[f'mean_{metric}'])
        plt.yticks(range(len(top_variants)), top_variants['variant'], fontsize=8)
        plt.xlabel(f'Mean {metric.upper()}')
        plt.title(f'Top 20 Variants by {metric.upper()}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'comparison_{metric}.png', bbox_inches='tight')
        plt.close()
    
    def plot_threshold_sensitivity_analysis(self, processed_results: List[Dict], cal_method: str, ma_window: int):
        """Analyze how metrics change with threshold"""
        thresholds = np.linspace(0, 1, 21)
        
        all_aucs = []
        all_sens = []
        all_fprs = []
        
        for threshold in thresholds:
            variant_name = f"cal_{cal_method}_ma_{ma_window}_thr_{threshold:.2f}"
            
            fold_aucs = []
            fold_sens = []
            fold_fprs = []
            
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    metrics = fold_result['variants'][variant_name]['metrics']
                    fold_aucs.append(metrics['auc'])
                    if not np.isnan(metrics['sensitivity']):
                        fold_sens.append(metrics['sensitivity'])
                    fold_fprs.append(metrics['fpr_per_hour'])
            
            all_aucs.append(np.mean(fold_aucs) if fold_aucs else np.nan)
            all_sens.append(np.mean(fold_sens) if fold_sens else np.nan)
            all_fprs.append(np.mean(fold_fprs) if fold_fprs else np.nan)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(thresholds, all_aucs, 'o-')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('AUC')
        axes[0].set_title('AUC vs Threshold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(thresholds, all_sens, 'o-')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Sensitivity')
        axes[1].set_title('Sensitivity vs Threshold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(thresholds, all_fprs, 'o-')
        axes[2].set_xlabel('Threshold')
        axes[2].set_ylabel('FPR/hour')
        axes[2].set_title('FPR/hour vs Threshold')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Analysis: cal={cal_method}, ma={ma_window}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'threshold_analysis_{cal_method}_ma{ma_window}.png', bbox_inches='tight')
        plt.close()
    
    def plot_ma_window_comparison(self, processed_results: List[Dict], cal_method: str, threshold: float):
        """Compare performance across different MA windows"""
        windows = [1, 3, 5, 7, 10]
        
        metrics_by_window = {w: {'auc': [], 'sens': [], 'fpr': []} for w in windows}
        
        for window in windows:
            if cal_method == 'percentile':
                # Use a default percentile for comparison
                variant_name = f"cal_{cal_method}_p10_ma_{window}_thr_{threshold:.2f}"
            else:
                variant_name = f"cal_{cal_method}_ma_{window}_thr_{threshold:.2f}"
            
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    metrics = fold_result['variants'][variant_name]['metrics']
                    metrics_by_window[window]['auc'].append(metrics['auc'])
                    if not np.isnan(metrics['sensitivity']):
                        metrics_by_window[window]['sens'].append(metrics['sensitivity'])
                    metrics_by_window[window]['fpr'].append(metrics['fpr_per_hour'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        mean_aucs = [np.mean(metrics_by_window[w]['auc']) for w in windows]
        std_aucs = [np.std(metrics_by_window[w]['auc']) for w in windows]
        axes[0].errorbar(windows, mean_aucs, yerr=std_aucs, marker='o', capsize=5)
        axes[0].set_xlabel('MA Window Size')
        axes[0].set_ylabel('AUC')
        axes[0].set_title('AUC vs MA Window')
        axes[0].grid(True, alpha=0.3)
        
        mean_sens = [np.mean(metrics_by_window[w]['sens']) if metrics_by_window[w]['sens'] else 0 for w in windows]
        std_sens = [np.std(metrics_by_window[w]['sens']) if metrics_by_window[w]['sens'] else 0 for w in windows]
        axes[1].errorbar(windows, mean_sens, yerr=std_sens, marker='o', capsize=5)
        axes[1].set_xlabel('MA Window Size')
        axes[1].set_ylabel('Sensitivity')
        axes[1].set_title('Sensitivity vs MA Window')
        axes[1].grid(True, alpha=0.3)
        
        mean_fprs = [np.mean(metrics_by_window[w]['fpr']) for w in windows]
        std_fprs = [np.std(metrics_by_window[w]['fpr']) for w in windows]
        axes[2].errorbar(windows, mean_fprs, yerr=std_fprs, marker='o', capsize=5)
        axes[2].set_xlabel('MA Window Size')
        axes[2].set_ylabel('FPR/hour')
        axes[2].set_title('FPR/hour vs MA Window')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'MA Window Comparison: cal={cal_method}, threshold={threshold}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'ma_comparison_{cal_method}_thr{threshold:.2f}.png', bbox_inches='tight')
        plt.close()
    
    def plot_calibration_method_comparison(self, processed_results: List[Dict], ma_window: int, threshold: float):
        """Compare different calibration methods"""
        cal_methods = ['none', 'percentile', 'beta', 'isotonic', 'temperature']
        
        metrics_data = []
        
        for cal_method in cal_methods:
            if cal_method == 'percentile':
                variant_name = f"cal_{cal_method}_p10_ma_{ma_window}_thr_{threshold:.2f}"
            else:
                variant_name = f"cal_{cal_method}_ma_{ma_window}_thr_{threshold:.2f}"
            
            fold_metrics = []
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    metrics = fold_result['variants'][variant_name]['metrics']
                    fold_metrics.append({
                        'method': cal_method,
                        'auc': metrics['auc'],
                        'sensitivity': metrics['sensitivity'],
                        'fpr_per_hour': metrics['fpr_per_hour']
                    })
            
            if fold_metrics:
                metrics_data.extend(fold_metrics)
        
        if not metrics_data:
            return
        
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # AUC comparison
        sns.boxplot(data=df, x='method', y='auc', ax=axes[0])
        axes[0].set_xlabel('Calibration Method')
        axes[0].set_ylabel('AUC')
        axes[0].set_title('AUC by Calibration Method')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Sensitivity comparison
        sns.boxplot(data=df, x='method', y='sensitivity', ax=axes[1])
        axes[1].set_xlabel('Calibration Method')
        axes[1].set_ylabel('Sensitivity')
        axes[1].set_title('Sensitivity by Calibration Method')
        axes[1].tick_params(axis='x', rotation=45)
        
        # FPR comparison
        sns.boxplot(data=df, x='method', y='fpr_per_hour', ax=axes[2])
        axes[2].set_xlabel('Calibration Method')
        axes[2].set_ylabel('FPR/hour')
        axes[2].set_title('FPR/hour by Calibration Method')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Calibration Method Comparison: ma={ma_window}, threshold={threshold}')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'calibration_comparison_ma{ma_window}_thr{threshold:.2f}.png', bbox_inches='tight')
        plt.close()
    
    def plot_pareto_frontier(self, summary_df: pd.DataFrame):
        """Plot Pareto frontier for sensitivity vs FPR tradeoff"""
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        valid_data = summary_df[
            ~summary_df['mean_sensitivity'].isna() & 
            ~summary_df['mean_fpr_per_hour'].isna()
        ].copy()
        
        if len(valid_data) == 0:
            return
        
        # Plot all points
        plt.scatter(valid_data['mean_fpr_per_hour'], 
                   valid_data['mean_sensitivity'],
                   c=valid_data['mean_auc'],
                   cmap='viridis',
                   s=50,
                   alpha=0.6)
        
        # Find Pareto frontier (maximize sensitivity, minimize FPR)
        pareto_points = []
        for idx, row in valid_data.iterrows():
            is_pareto = True
            for _, other_row in valid_data.iterrows():
                if (other_row['mean_sensitivity'] >= row['mean_sensitivity'] and
                    other_row['mean_fpr_per_hour'] <= row['mean_fpr_per_hour'] and
                    (other_row['mean_sensitivity'] > row['mean_sensitivity'] or
                     other_row['mean_fpr_per_hour'] < row['mean_fpr_per_hour'])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(idx)
        
        # Highlight Pareto frontier
        pareto_df = valid_data.loc[pareto_points].sort_values('mean_fpr_per_hour')
        plt.plot(pareto_df['mean_fpr_per_hour'], 
                pareto_df['mean_sensitivity'],
                'r-', linewidth=2, label='Pareto Frontier')
        
        plt.colorbar(label='Mean AUC')
        plt.xlabel('Mean FPR per Hour')
        plt.ylabel('Mean Sensitivity')
        plt.title('Sensitivity vs FPR Tradeoff (Pareto Frontier)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pareto_frontier.png', bbox_inches='tight')
        plt.close()
        
    def plot_pareto_frontier(self, summary_df: pd.DataFrame):
        """Plot Pareto frontier for sensitivity vs FPR tradeoff"""
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN values
        valid_data = summary_df[
            ~summary_df['mean_sensitivity'].isna() & 
            ~summary_df['mean_fpr_per_hour'].isna()
        ].copy()
        
        if len(valid_data) == 0:
            return
        
        # Plot all points
        plt.scatter(valid_data['mean_fpr_per_hour'], 
                   valid_data['mean_sensitivity'],
                   c=valid_data['mean_auc'],
                   cmap='viridis',
                   s=50,
                   alpha=0.6)
        
        # Find Pareto frontier (maximize sensitivity, minimize FPR)
        pareto_points = []
        for idx, row in valid_data.iterrows():
            is_pareto = True
            for _, other_row in valid_data.iterrows():
                if (other_row['mean_sensitivity'] >= row['mean_sensitivity'] and
                    other_row['mean_fpr_per_hour'] <= row['mean_fpr_per_hour'] and
                    (other_row['mean_sensitivity'] > row['mean_sensitivity'] or
                     other_row['mean_fpr_per_hour'] < row['mean_fpr_per_hour'])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(idx)
        
        # Highlight Pareto frontier
        pareto_df = valid_data.loc[pareto_points].sort_values('mean_fpr_per_hour')
        plt.plot(pareto_df['mean_fpr_per_hour'], 
                pareto_df['mean_sensitivity'],
                'r-', linewidth=2, label='Pareto Frontier')
        
        plt.colorbar(label='Mean AUC')
        plt.xlabel('Mean FPR per Hour')
        plt.ylabel('Mean Sensitivity')
        plt.title('Sensitivity vs FPR Tradeoff (Pareto Frontier)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pareto_frontier.png', bbox_inches='tight')
        plt.close()
        
        # Save Pareto optimal variants
        pareto_df.to_csv(self.run_dir / 'pareto_optimal_variants.csv', index=False)
    
    def plot_ensemble_correlation(self, processed_results: List[Dict]):
        """Plot correlation between ensemble models for each fold"""
        for fold_result in processed_results:
            fold_num = fold_result['fold']
            y_test = fold_result['y_test']
            
            # Get test predictions from all inner folds
            # We need to reconstruct this from raw data
            # This will be populated when we have access to raw results
            pass
    
    def plot_training_curves_summary(self, raw_results: Dict):
        """Generate training curve summaries from raw results"""
        out_dir = self.viz_dir / 'training_curves'
        out_dir.mkdir(exist_ok=True)
        
        for outer_fold in raw_results['outer_folds']:
            fold_num = outer_fold['outer_fold']
            
            # Plot training curves for each inner fold
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for inner_idx, inner_fold in enumerate(outer_fold['inner_folds']):
                if inner_idx >= 6:  # Max 6 subplots
                    break
                
                training_log = inner_fold['training_log']
                
                epochs = [log['epoch'] for log in training_log]
                val_aucs = [log['val_auc'] for log in training_log]
                val_losses = [log['val_loss'] for log in training_log]
                val_accs = [log['val_acc'] for log in training_log]
                
                ax = axes[inner_idx]
                ax2 = ax.twinx()
                
                l1 = ax.plot(epochs, val_aucs, 'b-', label='Val AUC')
                l2 = ax.plot(epochs, val_accs, 'g-', label='Val Acc')
                l3 = ax2.plot(epochs, val_losses, 'r-', label='Val Loss')
                
                # Mark best epoch
                best_auc_epoch = np.argmax(val_aucs) + 1
                ax.axvline(best_auc_epoch, color='k', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('AUC / Accuracy')
                ax2.set_ylabel('Loss')
                ax.set_title(f'Inner Fold {inner_fold["inner_fold"]} (Best: {best_auc_epoch})')
                ax.grid(True, alpha=0.3)
                
                # Combined legend
                lines = l1 + l2 + l3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='best')
            
            # Hide unused subplots
            for idx in range(len(outer_fold['inner_folds']), 6):
                axes[idx].axis('off')
            
            plt.suptitle(f'Training Curves - Outer Fold {fold_num}')
            plt.tight_layout()
            plt.savefig(out_dir / f'training_curves_fold_{fold_num}.png', bbox_inches='tight')
            plt.close()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

class SummaryGenerator:
    """Generate summary statistics and tables"""
    
    @staticmethod
    def create_variant_summary(processed_results: List[Dict]) -> pd.DataFrame:
        """Create summary table for all variants"""
        summary_data = []
        
        # Get all variant names
        all_variants = set()
        for fold_result in processed_results:
            all_variants.update(fold_result['variants'].keys())
        
        for variant_name in sorted(all_variants):
            variant_metrics = []
            
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    metrics = fold_result['variants'][variant_name]['metrics']
                    variant_metrics.append(metrics)
            
            if not variant_metrics:
                continue
            
            # Aggregate metrics
            summary = {'variant': variant_name}
            
            # Extract config
            config = processed_results[0]['variants'][variant_name]['config']
            summary.update({f'config_{k}': v for k, v in config.items()})
            
            # Compute mean and std for each metric
            for metric in variant_metrics[0].keys():
                values = [m[metric] for m in variant_metrics if not np.isnan(m[metric])]
                if values:
                    summary[f'mean_{metric}'] = np.mean(values)
                    summary[f'std_{metric}'] = np.std(values)
                else:
                    summary[f'mean_{metric}'] = np.nan
                    summary[f'std_{metric}'] = np.nan
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def create_best_variants_table(summary_df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Create tables of best variants for each metric"""
        metrics = ['auc', 'sensitivity', 'f1', 'precision', 'recall']
        best_tables = {}
        
        for metric in metrics:
            mean_col = f'mean_{metric}'
            if mean_col in summary_df.columns:
                best_variants = summary_df.nlargest(top_n, mean_col)
                best_tables[metric] = best_variants[
                    ['variant', mean_col, f'std_{metric}', 'mean_fpr_per_hour']
                ].copy()
        
        return best_tables
    
    @staticmethod
    def create_calibration_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compare calibration methods"""
        # Group by calibration method
        cal_methods = summary_df['config_calibration'].unique()
        
        comparison_data = []
        for cal_method in cal_methods:
            method_data = summary_df[summary_df['config_calibration'] == cal_method]
            
            comparison_data.append({
                'calibration_method': cal_method,
                'mean_auc': method_data['mean_auc'].mean(),
                'mean_sensitivity': method_data['mean_sensitivity'].mean(),
                'mean_fpr_per_hour': method_data['mean_fpr_per_hour'].mean(),
                'mean_f1': method_data['mean_f1'].mean(),
                'n_variants': len(method_data)
            })
        
        return pd.DataFrame(comparison_data).sort_values('mean_auc', ascending=False)
    
    @staticmethod
    def create_ma_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compare moving average windows"""
        # Group by MA window
        ma_windows = summary_df['config_ma_window'].unique()
        
        comparison_data = []
        for ma_window in sorted(ma_windows):
            window_data = summary_df[summary_df['config_ma_window'] == ma_window]
            
            comparison_data.append({
                'ma_window': ma_window,
                'mean_auc': window_data['mean_auc'].mean(),
                'mean_sensitivity': window_data['mean_sensitivity'].mean(),
                'mean_fpr_per_hour': window_data['mean_fpr_per_hour'].mean(),
                'mean_f1': window_data['mean_f1'].mean(),
                'n_variants': len(window_data)
            })
        
        return pd.DataFrame(comparison_data).sort_values('ma_window')
    
    @staticmethod
    def create_threshold_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compare thresholds"""
        # Group by threshold
        thresholds = summary_df['config_threshold'].unique()
        
        comparison_data = []
        for threshold in sorted(thresholds):
            threshold_data = summary_df[summary_df['config_threshold'] == threshold]
            
            comparison_data.append({
                'threshold': threshold,
                'mean_auc': threshold_data['mean_auc'].mean(),
                'mean_sensitivity': threshold_data['mean_sensitivity'].mean(),
                'mean_fpr_per_hour': threshold_data['mean_fpr_per_hour'].mean(),
                'mean_f1': threshold_data['mean_f1'].mean(),
                'n_variants': len(threshold_data)
            })
        
        return pd.DataFrame(comparison_data).sort_values('threshold')


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_run(run_dir: str, 
                calibration_methods: List[str] = None,
                ma_windows: List[int] = None,
                thresholds: List[float] = None,
                percentiles: List[int] = None):
    """Main analysis function"""
    
    run_path = Path(run_dir)
    
    # Load raw predictions
    raw_predictions_path = run_path / 'raw_predictions.pkl'
    if not raw_predictions_path.exists():
        print(f"Error: {raw_predictions_path} not found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Analyzing run: {run_dir}")
    print(f"{'='*80}\n")
    
    with open(raw_predictions_path, 'rb') as f:
        raw_results = pickle.load(f)
    
    # Load config
    config_path = run_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration:")
        for key, value in config['arguments'].items():
            print(f"  {key}: {value}")
        print()
    
    # Set defaults
    if calibration_methods is None:
        calibration_methods = ['none', 'percentile', 'beta', 'isotonic', 'temperature']
    if ma_windows is None:
        ma_windows = [1, 3, 5, 7, 10]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    if percentiles is None:
        percentiles = [5, 10, 15, 20]
    
    # Process all variants
    print("Processing all variants...")
    processor = ResultsProcessor(raw_results, run_path)
    processed_results = processor.process_all_variants(
        calibration_methods=calibration_methods,
        ma_windows=ma_windows,
        thresholds=thresholds,
        percentiles=percentiles
    )
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_df = SummaryGenerator.create_variant_summary(processed_results)
    summary_df.to_csv(run_path / 'variant_summary.csv', index=False)
    print(f"  Saved variant_summary.csv ({len(summary_df)} variants)")
    
    # Best variants tables
    best_tables = SummaryGenerator.create_best_variants_table(summary_df, top_n=10)
    for metric, table in best_tables.items():
        table.to_csv(run_path / f'best_variants_{metric}.csv', index=False)
        print(f"  Saved best_variants_{metric}.csv")
    
    # Comparison tables
    cal_comparison = SummaryGenerator.create_calibration_comparison_table(summary_df)
    cal_comparison.to_csv(run_path / 'calibration_comparison.csv', index=False)
    print(f"  Saved calibration_comparison.csv")
    
    ma_comparison = SummaryGenerator.create_ma_comparison_table(summary_df)
    ma_comparison.to_csv(run_path / 'ma_window_comparison.csv', index=False)
    print(f"  Saved ma_window_comparison.csv")
    
    threshold_comparison = SummaryGenerator.create_threshold_comparison_table(summary_df)
    threshold_comparison.to_csv(run_path / 'threshold_comparison.csv', index=False)
    print(f"  Saved threshold_comparison.csv")
    
    # Display top results
    print("\n" + "="*80)
    print("TOP 10 VARIANTS BY AUC:")
    print("="*80)
    top_auc = summary_df.nlargest(10, 'mean_auc')[
        ['variant', 'mean_auc', 'std_auc', 'mean_sensitivity', 'mean_fpr_per_hour']
    ]
    print(top_auc.to_string(index=False))

    print("\n" + "="*80)
    print("Top 10 Sensitivity and FPR:")
    print("="*80)
    top_sen_fpr = summary_df.sort_values(['mean_sensitivity', 'mean_fpr_per_hour'], ascending=[False, True])[
        ['variant', 'mean_auc', 'std_auc', 'mean_sensitivity', 'mean_fpr_per_hour', 'mean_fpr_sup']
    ]
    print(top_sen_fpr[0:10].to_string(index=False))
    
    print("\n" + "="*80)
    print("CALIBRATION METHOD COMPARISON:")
    print("="*80)
    print(cal_comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("MOVING AVERAGE WINDOW COMPARISON:")
    print("="*80)
    print(ma_comparison.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = Visualizer(run_path)
    
    # Get best variant for detailed plots
    best_variant = summary_df.nlargest(1, 'mean_auc').iloc[0]['variant']
    print(f"  Best variant: {best_variant}")
    
    # ROC curves for best variant
    visualizer.plot_roc_curves(processed_results, best_variant)
    print(f"  Generated ROC curves")
    
    # Precision-recall curves
    visualizer.plot_precision_recall_curves(processed_results, best_variant)
    print(f"  Generated PR curves")
    
    # Probability distributions
    visualizer.plot_probability_distributions(processed_results, best_variant)
    print(f"  Generated probability distributions")
    
    # Confusion matrices
    visualizer.plot_confusion_matrices(processed_results, best_variant)
    print(f"  Generated confusion matrices")
    
    # Metric comparisons
    for metric in ['auc', 'sensitivity', 'f1']:
        visualizer.plot_metric_comparison(summary_df, metric)
    print(f"  Generated metric comparisons")
    
    # Threshold analysis for each calibration method
    for cal_method in ['none', 'percentile', 'beta']:
        visualizer.plot_threshold_sensitivity_analysis(processed_results, cal_method, ma_window=5)
    print(f"  Generated threshold analyses")
    
    # MA window comparison
    for cal_method in ['none', 'percentile']:
        visualizer.plot_ma_window_comparison(processed_results, cal_method, threshold=0.5)
    print(f"  Generated MA window comparisons")
    
    # Calibration method comparison
    visualizer.plot_calibration_method_comparison(processed_results, ma_window=5, threshold=0.5)
    print(f"  Generated calibration method comparison")
    
    # Pareto frontier
    visualizer.plot_pareto_frontier(summary_df)
    print(f"  Generated Pareto frontier")
    
    # Training curves summary from raw data
    visualizer.plot_training_curves_summary(raw_results)
    print(f"  Generated training curves summary")
    
    # Ensemble correlation analysis
    print("\nGenerating ensemble correlation analysis...")
    for outer_fold in raw_results['outer_folds']:
        fold_num = outer_fold['outer_fold']
        
        # Extract test predictions from all inner folds
        test_probs_stack = []
        for inner_fold in outer_fold['inner_folds']:
            test_probs_stack.append(np.array(inner_fold['test_probs']))
        
        test_probs_stack = np.stack(test_probs_stack)
        y_test = np.array(outer_fold['y_test'])
        
        # Compute and plot correlation
        save_path = run_path / 'visualizations' / f'ensemble_correlation_fold_{fold_num}.png'
        compute_prediction_correlation(
            test_probs_stack,
            y_test,
            save_path=str(save_path)
        )
    print(f"  Generated ensemble correlation plots")
    
    # Generate detailed per-fold visualizations using TrainingVisualizer
    print("\nGenerating detailed per-fold visualizations...")
    generate_training_visualizations(raw_results, run_path)
    
    print(f"\nAnalysis complete! Results saved to {run_dir}")
    print(f"Visualizations saved to {run_path / 'visualizations'}")


def generate_training_visualizations(raw_results: Dict, run_path: Path):
    """Generate detailed training visualizations for all folds using TrainingVisualizer"""
    
    for outer_fold in raw_results['outer_folds']:
        fold_num = outer_fold['outer_fold']
        
        for inner_fold in outer_fold['inner_folds']:
            inner_num = inner_fold['inner_fold']
            
            # Create visualizer
            vis = TrainingVisualizer(
                run_dir=str(run_path),
                metric_for_best='auc',
                only_curves=False,
                only_best=True
            )
            vis.reset()
            
            # Populate history from training log
            training_log = inner_fold['training_log']
            val_probs = np.array(inner_fold['val_probs'])
            val_labels = np.array(inner_fold['val_labels'])
            
            for epoch_data in training_log:
                epoch = epoch_data['epoch']
                
                # For training visualizer, we need predictions
                # We only have the final best model's predictions, so we'll use those
                # This is a limitation - ideally we'd save predictions per epoch during training
                val_preds = (val_probs >= 0.5).astype(int)
                
                vis.update(
                    epoch=epoch,
                    tr_loss=epoch_data['train_loss'],
                    tr_acc=epoch_data['train_acc'],
                    val_loss=epoch_data['val_loss'],
                    val_acc=epoch_data['val_acc'],
                    vprobs=val_probs,
                    vpreds=val_preds,
                    vlabels=val_labels
                )
            
            # Render final visualizations
            vis.render(
                fold=fold_num,
                inner_fold=inner_num,
                vprobs=val_probs,
                vpreds=val_preds,
                vlabels=val_labels
            )
    
    print(f"  Generated detailed per-fold visualizations in folds/ directory")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of seizure prediction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze most recent run
  python analyze_results2.py
  
  # Analyze specific run
  python analyze_results2.py --run_dir runs/run1_20240101_120000
  
  # Customize analysis parameters
  python analyze_results2.py --run_dir runs/run1 --calibration_methods none percentile --ma_windows 1 3 5
        """
    )
    
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Path to run directory (default: most recent)')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Path to runs directory')
    parser.add_argument('--calibration_methods', nargs='+', 
                       choices=['none', 'percentile', 'beta', 'isotonic', 'temperature'],
                       help='Calibration methods to analyze')
    parser.add_argument('--ma_windows', type=int, nargs='+',
                       help='Moving average windows to analyze')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       help='Thresholds to analyze')
    parser.add_argument('--percentiles', type=int, nargs='+',
                       help='Percentiles for percentile calibration')
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        # Find most recent run
        runs_path = Path(args.runs_dir)
        if not runs_path.exists():
            print(f"Error: Runs directory {runs_path} does not exist")
            return
        
        run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run')]
        if not run_dirs:
            print(f"Error: No run directories found in {runs_path}")
            return
        
        run_dir = str(max(run_dirs, key=lambda x: x.stat().st_mtime))
        print(f"Analyzing most recent run: {Path(run_dir).name}")
    
    # Run analysis
    analyze_run(
        run_dir,
        calibration_methods=args.calibration_methods,
        ma_windows=args.ma_windows,
        thresholds=args.thresholds,
        percentiles=args.percentiles
    )


if __name__ == "__main__":
    main()