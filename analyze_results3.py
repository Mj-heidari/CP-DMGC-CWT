"""
Simplified Analysis Script for Seizure Prediction Results

This script performs post-processing, metrics computation, and visualization
on raw predictions from train.py. It focuses on:
- Multiple moving average windows
- Multiple thresholds
- Key metrics: AUC, FPR, and Sensitivity
- Essential visualizations only

Removed features:
- Calibration methods
- Training curves
- Folds folder visualizations
- Non-essential plots
- Extra metrics (only keeping AUC, FPR, Sensitivity)
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def moving_average(probs: np.ndarray, y: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply moving average smoothing within contiguous regions of equal labels.
    
    Args:
        probs: Predicted probabilities in time order
        y: True labels in time order
        window_size: Window size for moving average
    
    Returns:
        Smoothed probabilities in original time order
    """
    if len(probs) < window_size or window_size == 1:
        return probs.copy()
    
    y = np.array(y)
    smoothed_probs = probs.copy().astype(float)
    
    # Find contiguous regions of identical labels
    regions = []
    start = 0
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            regions.append((start, i))
            start = i
    regions.append((start, len(y)))  # Last region
    
    # Apply moving average within each region
    for start, end in regions:
        region_len = end - start
        if region_len >= window_size:
            for i in range(start, end):
                win_start = max(start, i - window_size + 1)
                win_end = i + 1
                smoothed_probs[i] = np.mean(probs[win_start:win_end])
        else:
            for i in range(start, end):
                win_start = start
                win_end = i + 1
                smoothed_probs[i] = np.mean(probs[win_start:win_end])
    
    return smoothed_probs


def weighted_ensemble(probs_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine predictions using weighted average"""
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / np.sum(weights)
    return np.tensordot(weights, probs_stack, axes=1)


def apply_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply threshold to get binary predictions"""
    return (probs >= threshold).astype(int)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

class MetricsCalculator:
    """Calculate key metrics: AUC, FPR, Sensitivity"""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_probs: np.ndarray,
        sampling_period: float = 5.0
    ) -> Dict:
        """
        Compute AUC, Sensitivity, and FPR per hour
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities
            sampling_period: Time per sample in seconds (default: 5.0)
        """
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['auc'] = np.nan
        
        # Sensitivity: at least one preictal detected
        has_preictal = np.any(y_true == 1)
        if has_preictal:
            detected = np.any((y_true == 1) & (y_pred == 1))
            metrics['sensitivity'] = 1.0 if detected else 0.0
        else:
            metrics['sensitivity'] = np.nan
        
        # FPR per hour - computed over interictal time only
        interictal_mask = (y_true == 0)
        false_positives = np.sum(interictal_mask & (y_pred == 1))
        interictal_samples = np.sum(interictal_mask)
        interictal_hours = (interictal_samples * sampling_period) / 3600.0
        metrics['fpr_per_hour'] = false_positives / interictal_hours if interictal_hours > 0 else np.nan
        
        return metrics


# ============================================================================
# RESULTS PROCESSOR
# ============================================================================

class ResultsProcessor:
    """Process raw predictions and generate all variants"""
    
    def __init__(self, raw_results: Dict, run_dir: Path, sampling_period: float = 5.0):
        self.raw_results = raw_results
        self.run_dir = run_dir
        self.processed_results = []
        self.sampling_period = sampling_period
    
    def process_all_variants(
        self,
        ma_windows: List[int] = [1, 3, 5, 7],
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    ):
        """Process all combinations of MA windows and thresholds"""
        
        print(f"  Processing variants with sampling_period={self.sampling_period}s")
        
        for outer_fold in self.raw_results['outer_folds']:
            fold_results = self._process_outer_fold(
                outer_fold,
                ma_windows,
                thresholds
            )
            self.processed_results.append(fold_results)
        
        return self.processed_results
    
    def _process_outer_fold(
        self,
        outer_fold: Dict,
        ma_windows: List[int],
        thresholds: List[float]
    ) -> Dict:
        """Process one outer fold with all variants"""
        
        fold_num = outer_fold['outer_fold']
        y_test = np.array(outer_fold['y_test'])
        
        # Extract test predictions and validation AUCs from inner folds
        test_probs_stack = []
        val_aucs = []
        
        for inner_fold in outer_fold['inner_folds']:
            test_probs_stack.append(np.array(inner_fold['test_probs']))
            val_aucs.append(inner_fold['best_val_auc'])
        
        test_probs_stack = np.stack(test_probs_stack)
        val_aucs = np.array(val_aucs)
        
        # Weight calculation based on validation AUC
        if np.sum(val_aucs) > 0:
            weights = val_aucs / val_aucs.sum()
        else:
            weights = np.ones_like(val_aucs) / len(val_aucs)
        
        # Base ensemble (weighted average)
        base_probs = weighted_ensemble(test_probs_stack, weights)
        
        fold_results = {
            'fold': fold_num,
            'y_test': y_test,
            'variants': {},
            'base_probs': base_probs
        }
        
        # Process each MA window and threshold combination
        for ma_window in ma_windows:
            probs = moving_average(base_probs, y_test, ma_window) if ma_window > 1 else base_probs.copy()
            
            # Store MA probs for threshold analysis
            fold_results[f'probs_ma{ma_window}'] = probs
            
            for threshold in thresholds:
                preds = apply_threshold(probs, threshold)
                metrics = MetricsCalculator.compute_metrics(
                    y_test, preds, probs, self.sampling_period
                )
                
                variant_name = f"ma_{ma_window}_thr_{threshold:.2f}"
                fold_results['variants'][variant_name] = {
                    'probs': probs,
                    'preds': preds,
                    'metrics': metrics,
                    'config': {
                        'ma_window': ma_window,
                        'threshold': threshold
                    }
                }
        
        return fold_results


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Generate essential visualizations"""
    
    def __init__(self, run_dir: Path, top_n: int = 20):
        self.run_dir = run_dir
        self.viz_dir = run_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        self.top_n = top_n
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def plot_roc_curves(self, processed_results: List[Dict], variant_name: str):
        """Plot ROC curves for a specific variant across all folds"""
        plt.figure(figsize=(10, 8))
        
        has_data = False
        for fold_result in processed_results:
            if variant_name not in fold_result['variants']:
                continue
            
            variant = fold_result['variants'][variant_name]
            y_test = fold_result['y_test']
            probs = variant['probs']
            
            try:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, alpha=0.7, 
                        label=f"Fold {fold_result['fold']} (AUC={roc_auc:.3f})")
                has_data = True
            except:
                continue
        
        if not has_data:
            plt.close()
            return
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {variant_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'roc_{variant_name}.png', bbox_inches='tight')
        plt.close()
    
    def plot_threshold_sensitivity_analysis(
        self, 
        processed_results: List[Dict], 
        ma_window: int,
        sampling_period: float = 5.0
    ):
        """Analyze how metrics change with threshold"""
        thresholds = np.linspace(0.1, 0.9, 17)
        
        all_aucs = []
        all_sens = []
        all_fprs = []
        
        for threshold in thresholds:
            fold_aucs = []
            fold_sens = []
            fold_fprs = []
            
            for fold_result in processed_results:
                probs_key = f'probs_ma{ma_window}'
                
                if probs_key not in fold_result:
                    continue
                
                probs = fold_result[probs_key]
                y_test = fold_result['y_test']
                
                # Re-threshold at this level
                preds = apply_threshold(probs, threshold)
                metrics = MetricsCalculator.compute_metrics(
                    y_test, preds, probs, sampling_period
                )
                
                fold_aucs.append(metrics['auc'])
                if not np.isnan(metrics['sensitivity']):
                    fold_sens.append(metrics['sensitivity'])
                fold_fprs.append(metrics['fpr_per_hour'])
            
            all_aucs.append(np.mean(fold_aucs) if fold_aucs else np.nan)
            all_sens.append(np.mean(fold_sens) if fold_sens else np.nan)
            all_fprs.append(np.mean(fold_fprs) if fold_fprs else np.nan)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(thresholds, all_aucs, 'o-', color='blue', linewidth=2)
        axes[0].set_xlabel('Threshold', fontsize=12)
        axes[0].set_ylabel('AUC', fontsize=12)
        axes[0].set_title('AUC vs Threshold', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(thresholds, all_sens, 'o-', color='green', linewidth=2)
        axes[1].set_xlabel('Threshold', fontsize=12)
        axes[1].set_ylabel('Sensitivity', fontsize=12)
        axes[1].set_title('Sensitivity vs Threshold', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(thresholds, all_fprs, 'o-', color='red', linewidth=2)
        axes[2].set_xlabel('Threshold', fontsize=12)
        axes[2].set_ylabel('FPR/hour', fontsize=12)
        axes[2].set_title('FPR/hour vs Threshold', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Sensitivity Analysis: MA Window = {ma_window}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'threshold_analysis_ma{ma_window}.png', bbox_inches='tight')
        plt.close()
    
    def plot_ma_window_comparison(self, processed_results: List[Dict], threshold: float):
        """Compare performance across different MA windows"""
        windows = [1, 3, 5, 7, 10]
        
        metrics_by_window = {w: {'auc': [], 'sens': [], 'fpr': []} for w in windows}
        
        for window in windows:
            variant_name = f"ma_{window}_thr_{threshold:.2f}"
            
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    metrics = fold_result['variants'][variant_name]['metrics']
                    if not np.isnan(metrics['auc']):
                        metrics_by_window[window]['auc'].append(metrics['auc'])
                    if not np.isnan(metrics['sensitivity']):
                        metrics_by_window[window]['sens'].append(metrics['sensitivity'])
                    if not np.isnan(metrics['fpr_per_hour']):
                        metrics_by_window[window]['fpr'].append(metrics['fpr_per_hour'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        mean_aucs = [np.mean(metrics_by_window[w]['auc']) if metrics_by_window[w]['auc'] else np.nan for w in windows]
        std_aucs = [np.std(metrics_by_window[w]['auc']) if metrics_by_window[w]['auc'] else 0 for w in windows]
        axes[0].errorbar(windows, mean_aucs, yerr=std_aucs, marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0].set_xlabel('MA Window Size', fontsize=12)
        axes[0].set_ylabel('AUC', fontsize=12)
        axes[0].set_title('AUC vs MA Window', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        mean_sens = [np.mean(metrics_by_window[w]['sens']) if metrics_by_window[w]['sens'] else np.nan for w in windows]
        std_sens = [np.std(metrics_by_window[w]['sens']) if metrics_by_window[w]['sens'] else 0 for w in windows]
        axes[1].errorbar(windows, mean_sens, yerr=std_sens, marker='o', capsize=5, linewidth=2, markersize=8)
        axes[1].set_xlabel('MA Window Size', fontsize=12)
        axes[1].set_ylabel('Sensitivity', fontsize=12)
        axes[1].set_title('Sensitivity vs MA Window', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        mean_fprs = [np.mean(metrics_by_window[w]['fpr']) if metrics_by_window[w]['fpr'] else np.nan for w in windows]
        std_fprs = [np.std(metrics_by_window[w]['fpr']) if metrics_by_window[w]['fpr'] else 0 for w in windows]
        axes[2].errorbar(windows, mean_fprs, yerr=std_fprs, marker='o', capsize=5, linewidth=2, markersize=8)
        axes[2].set_xlabel('MA Window Size', fontsize=12)
        axes[2].set_ylabel('FPR/hour', fontsize=12)
        axes[2].set_title('FPR/hour vs MA Window', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'MA Window Comparison: threshold={threshold}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'ma_comparison_thr{threshold:.2f}.png', bbox_inches='tight')
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
            plt.close()
            return
        
        # Plot all points
        scatter = plt.scatter(valid_data['mean_fpr_per_hour'], 
                   valid_data['mean_sensitivity'],
                   c=valid_data['mean_auc'],
                   cmap='viridis',
                   s=100,
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
                'r-', linewidth=2, label='Pareto Frontier', marker='o', markersize=10)
        
        plt.colorbar(scatter, label='Mean AUC')
        plt.xlabel('Mean FPR per Hour', fontsize=12)
        plt.ylabel('Mean Sensitivity', fontsize=12)
        plt.title('Sensitivity vs FPR Tradeoff (Pareto Frontier)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'pareto_frontier.png', bbox_inches='tight')
        plt.close()
        
        # Save Pareto optimal variants to CSV
        pareto_df.to_csv(self.run_dir / 'pareto_optimal_variants.csv', index=False)
        print(f"  ✅ Saved Pareto optimal variants ({len(pareto_df)} points)")


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
            for fold_result in processed_results:
                if variant_name in fold_result['variants']:
                    config = fold_result['variants'][variant_name]['config']
                    summary.update({f'config_{k}': v for k, v in config.items()})
                    break
            
            # Compute mean and std for key metrics
            for metric in ['auc', 'sensitivity', 'fpr_per_hour']:
                values = [m[metric] for m in variant_metrics if not (isinstance(m[metric], float) and np.isnan(m[metric]))]
                if values:
                    summary[f'mean_{metric}'] = np.mean(values)
                    summary[f'std_{metric}'] = np.std(values)
                else:
                    summary[f'mean_{metric}'] = np.nan
                    summary[f'std_{metric}'] = np.nan
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def create_best_variants_table(summary_df: pd.DataFrame, top_n: Union[int, str] = 10) -> Dict[str, pd.DataFrame]:
        """Create tables of best variants for each metric"""
        metrics = ['auc', 'sensitivity']
        best_tables = {}
        
        # Determine actual top_n value
        if isinstance(top_n, str) and top_n.lower() == 'all':
            n = len(summary_df)
        else:
            n = int(top_n)
        
        for metric in metrics:
            mean_col = f'mean_{metric}'
            if mean_col not in summary_df.columns:
                continue
            
            best_variants = summary_df.nlargest(n, mean_col)
            
            # Include relevant columns
            columns_to_include = ['variant', 'mean_auc', 'std_auc', 'mean_sensitivity', 'std_sensitivity', 
                                 'mean_fpr_per_hour', 'std_fpr_per_hour']
            
            best_tables[metric] = best_variants[
                [col for col in columns_to_include if col in best_variants.columns]
            ].copy()
        
        return best_tables
    
    @staticmethod
    def create_ma_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compare moving average windows"""
        if 'config_ma_window' not in summary_df.columns:
            return pd.DataFrame()
        
        ma_windows = summary_df['config_ma_window'].unique()
        
        comparison_data = []
        for ma_window in sorted(ma_windows):
            window_data = summary_df[summary_df['config_ma_window'] == ma_window]
            
            row = {
                'ma_window': ma_window, 
                'n_variants': len(window_data),
                'avg_auc': window_data['mean_auc'].mean(),
                'avg_sensitivity': window_data['mean_sensitivity'].mean(),
                'avg_fpr_per_hour': window_data['mean_fpr_per_hour'].mean()
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('ma_window')
    
    @staticmethod
    def create_threshold_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compare thresholds"""
        if 'config_threshold' not in summary_df.columns:
            return pd.DataFrame()
        
        thresholds = summary_df['config_threshold'].unique()
        
        comparison_data = []
        for threshold in sorted(thresholds):
            threshold_data = summary_df[summary_df['config_threshold'] == threshold]
            
            row = {
                'threshold': threshold, 
                'n_variants': len(threshold_data),
                'avg_auc': threshold_data['mean_auc'].mean(),
                'avg_sensitivity': threshold_data['mean_sensitivity'].mean(),
                'avg_fpr_per_hour': threshold_data['mean_fpr_per_hour'].mean()
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('threshold')


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_run(
    run_dir: str, 
    ma_windows: List[int] = None,
    thresholds: List[float] = None,
    top_n: Union[int, str] = 10,
    sampling_period: float = 5.0
):
    """Main analysis function"""
    
    run_path = Path(run_dir)
    
    # Load raw predictions
    raw_predictions_path = run_path / 'raw_predictions.pkl'
    if not raw_predictions_path.exists():
        print(f"❌ Error: {raw_predictions_path} not found!")
        return
    
    print(f"\n{'='*80}")
    print(f"🔬 ANALYZING RUN: {run_dir}")
    print(f"{'='*80}\n")
    
    with open(raw_predictions_path, 'rb') as f:
        raw_results = pickle.load(f)
    
    # Load config
    config_path = run_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("📋 Configuration:")
        for key, value in config['arguments'].items():
            print(f"  {key}: {value}")
        print()
    
    # Set defaults
    if ma_windows is None:
        ma_windows = [1, 3, 5, 7, 10]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Parse top_n
    if isinstance(top_n, str):
        if top_n.lower() == 'all':
            display_n = 'all'
            n_display = None
        else:
            display_n = int(top_n)
            n_display = display_n
    else:
        display_n = top_n
        n_display = top_n
    
    print(f"⚙️  Analysis Parameters:")
    print(f"  MA windows: {ma_windows}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Top-N display: {display_n}")
    print(f"  Sampling period: {sampling_period}s")
    print()
    
    # Process all variants
    print("🔄 Processing all variants...")
    processor = ResultsProcessor(raw_results, run_path, sampling_period)
    processed_results = processor.process_all_variants(
        ma_windows=ma_windows,
        thresholds=thresholds
    )
    
    # Generate summary statistics
    print("\n📊 Generating summary statistics...")
    summary_df = SummaryGenerator.create_variant_summary(processed_results)
    summary_df.to_csv(run_path / 'variant_summary.csv', index=False)
    print(f"  ✅ Saved variant_summary.csv ({len(summary_df)} variants)")
    
    # Best variants tables
    best_tables = SummaryGenerator.create_best_variants_table(summary_df, top_n=display_n)
    for metric, table in best_tables.items():
        table.to_csv(run_path / f'best_variants_{metric}.csv', index=False)
        print(f"  ✅ Saved best_variants_{metric}.csv")
    
    # Comparison tables
    ma_comparison = SummaryGenerator.create_ma_comparison_table(summary_df)
    ma_comparison.to_csv(run_path / 'ma_window_comparison.csv', index=False)
    print(f"  ✅ Saved ma_window_comparison.csv")
    
    threshold_comparison = SummaryGenerator.create_threshold_comparison_table(summary_df)
    threshold_comparison.to_csv(run_path / 'threshold_comparison.csv', index=False)
    print(f"  ✅ Saved threshold_comparison.csv")
    
    # Determine display count
    if n_display is None:
        n_display = len(summary_df)
    else:
        n_display = min(n_display, len(summary_df))
    
    # Display top results
    print("\n" + "="*80)
    print(f"🏆 TOP {n_display} VARIANTS BY AUC:")
    print("="*80)
    if len(summary_df) > 0:
        top_auc = summary_df.nlargest(n_display, 'mean_auc')[
            ['variant', 'mean_auc', 'std_auc', 'mean_sensitivity', 'mean_fpr_per_hour']
        ]
        print(top_auc.to_string(index=False))
    else:
        print("  No variants found!")

    print("\n" + "="*80)
    print(f"⚡ TOP {n_display} BY SENSITIVITY:")
    print("="*80)
    if len(summary_df) > 0:
        top_sen = summary_df.sort_values(['mean_sensitivity', 'mean_fpr_per_hour'], ascending=[False, True]).head(n_display)[
            ['variant', 'mean_auc', 'std_auc', 'mean_sensitivity', 'mean_fpr_per_hour']
        ]
        print(top_sen.to_string(index=False))
    else:
        print("  No variants found!")
    
    if len(ma_comparison) > 0:
        print("\n" + "="*80)
        print("📉 MOVING AVERAGE WINDOW COMPARISON:")
        print("="*80)
        print(ma_comparison.to_string(index=False))
    
    if len(threshold_comparison) > 0:
        print("\n" + "="*80)
        print("🎯 THRESHOLD COMPARISON:")
        print("="*80)
        print(threshold_comparison.to_string(index=False))
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    visualizer = Visualizer(run_path, top_n=n_display if isinstance(display_n, int) else 20)
    
    # Get best variant for detailed plots
    if len(summary_df) > 0:
        best_variant = summary_df.nlargest(1, 'mean_auc').iloc[0]['variant']
        print(f"  🌟 Best variant: {best_variant}")
        
        # ROC curves for best variant
        visualizer.plot_roc_curves(processed_results, best_variant)
        print(f"  ✅ Generated ROC curves")
        
        # Threshold analysis for different MA windows
        for ma_win in ma_windows:
            visualizer.plot_threshold_sensitivity_analysis(
                processed_results, ma_win, sampling_period=sampling_period
            )
        print(f"  ✅ Generated threshold sensitivity analyses")
        
        # MA window comparison for different thresholds
        for thr in [0.4, 0.5, 0.6]:
            visualizer.plot_ma_window_comparison(processed_results, threshold=thr)
        print(f"  ✅ Generated MA window comparisons")
        
        # Pareto frontier
        visualizer.plot_pareto_frontier(summary_df)
        print(f"  ✅ Generated Pareto frontier")
    
    print(f"\n{'='*80}")
    print(f"✨ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {run_dir}")
    print(f"📊 Visualizations saved to: {run_path / 'visualizations'}")
    print(f"🎯 Total variants analyzed: {len(summary_df)}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simplified Analysis of Seizure Prediction Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze most recent run
  python analyze_results3.py
  
  # Analyze specific run
  python analyze_results3.py --run_dir runs/run1_20240101_120000
  
  # Show all variants
  python analyze_results3.py --top_n all
  
  # Customize parameters
  python analyze_results3.py --ma_windows 1 3 5 7 --thresholds 0.3 0.5 0.7
  
  # Custom sampling period
  python analyze_results3.py --sampling_period 4.0

Features:
- ✅ Moving Average windows
- ✅ Threshold analysis
- ✅ Key metrics: AUC, Sensitivity, FPR/hour
- ✅ Essential visualizations only
- ❌ No calibration methods
- ❌ No training curves
- ❌ No per-fold visualizations
        """
    )
    
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Path to run directory (default: most recent)')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Path to runs directory')
    parser.add_argument('--ma_windows', type=int, nargs='+',
                       help='Moving average windows to analyze')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       help='Thresholds to analyze')
    parser.add_argument('--top_n', type=str, default='10',
                       help='Number of top variants to display (or "all")')
    parser.add_argument('--sampling_period', type=float, default=5.0,
                       help='Sampling period in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        # Find most recent run
        runs_path = Path(args.runs_dir)
        if not runs_path.exists():
            print(f"❌ Error: Runs directory {runs_path} does not exist")
            return
        
        run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run')]
        if not run_dirs:
            print(f"❌ Error: No run directories found in {runs_path}")
            return
        
        run_dir = str(max(run_dirs, key=lambda x: x.stat().st_mtime))
        print(f"📂 Analyzing most recent run: {Path(run_dir).name}\n")
    
    # Run analysis
    analyze_run(
        run_dir,
        ma_windows=args.ma_windows,
        thresholds=args.thresholds,
        top_n=args.top_n,
        sampling_period=args.sampling_period
    )


if __name__ == "__main__":
    main()