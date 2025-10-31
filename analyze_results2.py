import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score

def load_results(run_dir):
    """Load results from a run directory"""
    results_path = Path(run_dir) / 'results.json'
    detailed_results_path = Path(run_dir) / 'detailed_results.pkl'
    config_path = Path(run_dir) / 'config.json'
    
    results = None
    detailed_results = None
    config = None
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    if detailed_results_path.exists():
        with open(detailed_results_path, 'rb') as f:
            detailed_results = pickle.load(f)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return results, detailed_results, config


def moving_average_predictions(probs, window_size=3):
    """Apply moving average to predictions"""
    if len(probs) < window_size:
        return probs
    
    smoothed_probs = np.copy(probs).astype(float)
    
    for i in range(len(probs)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        smoothed_probs[i] = np.mean(probs[start_idx:end_idx])
    
    return smoothed_probs


def compute_metrics(probs, preds, y_true):
    """Compute AUC, sensitivity, and FPR/hour"""
    try:
        metric_auc = roc_auc_score(y_true, probs)
    except:
        metric_auc = np.nan
    
    has_preictal = np.any(y_true == 1)
    if has_preictal:
        detected = np.any((y_true == 1) & (preds == 1))
        sensitivity = 1 if detected else 0
    else:
        sensitivity = np.nan
    
    false_positives = np.sum((y_true == 0) & (preds == 1))
    hours = (len(y_true) * 5) / 3600.0
    fpr_per_hour = false_positives / hours if hours > 0 else np.nan
    
    return metric_auc, sensitivity, fpr_per_hour


def analyze_moving_average_windows(results, run_dir, window_sizes=[1, 3, 5, 7, 10, 15, 20]):
    """Analyze impact of different moving average window sizes"""
    
    # Check if results use new format (calibrated) or old format (raw/moving_average)
    use_calibrated = 'calibrated' in results[0]
    
    if use_calibrated:
        base_key = 'calibrated'
    else:
        base_key = 'raw'
    
    all_window_results = {ws: {'auc': [], 'sens': [], 'fpr': []} for ws in window_sizes}
    
    for fold_result in results:
        y_test = np.array(fold_result['predictions']['y_test'])
        
        if use_calibrated:
            base_probs = np.array(fold_result['predictions']['final_probs_calibrated'])
        else:
            base_probs = np.array(fold_result['predictions']['final_probs'])
        
        for ws in window_sizes:
            if ws == 1:
                probs_ma = base_probs
            else:
                probs_ma = moving_average_predictions(base_probs, ws)
            
            preds_ma = (probs_ma >= 0.5).astype(int)
            metric_auc, sensitivity, fpr_per_hour = compute_metrics(probs_ma, preds_ma, y_test)
            
            all_window_results[ws]['auc'].append(metric_auc)
            all_window_results[ws]['sens'].append(sensitivity if not np.isnan(sensitivity) else None)
            all_window_results[ws]['fpr'].append(fpr_per_hour)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # AUC vs window size
    mean_aucs = [np.mean(all_window_results[ws]['auc']) for ws in window_sizes]
    std_aucs = [np.std(all_window_results[ws]['auc']) for ws in window_sizes]
    axes[0].errorbar(window_sizes, mean_aucs, yerr=std_aucs, marker='o', capsize=5)
    axes[0].set_xlabel('Moving Average Window Size')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('AUC vs Moving Average Window Size')
    axes[0].grid(True, alpha=0.3)
    
    # Sensitivity vs window size
    sens_lists = [all_window_results[ws]['sens'] for ws in window_sizes]
    mean_sens = [np.mean([s for s in sl if s is not None]) if any(s is not None for s in sl) else 0 
                 for sl in sens_lists]
    std_sens = [np.std([s for s in sl if s is not None]) if any(s is not None for s in sl) else 0 
                for sl in sens_lists]
    axes[1].errorbar(window_sizes, mean_sens, yerr=std_sens, marker='o', capsize=5)
    axes[1].set_xlabel('Moving Average Window Size')
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_title('Sensitivity vs Moving Average Window Size')
    axes[1].grid(True, alpha=0.3)
    
    # FPR/hour vs window size
    mean_fprs = [np.mean(all_window_results[ws]['fpr']) for ws in window_sizes]
    std_fprs = [np.std(all_window_results[ws]['fpr']) for ws in window_sizes]
    axes[2].errorbar(window_sizes, mean_fprs, yerr=std_fprs, marker='o', capsize=5)
    axes[2].set_xlabel('Moving Average Window Size')
    axes[2].set_ylabel('FPR per Hour')
    axes[2].set_title('FPR/Hour vs Moving Average Window Size')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'moving_average_window_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    summary_df = pd.DataFrame({
        'Window_Size': window_sizes,
        'Mean_AUC': mean_aucs,
        'Std_AUC': std_aucs,
        'Mean_Sensitivity': mean_sens,
        'Std_Sensitivity': std_sens,
        'Mean_FPR_Hour': mean_fprs,
        'Std_FPR_Hour': std_fprs
    })
    summary_df.to_csv(Path(run_dir) / 'moving_average_window_summary.csv', index=False)
    
    print("\n=== Moving Average Window Analysis ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {run_dir}/moving_average_window_analysis.png")


def plot_threshold_curves(results, run_dir):
    """Plot sensitivity and FPR curves for different thresholds"""
    
    use_calibrated = 'calibrated' in results[0]
    
    thresholds = np.linspace(0, 1, 101)
    all_sens = []
    all_fprs = []
    
    for fold_result in results:
        y_test = np.array(fold_result['predictions']['y_test'])
        
        if use_calibrated:
            # Use best performing variant
            probs = np.array(fold_result['predictions']['final_probs_calibrated_ma'])
        else:
            probs = np.array(fold_result['predictions']['final_probs_ma'])
        
        fold_sens = []
        fold_fprs = []
        
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            
            # Sensitivity
            has_preictal = np.any(y_test == 1)
            if has_preictal:
                detected = np.any((y_test == 1) & (preds == 1))
                sensitivity = 1 if detected else 0
            else:
                sensitivity = np.nan
            
            # FPR/hour
            false_positives = np.sum((y_test == 0) & (preds == 1))
            hours = (len(y_test) * 5) / 3600.0
            fpr_per_hour = false_positives / hours if hours > 0 else np.nan
            
            fold_sens.append(sensitivity)
            fold_fprs.append(fpr_per_hour)
        
        all_sens.append(fold_sens)
        all_fprs.append(fold_fprs)
    
    # Convert to arrays
    all_sens = np.array(all_sens)
    all_fprs = np.array(all_fprs)
    
    # Compute mean and std
    mean_sens = np.nanmean(all_sens, axis=0)
    std_sens = np.nanstd(all_sens, axis=0)
    mean_fprs = np.nanmean(all_fprs, axis=0)
    std_fprs = np.nanstd(all_fprs, axis=0)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sensitivity vs Threshold
    axes[0].plot(thresholds, mean_sens, 'b-', linewidth=2, label='Mean Sensitivity')
    axes[0].fill_between(thresholds, mean_sens - std_sens, mean_sens + std_sens, 
                         alpha=0.3, color='b')
    axes[0].axvline(0.5, color='r', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Sensitivity')
    axes[0].set_title('Sensitivity vs Classification Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FPR/hour vs Threshold
    axes[1].plot(thresholds, mean_fprs, 'r-', linewidth=2, label='Mean FPR/hour')
    axes[1].fill_between(thresholds, mean_fprs - std_fprs, mean_fprs + std_fprs, 
                         alpha=0.3, color='r')
    axes[1].axvline(0.5, color='b', linestyle='--', alpha=0.7, label='Threshold = 0.5')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('FPR per Hour')
    axes[1].set_title('FPR/Hour vs Classification Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'threshold_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nThreshold curves saved to {run_dir}/threshold_curves.png")


def plot_all_inner_fold_histograms(detailed_results, results, run_dir, window_sizes=[1, 3, 5, 10]):
    """Plot histograms for all inner folds and test set with different moving averages"""
    
    use_calibrated = 'calibrated' in results[0]
    
    for outer_fold_idx, (fold_detail, fold_result) in enumerate(zip(detailed_results, results)):
        fold_dir = Path(run_dir) / f'fold_{outer_fold_idx + 1}_histograms'
        fold_dir.mkdir(exist_ok=True)
        
        # Get test data
        y_test = np.array(fold_result['predictions']['y_test'])
        
        # Plot for each inner fold
        for inner_fold_data in fold_detail['inner_fold_results']:
            inner_fold_idx = inner_fold_data['inner_fold']
            
            # For inner folds, we'd need to store validation predictions
            # Since we don't have them, we'll skip inner fold histograms
            pass
        
        # Plot test set histograms with different window sizes
        n_windows = len(window_sizes)
        fig, axes = plt.subplots(2, n_windows, figsize=(5*n_windows, 10))
        if n_windows == 1:
            axes = axes.reshape(-1, 1)
        
        # Before calibration
        if use_calibrated:
            probs_original = np.array(fold_result['predictions']['final_probs_original'])
        else:
            probs_original = np.array(fold_result['predictions']['final_probs'])
        
        for col, ws in enumerate(window_sizes):
            if ws == 1:
                probs_ma = probs_original
            else:
                probs_ma = moving_average_predictions(probs_original, ws)
            
            # Plot before calibration
            axes[0, col].hist(probs_ma[y_test == 0], bins=20, alpha=0.7, 
                            label='Interictal', density=True, color='green')
            axes[0, col].hist(probs_ma[y_test == 1], bins=20, alpha=0.7, 
                            label='Preictal', density=True, color='red')
            axes[0, col].axvline(0.5, color='black', linestyle='--', alpha=0.7)
            axes[0, col].set_title(f'Before Calibration (Window={ws})')
            axes[0, col].set_xlabel('Prediction Probability')
            axes[0, col].set_ylabel('Density')
            axes[0, col].legend()
            axes[0, col].grid(True, alpha=0.3)
        
        # After calibration (if available)
        if use_calibrated:
            probs_calibrated = np.array(fold_result['predictions']['final_probs_calibrated'])
            
            for col, ws in enumerate(window_sizes):
                if ws == 1:
                    probs_ma = probs_calibrated
                else:
                    probs_ma = moving_average_predictions(probs_calibrated, ws)
                
                # Plot after calibration
                axes[1, col].hist(probs_ma[y_test == 0], bins=20, alpha=0.7, 
                                label='Interictal', density=True, color='green')
                axes[1, col].hist(probs_ma[y_test == 1], bins=20, alpha=0.7, 
                                label='Preictal', density=True, color='red')
                axes[1, col].axvline(0.5, color='black', linestyle='--', alpha=0.7)
                axes[1, col].set_title(f'After Calibration (Window={ws})')
                axes[1, col].set_xlabel('Prediction Probability')
                axes[1, col].set_ylabel('Density')
                axes[1, col].legend()
                axes[1, col].grid(True, alpha=0.3)
        else:
            # No calibration - hide bottom row
            for col in range(n_windows):
                axes[1, col].text(0.5, 0.5, 'No Calibration', 
                                ha='center', va='center', fontsize=14)
                axes[1, col].set_xticks([])
                axes[1, col].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(fold_dir / f'test_histograms_outer_fold_{outer_fold_idx + 1}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Histograms saved for outer fold {outer_fold_idx + 1}")


def create_summary_table(results, config, run_dir):
    """Create summary table compatible with both old and new result formats"""
    
    use_calibrated = 'calibrated' in results[0]
    
    if use_calibrated:
        # New format with calibration
        variants = ['original', 'original_ma', 'calibrated', 'calibrated_ma']
        variant_labels = ['Original', 'Original + MA', 'Calibrated', 'Calibrated + MA']
    else:
        # Old format
        variants = ['raw', 'moving_average']
        variant_labels = ['Raw', 'Moving Average']
    
    summary_data = []
    
    for variant, label in zip(variants, variant_labels):
        aucs = [r[variant]['auc'] for r in results]
        sens = [r[variant]['sensitivity'] for r in results if not np.isnan(r[variant]['sensitivity'])]
        fprs = [r[variant]['fpr_per_hour'] for r in results]
        
        summary_data.append({
            'Variant': label,
            'Mean_AUC': f"{np.mean(aucs):.4f}",
            'Std_AUC': f"±{np.std(aucs):.4f}",
            'Mean_Sensitivity': f"{np.mean(sens):.4f}" if sens else "N/A",
            'Std_Sensitivity': f"±{np.std(sens):.4f}" if sens else "N/A",
            'Mean_FPR_Hour': f"{np.mean(fprs):.4f}",
            'Std_FPR_Hour': f"±{np.std(fprs):.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    
    # Configuration info
    config_info = pd.DataFrame({
        'Parameter': ['Subject ID', 'Model', 'Epochs', 'Batch Size', 'Learning Rate', 
                     'Outer CV', 'Inner CV', 'Moving Avg Window'],
        'Value': [config['arguments']['subject_id'], 
                 config['arguments']['model'],
                 config['arguments']['epochs'],
                 config['arguments']['batch_size'],
                 config['arguments']['lr'],
                 f"{config['arguments']['outer_cv_mode']} ({config['arguments'].get('outer_cv_method', 'N/A')})",
                 f"{config['arguments']['inner_cv_mode']} ({config['arguments'].get('inner_cv_method', 'N/A')})",
                 config['arguments']['moving_avg_window']]
    })
    
    # Add calibration info if available
    if use_calibrated:
        calibration_info = pd.DataFrame({
            'Parameter': ['Calibration Method', 'Target Percentile'],
            'Value': [config['arguments'].get('calibration_method', 'N/A'),
                     config['arguments'].get('target_percentile', 'N/A')]
        })
        config_info = pd.concat([config_info, calibration_info], ignore_index=True)
    
    # Save tables
    df.to_csv(Path(run_dir) / 'summary_results.csv', index=False)
    config_info.to_csv(Path(run_dir) / 'configuration.csv', index=False)
    
    print("\n=== Summary Results ===")
    print(df.to_string(index=False))
    print("\n=== Configuration ===")
    print(config_info.to_string(index=False))


def analyze_single_run(run_dir):
    """Analyze a single run with enhanced visualizations"""
    print(f"\n=== Analyzing Run: {run_dir} ===")
    
    results, detailed_results, config = load_results(run_dir)
    
    if not results:
        print(f"No results found in {run_dir}")
        return
    
    # Create summary table
    create_summary_table(results, config, run_dir)
    
    # Analyze moving average windows
    analyze_moving_average_windows(results, run_dir)
    
    # Plot threshold curves
    plot_threshold_curves(results, run_dir)
    
    # Plot histograms for all folds
    if detailed_results:
        plot_all_inner_fold_histograms(detailed_results, results, run_dir)
    
    print(f"\nAll analyses saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze seizure prediction results (v2)')
    parser.add_argument('--run_dir', type=str, help='Path to single run directory to analyze')
    parser.add_argument('--runs_dir', type=str, default='runs', 
                       help='Path to runs directory for multiple run analysis')
    
    args = parser.parse_args()
    
    if args.run_dir:
        analyze_single_run(args.run_dir)
    else:
        # Find and analyze the most recent run
        runs_dir = Path(args.runs_dir)
        if not runs_dir.exists():
            print(f"Runs directory {runs_dir} does not exist.")
            return
        
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
        if not run_dirs:
            print(f"No run directories found in {runs_dir}")
            return
        
        most_recent_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Analyzing most recent run: {most_recent_run.name}")
        analyze_single_run(str(most_recent_run))


if __name__ == "__main__":
    main()