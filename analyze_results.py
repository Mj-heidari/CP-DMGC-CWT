import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix
import os

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

def plot_training_curves(detailed_results, run_dir):
    """Plot training curves for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['train_loss', 'val_loss', 'train_acc', 'val_auc']
    titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation AUC']
    
    for metric_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[metric_idx]
        
        for fold_idx, fold_data in enumerate(detailed_results):
            for inner_fold_idx, inner_fold in enumerate(fold_data['inner_fold_results']):
                epochs = [log['epoch'] for log in inner_fold['epoch_logs']]
                values = [log[metric] for log in inner_fold['epoch_logs']]
                
                label = f"Fold {fold_idx+1}-{inner_fold_idx+1}" if len(detailed_results) <= 3 else None
                alpha = 0.7 if len(detailed_results) <= 3 else 0.3
                ax.plot(epochs, values, alpha=alpha, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if len(detailed_results) <= 3:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison(results, run_dir):
    """Plot comparison between raw and moving average results"""
    folds = [r['fold'] for r in results]
    
    # Extract metrics
    raw_aucs = [r['raw']['auc'] for r in results]
    ma_aucs = [r['moving_average']['auc'] for r in results]
    
    raw_sens = [r['raw']['sensitivity'] for r in results if not np.isnan(r['raw']['sensitivity'])]
    ma_sens = [r['moving_average']['sensitivity'] for r in results if not np.isnan(r['moving_average']['sensitivity'])]
    
    raw_fprs = [r['raw']['fpr_per_hour'] for r in results]
    ma_fprs = [r['moving_average']['fpr_per_hour'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AUC comparison
    x = np.arange(len(folds))
    width = 0.35
    
    axes[0].bar(x - width/2, raw_aucs, width, label='Raw', alpha=0.8)
    axes[0].bar(x + width/2, ma_aucs, width, label='Moving Average', alpha=0.8)
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('AUC Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Fold {f}' for f in folds])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sensitivity comparison (if available)
    if raw_sens and ma_sens:
        x_sens = np.arange(len(raw_sens))
        axes[1].bar(x_sens - width/2, raw_sens, width, label='Raw', alpha=0.8)
        axes[1].bar(x_sens + width/2, ma_sens, width, label='Moving Average', alpha=0.8)
        axes[1].set_xlabel('Fold (with seizures)')
        axes[1].set_ylabel('Sensitivity')
        axes[1].set_title('Sensitivity Comparison')
        axes[1].set_xticks(x_sens)
        axes[1].set_xticklabels([f'Fold {i+1}' for i in range(len(raw_sens))])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # FPR comparison
    axes[2].bar(x - width/2, raw_fprs, width, label='Raw', alpha=0.8)
    axes[2].bar(x + width/2, ma_fprs, width, label='Moving Average', alpha=0.8)
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('FPR per Hour')
    axes[2].set_title('False Positive Rate Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'Fold {f}' for f in folds])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_distributions(results, run_dir):
    """Plot distribution of predictions"""
    fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)
    
    for fold_idx, fold_result in enumerate(results):
        y_true = np.array(fold_result['predictions']['y_test'])
        probs_raw = np.array(fold_result['predictions']['final_probs'])
        probs_ma = np.array(fold_result['predictions']['final_probs_ma'])
        
        # Raw predictions
        axes[0, fold_idx].hist(probs_raw[y_true == 0], alpha=0.7, bins=20, 
                              label='Interictal', density=True)
        axes[0, fold_idx].hist(probs_raw[y_true == 1], alpha=0.7, bins=20, 
                              label='Preictal', density=True)
        axes[0, fold_idx].axvline(0.5, color='red', linestyle='--', alpha=0.7)
        axes[0, fold_idx].set_title(f'Fold {fold_idx+1} - Raw Predictions')
        axes[0, fold_idx].set_xlabel('Prediction Probability')
        axes[0, fold_idx].set_ylabel('Density')
        axes[0, fold_idx].legend()
        axes[0, fold_idx].grid(True, alpha=0.3)
        
        # Moving average predictions
        axes[1, fold_idx].hist(probs_ma[y_true == 0], alpha=0.7, bins=20, 
                              label='Interictal', density=True)
        axes[1, fold_idx].hist(probs_ma[y_true == 1], alpha=0.7, bins=20, 
                              label='Preictal', density=True)
        axes[1, fold_idx].axvline(0.5, color='red', linestyle='--', alpha=0.7)
        axes[1, fold_idx].set_title(f'Fold {fold_idx+1} - Moving Average Predictions')
        axes[1, fold_idx].set_xlabel('Prediction Probability')
        axes[1, fold_idx].set_ylabel('Density')
        axes[1, fold_idx].legend()
        axes[1, fold_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(run_dir) / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results, config, run_dir):
    """Create a summary table of results"""
    raw_aucs = [r['raw']['auc'] for r in results]
    ma_aucs = [r['moving_average']['auc'] for r in results]
    
    raw_sens = [r['raw']['sensitivity'] for r in results if not np.isnan(r['raw']['sensitivity'])]
    ma_sens = [r['moving_average']['sensitivity'] for r in results if not np.isnan(r['moving_average']['sensitivity'])]
    
    raw_fprs = [r['raw']['fpr_per_hour'] for r in results]
    ma_fprs = [r['moving_average']['fpr_per_hour'] for r in results]
    
    summary_data = {
        'Metric': ['AUC', 'Sensitivity', 'FPR/hour'],
        'Raw Mean': [f"{np.mean(raw_aucs):.4f}", 
                    f"{np.mean(raw_sens):.4f}" if raw_sens else "N/A",
                    f"{np.mean(raw_fprs):.4f}"],
        'Raw Std': [f"±{np.std(raw_aucs):.4f}", 
                   f"±{np.std(raw_sens):.4f}" if raw_sens else "N/A",
                   f"±{np.std(raw_fprs):.4f}"],
        'MA Mean': [f"{np.mean(ma_aucs):.4f}", 
                   f"{np.mean(ma_sens):.4f}" if ma_sens else "N/A",
                   f"{np.mean(ma_fprs):.4f}"],
        'MA Std': [f"±{np.std(ma_aucs):.4f}", 
                  f"±{np.std(ma_sens):.4f}" if ma_sens else "N/A",
                  f"±{np.std(ma_fprs):.4f}"]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Add configuration info
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
    
    # Save tables
    df.to_csv(Path(run_dir) / 'summary_results.csv', index=False)
    config_info.to_csv(Path(run_dir) / 'configuration.csv', index=False)
    
    # Create HTML report
    html_content = f"""
    <html>
    <head><title>Seizure Prediction Results - {config['model_name']}</title></head>
    <body>
    <h1>Seizure Prediction Results</h1>
    <h2>Configuration</h2>
    {config_info.to_html(index=False, table_id="config_table")}
    
    <h2>Results Summary</h2>
    {df.to_html(index=False, table_id="results_table")}
    
    <h2>Per-Fold Results</h2>
    <h3>Raw Predictions</h3>
    <ul>
    {"".join([f"<li>Fold {r['fold']}: AUC={r['raw']['auc']:.4f}, Sensitivity={r['raw']['sensitivity']}, FPR/h={r['raw']['fpr_per_hour']:.4f}</li>" for r in results])}
    </ul>
    
    <h3>Moving Average Predictions</h3>
    <ul>
    {"".join([f"<li>Fold {r['fold']}: AUC={r['moving_average']['auc']:.4f}, Sensitivity={r['moving_average']['sensitivity']}, FPR/h={r['moving_average']['fpr_per_hour']:.4f}</li>" for r in results])}
    </ul>
    
    <h2>Visualizations</h2>
    <img src="performance_comparison.png" alt="Performance Comparison" style="max-width:100%;">
    <br><br>
    <img src="prediction_distributions.png" alt="Prediction Distributions" style="max-width:100%;">
    <br><br>
    <img src="training_curves.png" alt="Training Curves" style="max-width:100%;">
    
    <style>
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    </style>
    </body>
    </html>
    """
    
    with open(Path(run_dir) / 'report.html', 'w') as f:
        f.write(html_content)

def compare_multiple_runs(run_dirs):
    """Compare results from multiple runs"""
    all_results = []
    all_configs = []
    
    for run_dir in run_dirs:
        results, _, config = load_results(run_dir)
        if results and config:
            all_results.append(results)
            all_configs.append(config)
    
    if not all_results:
        print("No valid results found in the specified directories.")
        return
    
    # Create comparison table
    comparison_data = []
    
    for i, (results, config) in enumerate(zip(all_results, all_configs)):
        raw_aucs = [r['raw']['auc'] for r in results]
        ma_aucs = [r['moving_average']['auc'] for r in results]
        
        raw_sens = [r['raw']['sensitivity'] for r in results if not np.isnan(r['raw']['sensitivity'])]
        ma_sens = [r['moving_average']['sensitivity'] for r in results if not np.isnan(r['moving_average']['sensitivity'])]
        
        raw_fprs = [r['raw']['fpr_per_hour'] for r in results]
        ma_fprs = [r['moving_average']['fpr_per_hour'] for r in results]
        
        comparison_data.append({
            'Run': f"Run {i+1}",
            'Model': config['arguments']['model'],
            'Subject': config['arguments']['subject_id'],
            'Raw AUC': f"{np.mean(raw_aucs):.4f}±{np.std(raw_aucs):.4f}",
            'MA AUC': f"{np.mean(ma_aucs):.4f}±{np.std(ma_aucs):.4f}",
            'Raw Sens': f"{np.mean(raw_sens):.4f}±{np.std(raw_sens):.4f}" if raw_sens else "N/A",
            'MA Sens': f"{np.mean(ma_sens):.4f}±{np.std(ma_sens):.4f}" if ma_sens else "N/A",
            'Raw FPR/h': f"{np.mean(raw_fprs):.4f}±{np.std(raw_fprs):.4f}",
            'MA FPR/h': f"{np.mean(ma_fprs):.4f}±{np.std(ma_fprs):.4f}",
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n=== Multiple Runs Comparison ===")
    print(df_comparison.to_string(index=False))
    
    # Save comparison
    df_comparison.to_csv('runs_comparison.csv', index=False)
    print(f"\nComparison saved to runs_comparison.csv")

def analyze_single_run(run_dir):
    """Analyze a single run"""
    print(f"\n=== Analyzing Run: {run_dir} ===")
    
    results, detailed_results, config = load_results(run_dir)
    
    if not results:
        print(f"No results found in {run_dir}")
        return
    
    print(f"Model: {config['arguments']['model']}")
    print(f"Subject: {config['arguments']['subject_id']}")
    print(f"Epochs: {config['arguments']['epochs']}")
    
    # Print summary
    raw_aucs = [r['raw']['auc'] for r in results]
    ma_aucs = [r['moving_average']['auc'] for r in results]
    
    raw_sens = [r['raw']['sensitivity'] for r in results if not np.isnan(r['raw']['sensitivity'])]
    ma_sens = [r['moving_average']['sensitivity'] for r in results if not np.isnan(r['moving_average']['sensitivity'])]
    
    raw_fprs = [r['raw']['fpr_per_hour'] for r in results]
    ma_fprs = [r['moving_average']['fpr_per_hour'] for r in results]
    
    print(f"\nRaw Results:")
    print(f"  AUC: {np.mean(raw_aucs):.4f} ± {np.std(raw_aucs):.4f}")
    print(f"  Sensitivity: {np.mean(raw_sens):.4f} ± {np.std(raw_sens):.4f}" if raw_sens else "  Sensitivity: N/A")
    print(f"  FPR/hour: {np.mean(raw_fprs):.4f} ± {np.std(raw_fprs):.4f}")
    
    print(f"\nMoving Average Results:")
    print(f"  AUC: {np.mean(ma_aucs):.4f} ± {np.std(ma_aucs):.4f}")
    print(f"  Sensitivity: {np.mean(ma_sens):.4f} ± {np.std(ma_sens):.4f}" if ma_sens else "  Sensitivity: N/A")
    print(f"  FPR/hour: {np.mean(ma_fprs):.4f} ± {np.std(ma_fprs):.4f}")
    
    # Generate plots and reports
    if detailed_results:
        plot_training_curves(detailed_results, run_dir)
        print(f"Training curves saved to {run_dir}/training_curves.png")
    
    plot_performance_comparison(results, run_dir)
    print(f"Performance comparison saved to {run_dir}/performance_comparison.png")
    
    plot_prediction_distributions(results, run_dir)
    print(f"Prediction distributions saved to {run_dir}/prediction_distributions.png")
    
    create_summary_table(results, config, run_dir)
    print(f"Summary report saved to {run_dir}/report.html")

def main():
    parser = argparse.ArgumentParser(description='Analyze seizure prediction results')
    parser.add_argument('--run_dir', type=str, help='Path to single run directory to analyze')
    parser.add_argument('--runs_dir', type=str, default='runs', help='Path to runs directory for multiple run analysis')
    parser.add_argument('--compare_all', action='store_true', help='Compare all runs in the runs directory')
    parser.add_argument('--run_ids', nargs='+', help='Specific run IDs to compare (e.g., run1_20231215_120000)')
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Analyze single run
        analyze_single_run(args.run_dir)
    
    elif args.compare_all:
        # Compare all runs
        runs_dir = Path(args.runs_dir)
        if not runs_dir.exists():
            print(f"Runs directory {runs_dir} does not exist.")
            return
        
        run_dirs = [str(d) for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
        if not run_dirs:
            print(f"No run directories found in {runs_dir}")
            return
        
        print(f"Found {len(run_dirs)} runs to compare")
        compare_multiple_runs(run_dirs)
    
    elif args.run_ids:
        # Compare specific runs
        runs_dir = Path(args.runs_dir)
        run_dirs = [str(runs_dir / run_id) for run_id in args.run_ids]
        
        # Check if directories exist
        valid_dirs = [d for d in run_dirs if Path(d).exists()]
        if not valid_dirs:
            print("None of the specified run directories exist.")
            return
        
        print(f"Comparing {len(valid_dirs)} runs")
        compare_multiple_runs(valid_dirs)
    
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
        
        # Sort by creation time and get the most recent
        most_recent_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        print(f"Analyzing most recent run: {most_recent_run.name}")
        analyze_single_run(str(most_recent_run))

if __name__ == "__main__":
    main()