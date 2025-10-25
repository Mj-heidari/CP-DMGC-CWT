import numpy as np
import pandas as pd
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Project Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

feature_transform_path = os.path.join(project_root, 'transforms', 'feature')
if feature_transform_path not in sys.path:
    sys.path.insert(0, feature_transform_path)

# Add dataset path for utils import
dataset_path = os.path.join(project_root, 'dataset')
if dataset_path not in sys.path:
     sys.path.insert(0, dataset_path)

# --- Local and Project Imports ---
# Use functions directly from features_analysis.utils
from utils import get_feature_extractors, extract_all_features #
# Use invert_uint16_scaling from dataset.utils
from dataset.utils import invert_uint16_scaling #

# --- Cohen's d Function ---
def cohen_d(group1, group2):
    """Calculate Cohen's d for independent samples."""
    # Drop NaNs or Infs which can occur during feature extraction
    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]

    # Check if enough valid samples remain
    if len(group1) < 2 or len(group2) < 2:
        return np.nan # Cannot compute std dev reliably

    # Calculate the size, mean, and standard deviation
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1) # Sample std dev

    # Check for zero standard deviation
    if std1 < 1e-8 and std2 < 1e-8:
         return 0.0 # No variance in either group
    elif std1 < 1e-8:
         pooled_std = std2 # Use the variance of the other group
    elif std2 < 1e-8:
         pooled_std = std1 # Use the variance of the other group
    else:
         # Calculate pooled standard deviation
         pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    if pooled_std < 1e-8:
         return np.inf * np.sign(mean1 - mean2) if mean1 != mean2 else 0.0 # Handle zero pooled_std
    d = (mean1 - mean2) / pooled_std
    return d


def analyze_and_rank_features(X_features, y_labels, feature_names, output_dir):
    """
    Performs statistical analysis, trains a model to rank feature importance,
    saves reports and plots, and exports features to Excel.
    """
    print("\n--- Starting Feature Analysis ---")

    # --- Create DataFrame ---
    df = pd.DataFrame(X_features, columns=feature_names) #
    df['label'] = y_labels #
    df['label_str'] = df['label'].map({0: 'interictal', 1: 'preictal'}) #

    # --- ADDED: Export features + labels to Excel ---
    try:
        excel_output_path = os.path.join(output_dir, "extracted_features_for_analysis.xlsx")
        # Include label_str for clarity in Excel
        df_to_save = df[['label', 'label_str'] + feature_names]
        df_to_save.to_excel(excel_output_path, index=False)
        print(f"Features (used for analysis) saved to: {excel_output_path}")
    except Exception as e:
        print(f"Error saving features to Excel: {e}. Try 'pip install openpyxl'.")
    # --- END Excel Export ---

    # Separate features for analysis
    interictal_features = df[df['label'] == 0][feature_names] #
    preictal_features = df[df['label'] == 1][feature_names] #

    if preictal_features.empty or interictal_features.empty:
        print("Not enough data for both classes found. Cannot perform comparative analysis.")
        # Save basic info if possible
        if not df.empty:
             report_path = os.path.join(output_dir, "feature_ranking_report.txt")
             with open(report_path, 'w') as f:
                 f.write("--- Feature Ranking Report ---\n\n")
                 f.write("Insufficient data for comparative analysis (missing preictal or interictal samples).\n")
                 f.write(f"Total samples analyzed: {len(df)}\n")
                 f.write(f"Label distribution:\n{df['label_str'].value_counts().to_string()}\n\n")
                 f.write("Feature list:\n")
                 f.write("\n".join(feature_names))
        return # Stop analysis if one class is missing

    # 1. Statistical Analysis (T-test)
    print("Performing statistical tests (t-test) for each feature...")
    stats_results = []
    for feature in feature_names: #
        # Perform test only if both groups have valid data for the feature
        inter_vals = interictal_features[feature].dropna()
        preict_vals = preictal_features[feature].dropna()
        if len(inter_vals) >= 2 and len(preict_vals) >= 2:
            stat, p_value = ttest_ind(inter_vals, preict_vals, equal_var=False) # Welch's t-test
        else:
            p_value = np.nan # Not enough data for test
        stats_results.append({'feature': feature, 'p_value': p_value}) #

    stats_df = pd.DataFrame(stats_results).sort_values(by='p_value').reset_index(drop=True) #

    # --- ADDED: Calculate Cohen's d ---
    print("Calculating Cohen's d for each feature...")
    cohen_results = []
    for feature in feature_names:
        d_value = cohen_d(
            interictal_features[feature], # Pass the pandas Series
            preictal_features[feature]
        )
        cohen_results.append({'feature': feature, 'cohen_d': d_value})

    cohen_df = pd.DataFrame(cohen_results)
    # --- END Cohen's d ---

    # 2. Feature Importance from Random Forest
    print("Training Random Forest to get feature importances...")
    # Ensure balanced data was passed in, or handle imbalance here if needed
    X_train, X_test, y_train, y_test = train_test_split( #
        df[feature_names], df['label'], test_size=0.3, random_state=42, stratify=df['label'] #
    )

    # Handle potential NaNs/Infs before scaling/training
    X_train = X_train.fillna(X_train.mean()).replace([np.inf, -np.inf], X_train.mean())
    X_test = X_test.fillna(X_train.mean()).replace([np.inf, -np.inf], X_train.mean()) # Use training mean for test set

    scaler = StandardScaler() #
    X_train_scaled = scaler.fit_transform(X_train) #
    # X_test_scaled = scaler.transform(X_test) # Scale test set if evaluating model performance

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') # Use balanced weights if data might be imbalanced
    model.fit(X_train_scaled, y_train) #

    importances = model.feature_importances_ #
    forest_df = pd.DataFrame({ #
        'feature': feature_names, #
        'importance': importances #
    }).sort_values(by='importance', ascending=False).reset_index(drop=True) #

    # 3. Combine and Save Report
    final_report_df = pd.merge(forest_df, stats_df, on='feature') #
    final_report_df = pd.merge(final_report_df, cohen_df, on='feature') # Merge Cohen's d results

    # Reorder columns for clarity
    final_report_df = final_report_df[['feature', 'importance', 'p_value', 'cohen_d']]

    report_path = os.path.join(output_dir, "feature_ranking_report.txt") #
    with open(report_path, 'w') as f: #
        f.write("--- Feature Ranking Report ---\n\n") #
        f.write("Features are ranked by Random Forest importance.\n") #
        f.write("p_value (t-test): Statistical significance of difference (lower is more significant).\n")
        f.write("cohen_d: Standardized effect size (magnitude of difference).\n\n")
        # Format numbers for better readability in text file
        f.write(final_report_df.to_string(index=False, float_format="%.4g")) # Use general format
    print(f"Full report saved to: {report_path}") #

    # 4. Generate and Save Plots
    # Use top features based on importance ranking
    top_n_features = 15
    top_features = final_report_df['feature'].head(top_n_features).tolist() #

    # Plot 1: Feature Importance Bar Chart
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=final_report_df.head(top_n_features), palette='viridis')
    plt.title(f'Top {top_n_features} Most Important Features (Random Forest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    importance_plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {importance_plot_path}")

    # Plot 2: Box Plots of Top Features
    num_plots = len(top_features)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()
    for i, feature in enumerate(top_features):
        # Handle potential non-finite values before plotting
        plot_data = df[['label_str', feature]].dropna()
        sns.boxplot(x='label_str', y=feature, data=plot_data, ax=axes[i], palette='Set2')
        # Add p-value and Cohen's d from report
        p_val = final_report_df.loc[final_report_df['feature'] == feature, 'p_value'].iloc[0]
        d_val = final_report_df.loc[final_report_df['feature'] == feature, 'cohen_d'].iloc[0]
        axes[i].set_title(f'{feature}\n(p={p_val:.2g}, d={d_val:.2f})', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Value')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution Comparison of Top Features', fontsize=16, y=1.01)
    plt.tight_layout()
    distributions_plot_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(distributions_plot_path)
    plt.close()
    print(f"Feature distribution plots saved to: {distributions_plot_path}")


    # Plot 3: Scatter Plot of Top 2 Features
    if len(top_features) >= 2:
        plt.figure(figsize=(10, 8))
        # Handle potential non-finite values before plotting
        scatter_data = df[[top_features[0], top_features[1], 'label_str']].dropna()
        sns.scatterplot(
            x=top_features[0], y=top_features[1], hue='label_str', data=scatter_data,
            palette={'interictal': 'blue', 'preictal': 'red'}, alpha=0.6
        )
        plt.title(f'Scatter Plot: {top_features[0]} vs {top_features[1]}')
        plt.legend(title='Label')
        plt.grid(True, alpha=0.3)
        scatter_plot_path = os.path.join(output_dir, "top_features_scatterplot.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f"Top 2 features scatter plot saved to: {scatter_plot_path}")


    # Plot 4: Correlation Heatmap of Top Features
    plt.figure(figsize=(12, 10))
    # Calculate correlation on the original (potentially non-finite) data, handle NaNs in heatmap
    corr = df[top_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, annot_kws={"size": 8})
    plt.title(f'Correlation Matrix of Top {top_n_features} Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "feature_correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Feature correlation heatmap saved to: {heatmap_path}")


    # Plot 5: PCA Plot
    try:
         # Use the scaled data prepared for Random Forest (already handled NaNs/Infs)
         pca = PCA(n_components=2)
         X_pca = pca.fit_transform(X_train_scaled) # Use scaled training data for PCA fit/transform

         plt.figure(figsize=(10, 8))
         scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.7) # Use y_train labels
         plt.title('PCA of Features (on Scaled Training Data)')
         plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
         plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
         # Ensure labels match the plotted data (y_train)
         legend_labels = ['interictal' if i == 0 else 'preictal' for i in model.classes_]
         plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
         plt.grid(True, alpha=0.3)
         pca_plot_path = os.path.join(output_dir, "pca_plot.png")
         plt.savefig(pca_plot_path)
         plt.close()
         print(f"PCA plot saved to: {pca_plot_path}")
    except Exception as e:
         print(f"Could not generate PCA plot: {e}")


    # Plot 6: Feature Mean Difference Heatmap
    # Calculate means only on finite values
    mean_interictal = interictal_features[top_features].apply(lambda x: x[np.isfinite(x)].mean())
    mean_preictal = preictal_features[top_features].apply(lambda x: x[np.isfinite(x)].mean())

    mean_comparison = pd.DataFrame({
        'interictal': mean_interictal,
        'preictal': mean_preictal
    })
    # Calculate percentage difference carefully, avoiding division by zero
    diff_num = mean_comparison['preictal'] - mean_comparison['interictal']
    diff_den = mean_comparison['interictal'].abs().replace(0, 1e-8) # Avoid division by zero
    mean_comparison['difference (%)'] = (diff_num / diff_den) * 100

    plt.figure(figsize=(8, 10)) # Adjusted size for vertical labels
    sns.heatmap(mean_comparison, annot=True, fmt=".2f", cmap='viridis', linewidths=.5, cbar_kws={'label': 'Mean Value / % Difference'})
    plt.title('Mean Feature Values & % Difference (Preictal vs. Interictal)')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    means_heatmap_path = os.path.join(output_dir, "feature_means_heatmap.png")
    plt.savefig(means_heatmap_path)
    plt.close()
    print(f"Feature means heatmap saved to: {means_heatmap_path}")


def main(file_path, channel_to_analyze, output_dir):
    """
    Main function to load data, extract all features, and run analysis.
    """
    # 1. Load Data
    try:
        data = np.load(file_path) #
        X_uint16_full = data['X'] #
        y_str_full = data['y'] #
        scales_full = data['scales'] #
        print(f"Data loaded successfully from '{file_path}'.") #
    except FileNotFoundError: #
        print(f"Error: Input file '{file_path}' not found.")
        return
    except KeyError as e: #
         print(f"Error: Missing key '{e}' in the .npz file. Expected 'X', 'y', 'scales'.")
         return
    except Exception as e: #
        print(f"An error occurred while loading the data: {e}") #
        return

    y_full = np.array([1 if label == 'preictal' else 0 for label in y_str_full]) #

    print("\n--- Full Dataset Label Counts ---") #
    unique_labels, counts = np.unique(y_full, return_counts=True) #
    label_map = {0: 'interictal', 1: 'preictal'} #
    for label, count in zip(unique_labels, counts): #
        print(f"Label {label} ({label_map.get(label, 'Unknown')}): {count} segments") #
    print("---------------------------------\n")

    # --- Create Balanced Dataset for Analysis ---
    preictal_indices = np.where(y_full == 1)[0] #
    interictal_indices = np.where(y_full == 0)[0] #

    if len(preictal_indices) == 0 or len(interictal_indices) == 0:
        print("Dataset does not contain samples from both classes. Cannot create a balanced dataset or perform comparative analysis. Exiting.")
        return

    # Undersample the majority class to match the minority class
    n_minority = min(len(preictal_indices), len(interictal_indices))
    np.random.seed(42) # for reproducibility
    random_preictal_indices = np.random.choice(preictal_indices, size=n_minority, replace=False)
    random_interictal_indices = np.random.choice(interictal_indices, size=n_minority, replace=False) #

    balanced_indices = np.concatenate([random_preictal_indices, random_interictal_indices]) #
    np.random.shuffle(balanced_indices) #

    X_uint16 = X_uint16_full[balanced_indices] #
    y = y_full[balanced_indices] # Use the balanced labels 'y' from now on
    scales = scales_full[balanced_indices] #

    print(f"--- NOTE: Created a balanced dataset for analysis with {n_minority} preictal and {n_minority} interictal segments. ---") #

    # Invert scaling for the balanced dataset
    print("Inverting uint16 scaling for the balanced dataset...")
    X_float = invert_uint16_scaling(X_uint16, scales) #

    # 2. Get Feature Extractors
    features_to_extract, bde, feature_names = get_feature_extractors() #

    # 3. Extract All Features from the balanced data
    print(f"Extracting features for channel {channel_to_analyze} from {len(X_float)} balanced segments...")
    all_features = extract_all_features(X_float, channel_to_analyze, features_to_extract, bde, feature_names) #

    # 4. Analyze and Rank Features
    os.makedirs(output_dir, exist_ok=True) #
    analyze_and_rank_features(all_features, y, feature_names, output_dir) #

    print("\nAnalysis complete.") #


if __name__ == "__main__":
    parser = argparse.ArgumentParser( #
        description="Analyze and rank EEG features using balanced data. Calculates importance, p-values, Cohen's d, saves plots, and exports features to Excel.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument( #
        '--file_path',
        type=str,
        default='File/processed_segments_zscore_T_uint16.npz', # Adjust default as needed
        help='Path to the processed .npz file (containing X, y, scales).'
    )
    parser.add_argument( #
        '--channel',
        type=int,
        default=0,
        help="The index of the EEG channel to extract features from (default: 0)."
    )
    parser.add_argument( #
        '--output_dir',
        type=str,
        default='features_analysis/analysis_report',
        help="Directory to save the analysis report, plots, and Excel file (default: 'features_analysis/analysis_report')."
    )
    args = parser.parse_args() #

    # --- Add input validation ---
    if not os.path.exists(args.file_path):
         print(f"Error: Input file not found at '{args.file_path}'.")
         sys.exit(1)

    # Check channel index after loading data in main
    main(args.file_path, args.channel, args.output_dir) #
