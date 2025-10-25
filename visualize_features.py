import numpy as np
import mne
import argparse
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt

# --- Project Setup ---
# Dynamically modifies the Python path to ensure project modules can be imported.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add transforms/feature directory to path for base import
feature_transform_path = os.path.join(project_root, 'transforms', 'feature')
if feature_transform_path not in sys.path:
    sys.path.insert(0, feature_transform_path)
# Add dataset directory to path for utils import
dataset_path = os.path.join(project_root, 'dataset')
if dataset_path not in sys.path:
     sys.path.insert(0, dataset_path)


# --- Imports from your project structure ---
from transforms.feature.base import FeatureTransform #
from transforms.feature.basic_stats import ( #
    MeanAmplitude, StandardDeviation, Skewness, Kurtosis,
    RootMeanSquare, LineLength, ZeroCrossingRate,
    HjorthActivity, HjorthMobility, HjorthComplexity,
)
from transforms.feature.band_power import ( #
    BandPowerTransform, DeltaPower, ThetaPower, AlphaPower, BetaPower, GammaPower,
)
from transforms.feature.spectral_summary import ( #
    SpectralEntropy, IntensityWeightedMeanFrequency,
    SpectralEdgeFrequency, PeakFrequency,
)
from transforms.feature.connectivity import MeanAbsCorrelation #
from transforms.feature.differential_entropy import BandDifferentialEntropy #
from dataset.utils import invert_uint16_scaling #

# --- Custom Features ---
class LowAlphaPower(BandPowerTransform): #
    """Custom feature for power in the low alpha band."""
    def __init__(self, sampling_rate=128): super().__init__(8, 10, sampling_rate)

class HighAlphaPower(BandPowerTransform): #
    """Custom feature for power in the high alpha band."""
    def __init__(self, sampling_rate=128): super().__init__(10, 13, sampling_rate)

class LowBetaPower(BandPowerTransform): #
    """Custom feature for power in the low beta band."""
    def __init__(self, sampling_rate=128): super().__init__(13, 20, sampling_rate)

class HighBetaPower(BandPowerTransform): #
    """Custom feature for power in the high beta band."""
    def __init__(self, sampling_rate=128): super().__init__(20, 30, sampling_rate)

class AlphaBetaRatio(FeatureTransform): #
    """Custom feature for the ratio of alpha to beta band power."""
    def __init__(self, sampling_rate=128):
        self.alpha_power = AlphaPower(sampling_rate)
        self.beta_power = BetaPower(sampling_rate)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray: #
        alpha = self.alpha_power.apply(eeg, **kwargs) #
        beta = self.beta_power.apply(eeg, **kwargs) #
        ratio = alpha / (beta + 1e-8) #
        return ratio #


def extract_features_for_range(X_data, channel, features_to_extract, bde, feature_names):
    """
    Extracts features only for a specified range of segments.
    """
    num_chunk_segments = X_data.shape[0] #
    num_features = len(feature_names) #
    extracted_features = np.zeros((num_chunk_segments, num_features)) #

    for i in tqdm(range(num_chunk_segments), desc="Extracting Features", leave=False, bar_format='{l_bar}{bar:10}{r_bar}'): #
        segment = X_data[i] #
        feature_vector = [] #

        # Extract non-BDE features
        for feature in features_to_extract[:-1]: #
            # Assumes feature.apply returns shape (n_channels, 1) or similar
            result = feature.apply(segment)[channel] #
            # Ensure result is a scalar
            feature_vector.append(result.item() if hasattr(result, 'item') else result) #

        # Extract BDE features
        bde_values = bde.apply(eeg=segment) # # Assumes shape (n_channels, n_bands)
        feature_vector.extend(bde_values[channel]) #

        extracted_features[i, :] = np.array(feature_vector) #
    return extracted_features #

def find_interictal_to_preictal_transitions(y):
    """Finds indices where label changes from 0 (interictal) to 1 (preictal)."""
    transitions = np.where((y[:-1] == 0) & (y[1:] == 1))[0] + 1 #
    return transitions #

def find_preictal_to_interictal_transitions(y):
    """Finds indices where label changes from 1 (preictal) to 0 (interictal)."""
    transitions = np.where((y[:-1] == 1) & (y[1:] == 0))[0] + 1 #
    return transitions #

def find_preictal_ranges(y):
    """Finds start and end indices of all continuous preictal blocks."""
    preictal_indices = np.where(y == 1)[0] #
    if not len(preictal_indices): #
        return [] #

    ranges = [] #
    start = preictal_indices[0] #
    # Check for gaps in consecutive preictal indices
    for i in range(1, len(preictal_indices)): #
        if preictal_indices[i] > preictal_indices[i-1] + 1: #
            ranges.append((start, preictal_indices[i-1] + 1)) # End index is exclusive #
            start = preictal_indices[i] #
    ranges.append((start, preictal_indices[-1] + 1)) # Add the last range #
    return ranges #

def process_and_save_plot(start, stop, X_data, y_data, scales_data, channel, features_to_extract, bde, feature_names, output_dir):
    """
    Handles feature extraction, plotting EEG signal and feature trends, and saving for a specific window.
    """
    if start >= stop:
        print(f"Skipping invalid range: start={start}, stop={stop}")
        return

    print(f"\nProcessing segments {start} to {stop-1}...")
    # Invert scaling and extract features only for the requested chunk
    X_chunk_uint16 = X_data[start:stop]
    scales_chunk = scales_data[start:stop]
    if X_chunk_uint16.size == 0 or scales_chunk.size == 0:
        print(f"Warning: Empty data or scales for segments {start}-{stop-1}. Skipping plot.")
        return

    # Use invert_uint16_scaling from dataset/utils.py
    X_chunk_float = invert_uint16_scaling(X_chunk_uint16, scales_chunk) #

    # Use extract_features_for_range from this script
    chunk_features = extract_features_for_range(
        X_chunk_float, channel, features_to_extract, bde, feature_names
    )
    if chunk_features.size == 0:
        print(f"Warning: No features extracted for segments {start}-{stop-1}. Skipping plot.")
        return

    # --- Create MNE Raw object for signal plot ---
    X_to_plot = np.concatenate(X_chunk_float, axis=1) # Shape: (n_channels, n_segments * segment_samples)
    sfreq = 128 # Assuming 128 Hz sampling rate after preprocessing
    ch_names_signal = [f'Ch {i}' for i in range(X_data.shape[1])] # Generic channel names
    info_signal = mne.create_info(ch_names=ch_names_signal, sfreq=sfreq, ch_types='eeg')
    raw_signal = mne.io.RawArray(X_to_plot, info_signal, verbose=False)

    # Add annotations based on segment labels
    window_y = y_data[start:stop]
    segment_duration_sec = 5.0 # Assuming 5-second segments
    onsets = [i * segment_duration_sec for i in range(len(window_y))]
    durations = [segment_duration_sec] * len(window_y)
    descriptions = ['preictal' if label == 1 else 'interictal' for label in window_y]
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw_signal.set_annotations(annotations)

    # Generate signal plot without showing it directly
    try:
        # Plot all channels available in the data
        fig_signal = raw_signal.plot(duration=len(window_y) * segment_duration_sec, scalings='auto',
                                     show=False, n_channels=X_data.shape[1], remove_dc=False,
                                     title=f'EEG Signal (Segments {start}-{stop-1})')
        if fig_signal:
            fig_signal.canvas.draw() # Ensure figure is drawn to render the content
        else:
            print("Warning: MNE plot failed to generate.")
            plt.close('all') # Close any potentially stuck figures
            return # Skip this plot if MNE fails
    except Exception as e:
        print(f"Error during MNE plotting: {e}. Skipping this plot.")
        plt.close('all') # Close figures
        return # Skip if MNE plot fails


    # --- Create combined plot ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 2]}) # Adjusted ratios

    # --- Draw MNE signal plot onto top subplot ---
    try:
        width, height = fig_signal.canvas.get_width_height()
        if width <= 0 or height <= 0:
             raise ValueError("MNE figure canvas has invalid dimensions.")
        # Render the MNE plot to an image buffer
        img_signal_argb = np.frombuffer(fig_signal.canvas.tostring_argb(), dtype=np.uint8)
        # Reshape and remove alpha channel if needed (ARGB -> RGB)
        img_signal = img_signal_argb.reshape(height, width, 4)[:, :, 1:4] # Keep RGB

        axes[0].imshow(img_signal)
        # The title is now set during raw_signal.plot()
        # axes[0].set_title(f'EEG Signal (Segments {start}-{stop-1}) with Annotations')
        axes[0].axis('off')
        plt.close(fig_signal) # Close original MNE figure to free memory
    except Exception as e:
         print(f"Error rendering MNE plot to image: {e}")
         axes[0].set_title(f'EEG Signal (Segments {start}-{stop-1}) - Plotting Error')
         axes[0].text(0.5, 0.5, 'Error displaying MNE plot', horizontalalignment='center', verticalalignment='center')
         if 'fig_signal' in locals() and fig_signal is not None:
             plt.close(fig_signal) # Ensure it's closed even on error

    # --- Manually draw feature plot on bottom subplot ---
    ax_features = axes[1]
    num_features = chunk_features.shape[1]
    num_segments_in_chunk = chunk_features.shape[0]

    # Normalize each feature across the chunk (0-1 range) for visualization
    normalized_features = np.zeros_like(chunk_features, dtype=float) # Use float for potential NaNs
    feature_means = np.zeros(num_features)
    feature_stds = np.zeros(num_features)
    for i in range(num_features):
        col = chunk_features[:, i]
        valid_col = col[np.isfinite(col)]
        if valid_col.size == 0:
             min_val, max_val = 0, 1
             feature_means[i], feature_stds[i] = np.nan, np.nan
             normalized_features[:, i] = np.nan # Set whole column to NaN if no valid data
        else:
             min_val, max_val = valid_col.min(), valid_col.max()
             feature_means[i], feature_stds[i] = valid_col.mean(), valid_col.std()
             # Apply normalization, keeping non-finite values as NaN
             range_val = max_val - min_val
             if range_val > 1e-8:
                 normalized_features[:, i] = np.where(np.isfinite(col), (col - min_val) / range_val, np.nan)
             else:
                 normalized_features[:, i] = np.where(np.isfinite(col), 0.5, np.nan) # Center if range is zero


    # Time axis representing the midpoint of each segment
    time_axis_segments = (np.arange(num_segments_in_chunk) + 0.5) * segment_duration_sec
    total_time_chunk = num_segments_in_chunk * segment_duration_sec

    # Stacking offset
    offsets = np.arange(num_features) * 1.2 # Vertical spacing

    # Plot each normalized feature
    colors = plt.cm.get_cmap('tab20', num_features)
    for i in range(num_features):
        # Only plot where data is finite
        mask = np.isfinite(normalized_features[:, i])
        if np.any(mask): # Only plot if there's valid data
             ax_features.plot(time_axis_segments[mask], normalized_features[mask, i] + offsets[i],
                              label=f"{feature_names[i]}", # Simple label
                              linewidth=1.0, marker='.', markersize=3, color=colors(i % 20)) # Cycle colors

    # Add vertical background shading for labels
    y_min_plot, y_max_plot = -0.5, offsets.max() + 1.5
    ax_features.set_ylim(y_min_plot, y_max_plot)
    for k in range(num_segments_in_chunk):
        time_start = k * segment_duration_sec
        color = 'lightcoral' if window_y[k] == 1 else 'lightgreen'
        ax_features.axvspan(time_start, time_start + segment_duration_sec, facecolor=color, alpha=0.2, zorder=-10) # Lighter alpha

    # Configure feature plot axes
    ax_features.set_yticks(offsets + 0.5)
    ax_features.set_yticklabels(feature_names, fontsize=8) # Use feature names as labels
    ax_features.tick_params(axis='y', length=0) # Hide y-axis ticks
    ax_features.set_xlim(0, total_time_chunk)
    ax_features.set_xlabel("Time (s)")
    ax_features.set_title(f'Normalized Features (Channel {channel}, Segments {start}-{stop-1})')
    ax_features.grid(True, axis='x', linestyle=':', alpha=0.5)

    # Add legend outside the plot if too many features
    # handles, labels = ax_features.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', fontsize='x-small', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Add some bottom margin, adjust right if legend is outside

    # --- Create descriptive filename ---
    unique_labels_in_window = sorted(np.unique(window_y))
    label_parts = []
    if 0 in unique_labels_in_window: label_parts.append("interictal")
    if 1 in unique_labels_in_window: label_parts.append("preictal")
    label_str = "_".join(label_parts) if label_parts else "nolabels"

    filename = f'segments_{start}_to_{stop-1}_(ch{channel}_{label_str}).png'
    save_path = os.path.join(output_dir, filename)

    try:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")

    plt.close(fig) # Close the combined figure


def main(file_path, channel_to_analyze, output_dir):
    """
    Main function to load data, find key events, extract features, and save plots.
    """
    # 1. Load Data
    try:
        # Assumes file contains 'X', 'y', 'scales'
        data = np.load(file_path)
        X_data = data['X'] # uint16 data
        y_str = data['y'] # String labels 'preictal', 'interictal'
        scales_data = data['scales'] # Scaling factors
        print(f"Data loaded successfully from '{file_path}'.")
        print(f"Shape of EEG data (segments, channels, samples): {X_data.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except KeyError as e:
         print(f"Error: Missing key '{e}' in the .npz file. Expected 'X', 'y', 'scales'.")
         return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # Convert string labels to integer labels (0 for interictal, 1 for preictal)
    y_data = np.array([1 if label == 'preictal' else 0 for label in y_str]) #

    # --- Diagnostic Print ---
    print("\n--- Full Dataset Label Counts ---")
    unique_labels, counts = np.unique(y_data, return_counts=True)
    label_map = {0: 'interictal', 1: 'preictal'}
    if not len(unique_labels):
        print("No labels found in the data.")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label} ({label_map.get(label, 'Unknown')}): {count} segments")
    print("-------------------------------------\n")


    # 2. Instantiate Feature Extractors
    # Make sure BandDifferentialEntropy is the last one for indexing logic in extract_features_for_range
    features_to_extract = [ #
        MeanAmplitude(), StandardDeviation(), Skewness(), Kurtosis(),
        RootMeanSquare(), LineLength(), ZeroCrossingRate(),
        HjorthActivity(), HjorthMobility(), HjorthComplexity(),
        DeltaPower(), ThetaPower(), AlphaPower(), BetaPower(), GammaPower(),
        LowAlphaPower(), HighAlphaPower(), LowBetaPower(), HighBetaPower(),
        AlphaBetaRatio(),
        SpectralEntropy(), IntensityWeightedMeanFrequency(),
        SpectralEdgeFrequency(), PeakFrequency(), MeanAbsCorrelation(),
        BandDifferentialEntropy() # This should be last
    ]
    bde = BandDifferentialEntropy() #
    # Generate feature names list including BDE bands
    feature_names = [f.__class__.__name__ for f in features_to_extract[:-1]] #
    # Add BDE band names
    for band in bde.band_dict.keys(): #
        feature_names.append(f"BDE_{band}") #

    num_total_segments = X_data.shape[0]

    os.makedirs(output_dir, exist_ok=True) #

    # Find all key event indices
    i_to_p_transitions = find_interictal_to_preictal_transitions(y_data) #
    p_to_i_transitions = find_preictal_to_interictal_transitions(y_data) #
    preictal_ranges = find_preictal_ranges(y_data) #

    has_preictal_events = i_to_p_transitions.size > 0 or p_to_i_transitions.size > 0 or preictal_ranges # Check if any events exist

    if not has_preictal_events:
        print("\nNo preictal events or transitions found in the data. Cannot generate plots around events.")
        # Optionally, plot the first few segments anyway
        print("Plotting first 10 segments as an example.")
        first_segments_dir = os.path.join(output_dir, "first_segments")
        os.makedirs(first_segments_dir, exist_ok=True)
        process_and_save_plot(0, min(10, num_total_segments), X_data, y_data, scales_data, channel_to_analyze, features_to_extract, bde, feature_names, first_segments_dir)
        return

    # --- Interactive Prompt ---
    total_preictal_chunks = 0
    if preictal_ranges:
        for r_start, r_end in preictal_ranges:
            total_preictal_chunks += (r_end - r_start + 9) // 10 # Ceiling division

    print("\n--- Summary of Found Event Ranges ---")
    print(f"Found {len(i_to_p_transitions)} [Interictal -> Preictal] transitions.")
    print(f"Found {len(p_to_i_transitions)} [Preictal -> Interictal] transitions.")
    print(f"Found {len(preictal_ranges)} continuous [Preictal Periods] (total {total_preictal_chunks} 10-segment plots).")
    print("-------------------------------------\n")

    limit_ranges = -1
    while limit_ranges < 0:
        try:
            user_input = input("Enter the maximum number of plots to save for EACH category (transitions, full periods) (enter 0 to save all): ") #
            limit_ranges = int(user_input) #
            if limit_ranges < 0: #
                print("Please enter a non-negative number.")
        except ValueError: #
            print("Invalid input. Please enter an integer.")

    # Process Interictal -> Preictal transitions
    num_to_process = len(i_to_p_transitions) if limit_ranges == 0 else min(len(i_to_p_transitions), limit_ranges) #
    if num_to_process > 0:
        subdir = os.path.join(output_dir, "interictal_to_preictal") #
        os.makedirs(subdir, exist_ok=True) #
        print(f"\nProcessing {num_to_process} of {len(i_to_p_transitions)} [Interictal -> Preictal] transitions...")
        for t_idx in i_to_p_transitions[:num_to_process]: #
            # Create a window of 10 segments centered around the transition index 't_idx'
            start = max(0, t_idx - 5) #
            stop = min(num_total_segments, start + 10) # Ensure window size is 10 if possible #
            # Adjust start if stop hit the boundary early
            start = max(0, stop - 10)
            process_and_save_plot(start, stop, X_data, y_data, scales_data, channel_to_analyze, features_to_extract, bde, feature_names, subdir) #

    # Process Preictal -> Interictal transitions
    num_to_process = len(p_to_i_transitions) if limit_ranges == 0 else min(len(p_to_i_transitions), limit_ranges) #
    if num_to_process > 0:
        subdir = os.path.join(output_dir, "preictal_to_interictal") #
        os.makedirs(subdir, exist_ok=True) #
        print(f"\nProcessing {num_to_process} of {len(p_to_i_transitions)} [Preictal -> Interictal] transitions...")
        for t_idx in p_to_i_transitions[:num_to_process]: #
            start = max(0, t_idx - 5) #
            stop = min(num_total_segments, start + 10) #
            start = max(0, stop - 10)
            process_and_save_plot(start, stop, X_data, y_data, scales_data, channel_to_analyze, features_to_extract, bde, feature_names, subdir) #

    # Process full preictal ranges (in chunks of 10)
    num_ranges_to_process = len(preictal_ranges) if limit_ranges == 0 else min(len(preictal_ranges), limit_ranges) # Limit number of *ranges*
    processed_plot_count = 0
    plots_limit = total_preictal_chunks if limit_ranges == 0 else limit_ranges # Limit number of *plots* from this category
    if num_ranges_to_process > 0:
        subdir = os.path.join(output_dir, "full_preictal_periods") #
        os.makedirs(subdir, exist_ok=True) #
        print(f"\nProcessing {num_ranges_to_process} of {len(preictal_ranges)} [Full Preictal Periods] (max {plots_limit} plots)...")
        for r_start, r_end in preictal_ranges[:num_ranges_to_process]: #
             if processed_plot_count >= plots_limit and limit_ranges != 0: break # Stop if plot limit reached
             # Iterate through the range in chunks of 10 segments
             for i in range(r_start, r_end, 10): #
                 if processed_plot_count >= plots_limit and limit_ranges != 0: break # Stop if plot limit reached
                 start = i #
                 stop = min(r_end, i + 10) #
                 process_and_save_plot(start, stop, X_data, y_data, scales_data, channel_to_analyze, features_to_extract, bde, feature_names, subdir) #
                 processed_plot_count += 1


    print("\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( #
        description="Automatically find key EEG events (transitions, preictal periods), extract features for a specific channel, plot signal and features over time, and save plots.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument( #
        '--file_path',
        type=str,
        default='File/processed_segments_zscore_T_uint16.npz', # Example path, adjust as needed
        help='Path to the processed .npz file containing keys: X, y, scales.'
    )
    parser.add_argument( #
        '--channel',
        type=int,
        default=0,
        help="The index of the EEG channel to analyze features for (default: 0)."
    )
    parser.add_argument( #
        '--output_dir',
        type=str,
        default='output_feature_plots', # Changed default output dir name
        help="Directory to save the output PNG plot files (default: 'output_feature_plots')."
    )
    args = parser.parse_args() #

    # Add error checking for channel index
    try:
        data = np.load(args.file_path)
        num_channels = data['X'].shape[1]
        if not (0 <= args.channel < num_channels):
             print(f"Error: Channel index {args.channel} is out of bounds (0-{num_channels-1}).")
             sys.exit(1) # Exit if channel index is invalid
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.file_path}'. Cannot check channel bounds.")
        # Proceed, but main function will handle the FileNotFoundError
    except KeyError:
         print(f"Error: Input file '{args.file_path}' does not contain key 'X'. Cannot check channel bounds.")
         # Proceed, but main function will handle the KeyError
    except Exception as e:
         print(f"An error occurred reading the input file to check channel bounds: {e}")
         # Proceed, main function might handle it

    main(args.file_path, args.channel, args.output_dir) #