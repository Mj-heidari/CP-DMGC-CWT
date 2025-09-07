from utils import (
    preprocess_chbmit,
    add_seizure_annotations,
    extract_segments_with_labels,
    convert_to_preactal_interactal,
    add_seizure_annotations_bids,
    infer_preictal_interactal,
)
import os
import mne
import numpy as np
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt


def process_chbmit_dataset(dataset_dir: str, output_dir: str):
    """
    Process all subjects in CHB-MIT dataset and save per-subject segments and labels.

    Parameters
    ----------
    dataset_dir : str
        Path to CHB-MIT dataset (contains chb01, chb02, ..., chb24 folders)
    output_dir : str
        Directory to save per-subject .npz files
    """
    os.makedirs(output_dir, exist_ok=True)

    for subj in sorted(os.listdir(dataset_dir)):
        subj_path = os.path.join(dataset_dir, subj)
        if not os.path.isdir(subj_path):
            continue

        print(f"Processing {subj}...")
        summary_file = os.path.join(subj_path, f"{subj}-summary.txt")
        if not os.path.exists(summary_file):
            print(f"  Summary file not found, skipping {subj}")
            continue

        all_X, all_y = [], []

        # Iterate over all EDF files
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith(".edf"):
                continue

            edf_path = os.path.join(subj_path, fname)
            print(f"  Loading {fname}...")
            raw = mne.io.read_raw_edf(edf_path, preload=True)

            # Add seizure annotations
            raw = add_seizure_annotations(raw, summary_file)

            # Preprocess (without robust normalization, returns Raw object)
            raw_proc = preprocess_chbmit(raw)

            # Extract 5s segments and labels
            X, y = extract_segments_with_labels(
                raw_proc, segment_sec=5.0, seizure_threshold=0.6
            )

            all_X.append(X)
            all_y.append(y)

        if len(all_X) == 0:
            print(f"  No EDF files processed for {subj}")
            continue

        # Concatenate all files for the subject
        subj_X = np.concatenate(all_X, axis=0)
        subj_y = np.concatenate(all_y, axis=0)

        subj_X, subj_y = convert_to_preactal_interactal(subj_X, subj_y)

        # Save per-subject
        out_file = os.path.join(output_dir, f"{subj}.npz")
        np.savez_compressed(out_file, X=subj_X, y=subj_y)
        print(f"  Saved {subj_X.shape[0]} segments to {out_file}\n")


def process_chbmit_bids_dataset(dataset_dir: str, output_dir: str):
    """
    Process all subjects in CHB-MIT (BIDS format) dataset and save per-subject segments and labels.

    Parameters
    ----------
    dataset_dir : str
        Path to CHB-MIT dataset (contains chb01, chb02, ..., chb24 folders)
    output_dir : str
        Directory to save per-subject .npz files
    """

    os.makedirs(output_dir, exist_ok=True)

    for subj in sorted(os.listdir(dataset_dir))[1:]:
        subj_path = os.path.join(dataset_dir, subj)
        if not os.path.isdir(subj_path):
            continue
        edf_files = sorted(glob.glob(subj_path + "/ses-01/eeg/*.edf"))
        raws = []
        for i, raw_file_path in enumerate(edf_files):
            annotation_file_path = raw_file_path.replace("_eeg.edf", "_events.tsv")
            metadata_file_path = raw_file_path.replace("_eeg.edf", "_eeg.json")

            raw = mne.io.read_raw_edf(raw_file_path, preload=True)
            annotations = pd.read_csv(annotation_file_path, sep="\t")
            metadata = json.load(open(metadata_file_path))

            raw = add_seizure_annotations_bids(raw, annotations)

            # raw = preprocess_chbmit(raw)

            raws.append(raw)
            if i == 25:
                break

        raw_all = mne.concatenate_raws(raws)
        raw_all = infer_preictal_interactal(raw_all)
        raw_all.plot(scalings="auto", duration=30)
        plt.show()
        break


if __name__ == "__main__":
    dataset_dir = "data\BIDS_CHB-MIT"
    output_dir = "processed"
    process_chbmit_bids_dataset(dataset_dir, output_dir)
