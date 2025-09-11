from utils import *
import os
import mne
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter



def process_chbmit_bids_dataset(dataset_dir: str, output_dir: str, save_uint16: bool = True):
    """
    Process all subjects in CHB-MIT (BIDS format) dataset and save per-subject segments and labels.

    Parameters
    ----------
    dataset_dir : str
        Path to CHB-MIT dataset (contains chb01, chb02, ..., chb24 folders)
    output_dir : str
        Directory to save per-subject .npz files
    save_uint16 : bool, optional
        If True, saves EEG data scaled to uint16 with per-sample min/max for reconstruction.
        Default is False (save as float32).
    """

    os.makedirs(output_dir, exist_ok=True)
    sessions_pathes = glob.glob("./data/BIDS_CHB-MIT/*/*")
    for session_path in sorted(sessions_pathes):
        edf_files = sorted(glob.glob(session_path + "/eeg/*.edf"))
        raws = []
        for raw_file_path in edf_files:
            annotation_file_path = raw_file_path.replace("_eeg.edf", "_events.tsv")

            raw = mne.io.read_raw_edf(raw_file_path, preload=True)
            annotations = pd.read_csv(annotation_file_path, sep="\t")

            raw = add_seizure_annotations_bids(raw, annotations)

            raw = preprocess_chbmit(raw, only_resample = True)

            raws.append(raw)

        raw_all = mne.concatenate_raws(raws)
        raw_all = infer_preictal_interactal(raw_all)

        # plot the annotation
        # raw_all.plot(scalings="auto", duration=30)
        # plt.show()

        X, y, group_ids = extract_segments_with_labels_bids(
            raw_all, segment_sec=5, overlap=0.0, keep_labels={"preictal", "interictal"}
        )

        # --- Print statistics ---
        print("\n=== Extraction statistics ===")
        print(f"Total segments: {len(y)}")
        counts = Counter(y)
        for label, cnt in counts.items():
            print(f"  {label}: {cnt}")
        group_counts = Counter(group_ids)
        print(f"Groups extracted: {len(group_counts)}")
        for gid, cnt in group_counts.items():
            print(f"  {gid}: {cnt} segments")
        print("=============================\n")

        # Convert to float32 before saving
        X = X.astype(np.float32)

        if save_uint16:
            X, scales = scale_to_uint16(X)
            np.savez_compressed(
                session_path + "/eeg/processed_segments_uint16.npz",
                X=X,
                y=y,
                group_ids=group_ids,
                scales=scales,
            )
        else:
            X = X.astype(np.float32)
            np.savez_compressed(
                session_path + "/eeg/processed_segments.npz",
                X=X,
                y=y,
                group_ids=group_ids,
            )


        break


if __name__ == "__main__":
    dataset_dir = "data\BIDS_CHB-MIT"
    output_dir = "processed"
    process_chbmit_bids_dataset(dataset_dir, output_dir)
