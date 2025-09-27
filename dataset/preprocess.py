from utils import (
    add_seizure_annotations_bids,
    preprocess_chbmit,
    infer_preictal_interactal,
    extract_segments_with_labels_bids,
    scale_to_uint16,
)
import os
import mne
import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, List
from collections import defaultdict


def process_chbmit_bids_dataset(
    dataset_dir: str,
    save_uint16: bool = False,
    normalization_method: Optional[str] = "zscore",
    apply_ica: bool = True,
    apply_filter: bool = True,
    plot: bool = False,
    plot_psd: bool = False,
    show_statistics: bool = True,
    subj_nums: Optional[List[int]] = None,
):
    """
    Process all subjects in CHB-MIT (BIDS format) dataset and save per-subject segments and labels.

    Parameters
    ----------
    dataset_dir : str
        Path to CHB-MIT dataset (contains chb01, chb02, ..., chb24 folders)
    save_uint16 : bool, optional
        If True, saves EEG data scaled to uint16 with per-sample min/max for reconstruction.
        Default is False (save as float32).
    normalization_method: {"zscore", "robust", None}
        Normalization method to apply after resampling. Default is "zscore".
    apply_ica: bool
        If true, the ica will be applied, default = True.
    plot : bool, optional
        If True, plot raw data with annotations before segmentation.
        Default is False.
    show_statistics : bool, optional
        If True, print extraction statistics (segments, labels, groups).
        Default is True.
    subj_nums: Optional[list[int]]
        A list containing the subject numbers that should be preprocessed.
        If None, the all subjects will be preprocessed.
        Default is None.
    """

    sessions_pathes = glob.glob(os.path.join(dataset_dir, "*", "*"))
    for session_path in sorted(sessions_pathes):
        print(session_path)
        subj_id = session_path.split("\\")[-2].split("-")[-1]
        if subj_nums is not None and (int(subj_id) not in subj_nums):
            print("skipping subject id:", subj_id)
            continue
        edf_files = sorted(glob.glob(session_path + "/eeg/*.edf"))
        raws = []
        for raw_file_path in edf_files:
            annotation_file_path = raw_file_path.replace("_eeg.edf", "_events.tsv")

            raw = mne.io.read_raw_edf(raw_file_path, preload=True)
            print("channels names:", raw.ch_names)
            annotations = pd.read_csv(annotation_file_path, sep="\t")

            raw = add_seizure_annotations_bids(raw, annotations)

            raw = preprocess_chbmit(
                raw, apply_ica=apply_ica, apply_filter=apply_filter, normalize=normalization_method
            )
            raws.append(raw)

        raw_all = mne.concatenate_raws(raws)
        raw_all = infer_preictal_interactal(raw_all)

        if plot_psd:
            spectrum = raw_all.compute_psd()
            fig = spectrum.plot(average=True)
            fig.savefig(session_path + f"/eeg/psd_plot_{str(normalization_method)}_{str(apply_ica)[0]}_{str(apply_filter)[0]}.png", dpi=300)
            plt.close(fig)

        # plot the annotation
        if plot:
            raw_all.plot(scalings="auto", duration=30)
            plt.show()

        X, y, group_ids, event_stats = extract_segments_with_labels_bids(
            raw_all, segment_sec=5, overlap=0.0, keep_labels={"preictal", "interictal"}
        )

        if show_statistics:
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

        # --- Save event stats to CSV ---
        stats_file = os.path.join(session_path, "eeg/event_stats.csv")
        with open(stats_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_id", "label", "onset_sec", "duration_sec", "n_segments"])
            writer.writeheader()
            writer.writerows(event_stats)
        print(f"Saved event stats to {stats_file}")

        if save_uint16:
            X, scales = scale_to_uint16(X)
            np.savez_compressed(
                session_path
                + f"/eeg/processed_segments_{str(normalization_method)}_{str(apply_ica)[0]}_{str(apply_filter)[0]}_uint16.npz",
                X=X,
                y=y,
                group_ids=group_ids,
                scales=scales,
            )
        else:
            X = X.astype(np.float32)
            np.savez_compressed(
                session_path
                + f"/eeg/processed_segments_{str(normalization_method)}_{str(apply_ica)[0]}_{str(apply_filter)[0]}_float.npz",
                X=X,
                y=y,
                group_ids=group_ids,
            )


def build_subject_summary_from_event_stats(dataset_dir: str):
    """
    Build subject-level summary using per-session event_stats.csv files.
    Avoids rerunning the whole pipeline.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset root (contains chb01, chb02, ...).
    """
    subj_stats = defaultdict(lambda: {"n_preictal_events": 0,
                                      "n_interictal_events": 0,
                                      "n_preictal_segments": 0,
                                      "n_interictal_segments": 0})

    # find all event_stats.csv
    event_stat_files = glob.glob(os.path.join(dataset_dir, "*", "*", "eeg", "event_stats.csv"))

    if not event_stat_files:
        print("No event_stats.csv files found. Run the pipeline first.")
        return

    for stats_file in event_stat_files:
        subj_id = stats_file.split(os.sep)[-4].split("-")[-1]  # e.g., chb01 -> "01"
        print(subj_id)
        with open(stats_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"]
                n_segments = int(row["n_segments"])

                if label == "preictal":
                    subj_stats[subj_id]["n_preictal_events"] += 1
                    subj_stats[subj_id]["n_preictal_segments"] += n_segments
                elif label == "interictal":
                    subj_stats[subj_id]["n_interictal_events"] += 1
                    subj_stats[subj_id]["n_interictal_segments"] += n_segments

    # save subject-level summary
    summary_file = os.path.join(dataset_dir, "subject_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "n_preictal_events", "n_interictal_events",
                        "n_preictal_segments", "n_interictal_segments"]
        )
        writer.writeheader()
        for subj_id, stats in sorted(subj_stats.items()):
            writer.writerow({"subject_id": subj_id, **stats})

    print(f"Saved subject-level summary to {summary_file}")

if __name__ == "__main__":
    dataset_dir = "data/BIDS_CHB-MIT"
    subjects_to_be_preprocessed = [13,14,15,16,17,18,19,20,21,22,23,24]
    # process_chbmit_bids_dataset(
    #     dataset_dir,
    #     save_uint16=True,
    #     normalization_method="zscore",
    #     apply_ica=False,
    #     apply_filter=True,
    #     subj_nums=subjects_to_be_preprocessed,
    #     plot_psd=True
    # )

    build_subject_summary_from_event_stats("data/BIDS_CHB-MIT")
