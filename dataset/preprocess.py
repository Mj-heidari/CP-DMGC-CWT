import argparse
from utils import (
    add_seizure_annotations_bids,
    infer_preictal_interactal,
    extract_segments_with_labels_bids,
    scale_to_uint16,
    preprocess_chbmit,
)
import os
import mne
import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from typing import Optional, List
from collections import defaultdict


def build_suffix(
    segment_sec,
    preictal_factor=1,
    seizure_factor=1,
    ica_applied=False,
    filter_applied=False,
    downsample_applied=False,
    normalize_applied=False,
):
    parts = []

    # --- flags ---
    flags = ""
    if ica_applied:
        flags += "i"
    if filter_applied:
        flags += "f"
    if downsample_applied:
        flags += "d"
    if normalize_applied:
        flags += "n"
    if flags:
        parts.append(flags)

    # --- segment length ---
    if segment_sec is not None:
        parts.append(f"{segment_sec}s")

    # --- oversample factors only if >1 ---
    if seizure_factor not in (None, 1):
        parts.append(f"szx{seizure_factor}")
    if preictal_factor not in (None, 1):
        parts.append(f"prex{preictal_factor}")

    if not parts:
        return ""  # no suffix at all

    return "_" + "_".join(parts)


def process_chbmit_bids_dataset(
    dataset_dir: str,
    save_uint16: bool = False,
    apply_filter: bool = True,
    apply_ica: bool = True,
    apply_downsampling: bool = True,
    filter_type: str = "FIR",
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    sfreq_new: float = 128.0,
    downsample_method: str = "polyphase",
    normalize: Optional[str] = None,
    plot: bool = False,
    plot_psd: bool = False,
    show_statistics: bool = True,
    subj_nums: Optional[List[int]] = None,
    preictal_oversample_factor: int = 1,
    seizure_oversample_factor: int = 1,
    preictal_minutes: int = 15,
    post_buffer_minutes: int = 60,
    pre_buffer_minutes: int = 45,
    segment_sec: int = 5,
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
    apply_filter : bool, optional
        If True, apply bandpass filtering to the data.
        Default is True.
    apply_ica : bool, optional
        If True, apply ICA to remove artifacts.
        Default is True.
    apply_downsampling : bool, optional 
        If True, downsample the data to sfreq_new.
        Default is True.
    filter_type : str, optional
        Type of bandpass filter to use ('IIR' or 'FIR').
        Default is 'FIR'.
    l_freq : float, optional
        Low cutoff frequency for bandpass filter.
        Default is 0.5 Hz.
    h_freq : float, optional
        High cutoff frequency for bandpass filter.
        Default is 50.0 Hz.
    sfreq_new : float, optional
        New sampling frequency after downsampling.
        Default is 128.0 Hz.
    downsample_method : str, optional
        Method for downsampling ('polyphase', 'fft', 'resample').
        Default is 'polyphase'.
    normalize : Optional[str], optional
        Normalization method to apply ('zscore', 'robust', or None).
        Default is None (no normalization).
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
    preictal_oversample_factor : int, optional
        Oversampling factor for preictal segments.
        Default is 1 (no oversampling).
    seizure_oversample_factor : int, optional
        Oversampling factor for seizure segments.
        Default is 1 (no oversampling).
    preictal_minutes : int, optional
        Duration of preictal period in minutes.
        Default is 15 minutes.
    post_buffer_minutes : int, optional
        Postictal buffer duration in minutes.
        Default is 60 minutes.
    pre_buffer_minutes : int, optional 
        Interictal buffer duration in minutes.
        Default is 45 minutes.
    segment_sec : int, optional
        Length of each segment in seconds.
        Default is 5 seconds.

    Returns
    -------
    None
    """

    print("Processing CHB-MIT BIDS dataset...")
    print("Settings:")
    print(f"  Dataset dir: {dataset_dir}")
    print(f"  Save uint16: {save_uint16}")
    print(
        f"  Apply filter: {apply_filter} (type: {filter_type}, l_freq: {l_freq}, h_freq: {h_freq})"
    )
    print(f"  Apply ICA: {apply_ica}")
    print(
        f"  Apply downsampling: {apply_downsampling} (method: {downsample_method}, sfreq_new: {sfreq_new})"
    )
    print(f"  Normalize: {normalize}")
    print(f"  Plot raw data: {plot}")
    print(f"  Plot PSD: {plot_psd}")
    print(f"  Show statistics: {show_statistics}")
    print(f"  Subjects to process: {subj_nums if subj_nums is not None else 'All'}")
    print(f"  Preictal oversample factor: {preictal_oversample_factor}")
    print(f"  Seizure oversample factor: {seizure_oversample_factor}")

    sessions_pathes = glob.glob(os.path.join(dataset_dir, "*", "*"))
    print(f"Found {len(sessions_pathes)} sessions in dataset.")
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

            annotations = pd.read_csv(annotation_file_path, sep="\t")

            raw = add_seizure_annotations_bids(raw, annotations)

            raw = preprocess_chbmit(
                raw,
                apply_filter=apply_filter,
                filter_type=filter_type,
                l_freq=l_freq,
                h_freq=h_freq,
                apply_ica=apply_ica,
                apply_downsampling=apply_downsampling,
                downsample_method=downsample_method,
                sfreq_new=sfreq_new,
                normalize=normalize,
            )
            raws.append(raw)

        raw_all = mne.concatenate_raws(raws)
        raw_all = infer_preictal_interactal(
            raw_all,
            post_buffer_minutes=post_buffer_minutes,
            pre_buffer_minutes=pre_buffer_minutes,
            preictal_minutes=preictal_minutes,
        )

        if plot_psd:
            spectrum = raw_all.compute_psd()
            fig = spectrum.plot(average=True)
            fig.savefig(
                session_path + f"/eeg/psd_plot_{str(apply_filter)[0]}.png", dpi=300
            )
            plt.close(fig)

        # plot the annotation
        if plot:
            raw_all.plot(scalings="auto", duration=30)
            plt.show()

        X, y, meta_df, event_stats = extract_segments_with_labels_bids(
            raw_all,
            segment_sec=5,
            preictal_oversample_factor=preictal_oversample_factor,
            seizure_oversample_factor=seizure_oversample_factor,
        )

        if show_statistics:
            # --- Print statistics ---
            print("\n=== Extraction statistics ===")
            print(f"Total segments: {len(y)}")

            counts = Counter(y)
            for label, cnt in counts.items():
                print(f"  {label}: {cnt}")

            # ----- Updated: count by event_id or any grouping column in meta_df -----
            if "event_id" in meta_df.columns:
                group_col = "event_id"
            elif "group_id" in meta_df.columns:
                group_col = "group_id"
            else:
                # fallback: first column
                group_col = meta_df.columns[0]

            group_counts = meta_df[group_col].value_counts()

            print(f"Groups extracted: {len(group_counts)}")
            for gid, cnt in group_counts.items():
                print(f"  {gid}: {cnt} segments")
            print("=============================\n")

        # --- Save event stats to CSV ---
        stats_file = os.path.join(session_path, "eeg/event_stats.csv")
        with open(stats_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "event_id",
                    "label",
                    "onset_sec",
                    "duration_sec",
                    "n_segments",
                    "applied_overlap_sec",
                    "applied_factor",
                ],
            )
            writer.writeheader()
            writer.writerows(event_stats)

        print(f"Saved event stats to {stats_file}")

        suffix = build_suffix(
            segment_sec=segment_sec,
            preictal_factor=preictal_oversample_factor,
            seizure_factor=seizure_oversample_factor,
            ica_applied=apply_ica,
            filter_applied=apply_filter,
            downsample_applied=apply_downsampling,
            normalize_applied=(normalize is not None),
        )

        # =============== Saving arrays =================
        if save_uint16:
            X, scales = scale_to_uint16(X)
            np.savez_compressed(
                os.path.join(
                    session_path, f"eeg/processed_segments{suffix}_uint16.npz"
                ),
                X=X,
                y=y,
                meta_df=meta_df.to_dict("list"),
                scales=scales,
            )
        else:
            X = X.astype(np.float32)
            np.savez_compressed(
                os.path.join(session_path, f"eeg/processed_segments{suffix}_float.npz"),
                X=X,
                y=y,
                meta_df=meta_df.to_dict("list"),
            )

        # save options into a text file
        options_file = os.path.join(
            session_path, f"eeg/processing_options{suffix}.txt"
        )
        with open(options_file, "w") as f:
            f.write("Creation Time:\n")
            f.write(f"{time.ctime()}\n\n")
            f.write("Processing options:\n")
            f.write(f"save_uint16: {save_uint16}\n")
            f.write(f"apply_filter: {apply_filter}\n")
            f.write(f"filter_type: {filter_type}\n")
            f.write(f"l_freq: {l_freq}\n")
            f.write(f"h_freq: {h_freq}\n")
            f.write(f"apply_ica: {apply_ica}\n")
            f.write(f"apply_downsampling: {apply_downsampling}\n")
            f.write(f"downsample_method: {downsample_method}\n")
            f.write(f"sfreq_new: {sfreq_new}\n")
            f.write(f"normalize: {normalize}\n")
            f.write(f"segment_sec: {segment_sec}\n")
            f.write(f"preictal_oversample_factor: {preictal_oversample_factor}\n")
            f.write(f"seizure_oversample_factor: {seizure_oversample_factor}\n")
            f.write(f"preictal_minutes: {preictal_minutes}\n")
            f.write(f"post_buffer_minutes: {post_buffer_minutes}\n")
            f.write(f"pre_buffer_minutes: {pre_buffer_minutes}\n")
        print(f"Saved processed segments to {session_path}/eeg/processed_segments{suffix}_*.npz\n")


def build_subject_summary_from_event_stats(dataset_dir: str):
    """
    Build subject-level summary using per-session event_stats.csv files.
    Avoids rerunning the whole pipeline.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset root (contains chb01, chb02, ...).
    """
    subj_stats = defaultdict(
        lambda: {
            "n_preictal_events": 0,
            "n_interictal_events": 0,
            "n_preictal_segments": 0,
            "n_interictal_segments": 0,
        }
    )

    # find all event_stats.csv
    event_stat_files = glob.glob(
        os.path.join(dataset_dir, "*", "*", "eeg", "event_stats.csv")
    )

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
            fieldnames=[
                "subject_id",
                "n_preictal_events",
                "n_interictal_events",
                "n_preictal_segments",
                "n_interictal_segments",
            ],
        )
        writer.writeheader()
        for subj_id, stats in sorted(subj_stats.items()):
            writer.writerow({"subject_id": subj_id, **stats})

    print(f"Saved subject-level summary to {summary_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process CHB-MIT EEG dataset in BIDS format"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/BIDS_CHB-MIT",
        help="Path to CHB-MIT dataset directory",
    )

    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=None,
        help="List of subject numbers to preprocess (e.g., 1 2 3). If not specified, all subjects will be processed.",
    )

    parser.add_argument(
        "--save_uint16",
        action="store_true",
        help="Save EEG data as uint16 instead of float32",
    )

    parser.add_argument(
        "--apply_filter", action="store_true", help="Apply filtering to the data"
    )

    parser.add_argument(
        "--filter_type",
        type=str,
        default="FIR",
        help="Type of bandpass filter to use (IIR or FIR)",
    )

    parser.add_argument(
        "--apply_ica", action="store_true", help="Apply ICA to the data"
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        help="Normalization method (zscore, robust, none)",
    )

    parser.add_argument(
        "--sfreq_new",
        type=float,
        default=128.0,
        help="New sampling frequency after downsampling",
    )

    parser.add_argument(
        "--l_freq",
        type=float,
        default=0.5,
        help="Low cutoff frequency for bandpass filter",
    )

    parser.add_argument(
        "--h_freq",
        type=float,
        default=50.0,
        help="High cutoff frequency for bandpass filter",
    )

    parser.add_argument(
        "--apply_downsampling",
        action="store_true",
        help="Apply downsampling to the data",
    )

    parser.add_argument(
        "--downsample_method",
        type=str,
        default="polyphase",
        help="Method for downsampling (polyphase, fft, resample)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Plot raw data with annotations"
    )

    parser.add_argument(
        "--plot_psd", action="store_true", help="Plot power spectral density"
    )

    parser.add_argument(
        "--no_statistics",
        action="store_true",
        help="Disable printing extraction statistics",
    )

    parser.add_argument(
        "--build_summary",
        action="store_true",
        help="Build subject-level summary from event stats",
    )

    parser.add_argument(
        "--preictal_oversample_factor",
        type=int,
        default=1,
        help="Oversampling factor for preictal segments",
    )

    parser.add_argument(
        "--seizure_oversample_factor",
        type=int,
        default=1,
        help="Oversampling factor for seizure segments",
    )

    parser.add_argument(
        "--preictal_minutes",
        type=int,
        default=15,
        help="Duration of preictal period in minutes",
    )

    parser.add_argument(
        "--post_buffer_minutes",
        type=int,
        default=60,
        help="Postictal buffer duration in minutes",
    )

    parser.add_argument(
        "--pre_buffer_minutes",
        type=int,
        default=45,
        help="Interictal buffer duration in minutes",
    )

    parser.add_argument(
        "--segment_sec", type=int, default=5, help="Segment length in seconds"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert "none" to None for normalization

    process_chbmit_bids_dataset(
        dataset_dir=args.dataset_dir,
        save_uint16=args.save_uint16,
        apply_filter=args.apply_filter,
        apply_ica=args.apply_ica,
        filter_type=args.filter_type,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        sfreq_new=args.sfreq_new,
        downsample_method=args.downsample_method,
        apply_downsampling=args.apply_downsampling,
        normalize=None if args.normalize.lower() == "none" else args.normalize,
        plot=args.plot,
        plot_psd=args.plot_psd,
        show_statistics=not args.no_statistics,
        subj_nums=args.subjects,
        preictal_oversample_factor=args.preictal_oversample_factor,
        seizure_oversample_factor=args.seizure_oversample_factor,
        preictal_minutes=args.preictal_minutes,
        post_buffer_minutes=args.post_buffer_minutes,
        pre_buffer_minutes=args.pre_buffer_minutes,
        segment_sec=args.segment_sec,
    )

    if args.build_summary:
        build_subject_summary_from_event_stats(args.dataset_dir)
