import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Tuple, List
import re
import pandas as pd


def preprocess_chbmit(
    raw: mne.io.Raw, sfreq_new: int = 128, l_freq: float = 0.5, h_freq: float = 40.0
) -> Tuple[np.ndarray, List[str], int]:
    """
    Preprocessing pipeline for CHB-MIT EEG dataset.

    Steps:
      - Bandpass filter
      - Notch filter (60 Hz harmonics)
      - ICA (blink artifact removal via frontal proxies)
      - Downsampling
      - Robust normalization (median + IQR)

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data from CHB-MIT (23 bipolar channels, 256 Hz).
    sfreq_new : int
        New sampling frequency, default = 128 Hz.
    l_freq : float
        Lower frequency bound for bandpass filter, default = 0.5 Hz.
    h_freq : float
        Upper frequency bound for bandpass filter, default = 40 Hz.

    Returns
    -------
    data : np.ndarray
        Preprocessed EEG (channels × time).
    ch_names : List[str]
        Channel names after preprocessing.
    components_removed : int
        Number of ICA components removed.
    """
    components_removed = 0
    try:
        raw_proc = raw.copy().pick_types(eeg=True)

        # 1. Bandpass filter
        raw_proc.filter(l_freq, h_freq, fir_design="firwin", phase="zero-double")

        # # 2. Notch filter (60 Hz + harmonics)
        # raw_proc.notch_filter(np.arange(60, h_freq, 60), fir_design="firwin")

        # 3. ICA
        ica = ICA(
            n_components=None,
            method="fastica",
            max_iter="auto",
            random_state=42,
        )
        ica.fit(raw_proc, picks="eeg", decim=3)

        exclude = set()

        # Proxy blink detection via frontal channels
        proxy_candidates = [
            ch
            for ch in ["FP1-F7", "FP1-F3", "FP2-F4", "FP2-F8"]
            if ch in raw_proc.ch_names
        ]
        for ch in proxy_candidates:
            try:
                inds, _ = ica.find_bads_eog(raw_proc, ch_name=ch)
                exclude.update(inds)
            except Exception:
                pass

        ica.exclude = sorted(exclude)
        components_removed = len(ica.exclude)

        if components_removed > 0:
            print(f"Removing {components_removed} ICA components (blink proxies)")
            ica.apply(raw_proc)

        # 4. Downsampling
        raw_proc.resample(sfreq_new, npad="auto")

        return raw_proc

    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")


def add_seizure_annotations(raw: mne.io.Raw, summary_txt: str) -> mne.io.Raw:
    """
    Parse a CHB-MIT summary text file and add seizure annotations to a Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object corresponding to one EDF file.
    summary_txt : str
        Path to the subject summary TXT file (e.g., chb23-summary.txt).

    Returns
    -------
    raw : mne.io.Raw
        Raw object with MNE Annotations added for seizures.
    """
    # Extract the current EDF filename
    raw_fname = (
        str(raw.filenames[0]).split("/")[-1].split("\\")[-1]
    )  # handle Windows/Unix paths

    seizures = []

    with open(summary_txt, "r", encoding="latin-1") as f:
        content = f.read()

    # Split the content by "File Name:" to separate EDF entries
    files = content.split("File Name:")
    for file_block in files:
        if raw_fname not in file_block:
            continue

        # flexible regex to handle files with or without seizure numbers
        starts = [
            int(m.group(1))
            for m in re.finditer(r"Seizure(?: \d+)? Start Time: (\d+)", file_block)
        ]
        ends = [
            int(m.group(1))
            for m in re.finditer(r"Seizure(?: \d+)? End Time: (\d+)", file_block)
        ]

        seizures = list(zip(starts, ends))
        break

    if not seizures:
        print(f"No seizures found for {raw_fname}")
        return raw

    # Create MNE annotations
    onsets = [s[0] for s in seizures]
    durations = [s[1] - s[0] for s in seizures]
    descriptions = ["seizure"] * len(seizures)

    annotations = mne.Annotations(
        onset=onsets, duration=durations, description=descriptions
    )

    raw.set_annotations(annotations)
    print(f"Added {len(seizures)} seizure annotations to {raw_fname}")
    return raw


def add_seizure_annotations_bids(
    raw: mne.io.Raw, annotations_df: pd.DataFrame
) -> mne.io.Raw:
    """
    Add seizure annotations to a Raw object based on provided seizure times.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object.
    seizure_times : List[Tuple[float, float]]
        List of tuples with (start_time, end_time) in seconds.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with MNE Annotations added for seizures.
    """
    if not annotations_df.shape[0]:
        print("No seizures provided.")
        return raw

    onsets = [
        start
        for start, event_type in zip(
            annotations_df["onset"], annotations_df["eventType"]
        )
        if event_type == "sz"
    ]
    durations = [
        duration
        for duration, event_type in zip(
            annotations_df["duration"], annotations_df["eventType"]
        )
        if event_type == "sz"
    ]

    descriptions = ["seizure"] * len(onsets)
    if len(onsets) > 0:
        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )
        raw.set_annotations(annotations)
    return raw


def extract_segments_with_labels(
    raw: mne.io.Raw, segment_sec: float = 5.0, seizure_threshold: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-length segments from a Raw object and assign seizure/non-seizure labels.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG with annotations for seizures.
    segment_sec : float
        Segment length in seconds, default 5 s.
    seizure_threshold : float
        Fraction of segment that must overlap with seizure to label as 1, default 0.6.

    Returns
    -------
    X : np.ndarray
        Segments array of shape (n_segments, n_channels, n_samples).
    y : np.ndarray
        Labels array of shape (n_segments,), 1=seizure, 0=non-seizure.
    """
    sfreq = raw.info["sfreq"]
    n_samples_per_segment = int(segment_sec * sfreq)
    n_channels = raw.info["nchan"]

    data = raw.get_data()  # shape: (n_channels, n_times)
    n_total_samples = data.shape[1]

    # Build an array of same length as EEG, 1 where seizure, 0 elsewhere
    seizure_mask = np.zeros(n_total_samples, dtype=int)
    if raw.annotations is not None:
        for annot in raw.annotations:
            if annot["description"].lower() == "seizure":
                start_sample = int(annot["onset"] * sfreq)
                end_sample = int((annot["onset"] + annot["duration"]) * sfreq)
                seizure_mask[start_sample:end_sample] = 1

    segments = []
    labels = []

    # Slide over the signal in non-overlapping 5s windows
    for start in range(0, n_total_samples, n_samples_per_segment):
        end = start + n_samples_per_segment
        if end > n_total_samples:
            break  # discard last incomplete segment

        segment = data[:, start:end]
        segment_label = (
            1
            if seizure_mask[start:end].sum()
            >= seizure_threshold * n_samples_per_segment
            else 0
        )

        segments.append(segment)
        labels.append(segment_label)

    X = np.stack(segments)  # shape: (n_segments, n_channels, n_samples)
    y = np.array(labels)  # shape: (n_segments,)

    return X, y


def infer_preictal_interactal(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Infer preictal and interictal periods in the Raw object based on seizure annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG with seizure annotations.

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG with updated annotations for preictal and interictal periods.
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        print("No seizures to infer preictal/interictal periods.")
        return raw

    sfreq = raw.info["sfreq"]
    total_duration = raw.times[-1]

    new_annotations = []

    seizure_onsets = [
        annot["onset"]
        for annot in raw.annotations
        if annot["description"].lower() == "seizure"
    ]
    seizure_offsets = [
        annot["onset"] + annot["duration"]
        for annot in raw.annotations
        if annot["description"].lower() == "seizure"
    ]

    for onset, offset in zip(seizure_onsets, seizure_offsets):
        # mark 120 mins after offset as excluded
        exclude_start = offset
        ends = [start for start in seizure_onsets if start - offset > 0] + [
            offset + 120 * 60,
            total_duration,
        ]
        exclude_end = min(ends)
        new_annotations.append(
            {
                "onset": exclude_start,
                "duration": exclude_end - exclude_start,
                "description": "excluded",
            }
        )

    for i, onset in enumerate(seizure_onsets):
        # mark 15 mins before onset as preictal if the period is not excluded
        preictal_start = max(0, onset - 15 * 60)
        preictal_end = onset

        if not any(
            [
                annot["onset"] < preictal_end - 1 < annot["onset"] + annot["duration"]
                for annot in new_annotations
                if annot["description"] == "excluded"
            ]
        ):
            new_annot = {
                "duration": preictal_end - preictal_start,
                "onset": preictal_start,
            }
            for annot in new_annotations:
                if annot["onset"] < preictal_start < annot["onset"] + annot["duration"]:
                    print("here")
                    new_annot["duration"] = preictal_end - (
                        annot["onset"] + annot["duration"]
                    )
                    new_annot["onset"] = annot["onset"] + annot["duration"]

            new_annotations.append(
                {
                    "onset": new_annot["onset"],
                    "duration": new_annot["duration"],
                    "description": "preictal",
                }
            )

    # mark 105 before each preictal as excluded
    preictal_onsets = [
        annot["onset"]
        for annot in new_annotations
        if annot["description"] == "preictal"
    ]
    for pre_onset in preictal_onsets:
        exclude_start = max(0, pre_onset - 105 * 60)
        exclude_end = pre_onset
        flag = True
        for new_annot in new_annotations:
            if new_annot["description"] == "excluded" and (
                new_annot["onset"]
                < exclude_start
                < new_annot["onset"] + new_annot["duration"]
            ):
                new_annot["duration"] = pre_onset - new_annot["onset"]
                flag = False

        if flag:
            new_annotations.append(
                {
                    "onset": exclude_start,
                    "duration": exclude_end - exclude_start,
                    "description": "excluded",
                }
            )

    # mark all other times as interictal
    all_annot_intervals = [
        (annot["onset"], annot["onset"] + annot["duration"])
        for annot in new_annotations
        if (
            annot["description"] != "BAD boundary"
            and annot["description"] != "EDGE boundary"
        )
    ] + [
        (annot["onset"], annot["onset"] + annot["duration"])
        for annot in raw.annotations
        if annot["description"].lower() == "seizure"
    ]

    all_annot_intervals.sort(key=lambda x: x[0])  # sort by onset time
    print(all_annot_intervals)
    current_time = 0.0
    for start, end in all_annot_intervals:
        if current_time < start:
            new_annotations.append(
                {
                    "onset": current_time,
                    "duration": start - current_time,
                    "description": "interictal",
                }
            )
        current_time = max(current_time, end)
    if current_time < total_duration:
        new_annotations.append(
            {
                "onset": current_time,
                "duration": total_duration - current_time,
                "description": "interictal",
            }
        )

    raw.annotations.append(
        onset=[annot["onset"] for annot in new_annotations],
        duration=[annot["duration"] for annot in new_annotations],
        description=[annot["description"] for annot in new_annotations],
    )

    return raw


def convert_to_preactal_interactal(X, y):
    new_y = np.zeros_like(y)
    while len(np.where(y == 1)[0]) > 0:
        start_idx = np.where(y == 1)[0][0]
        end_idx = np.where(y == 1)[0][0] + int(3600 * 2 / 5)

        X = np.delete(X, np.s_[start_idx:end_idx], axis=0)
        y = np.delete(y, np.s_[start_idx:end_idx], axis=0)
        new_y = np.delete(new_y, np.s_[start_idx:end_idx], axis=0)

        new_y[start_idx - int(15 * 60 / 5) : start_idx] = 1

        end_idx = start_idx - int(15 * 60 / 5)
        start_idx = start_idx - int(120 * 60 / 5)

        if start_idx < 0:
            start_idx = 0

        X = np.delete(X, np.s_[start_idx:end_idx], axis=0)
        y = np.delete(y, np.s_[start_idx:end_idx], axis=0)
        new_y = np.delete(new_y, np.s_[start_idx:end_idx], axis=0)

    return X, new_y
