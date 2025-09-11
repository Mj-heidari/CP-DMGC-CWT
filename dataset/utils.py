import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Tuple, List, Set
import re
import pandas as pd


def preprocess_chbmit(
    raw: mne.io.Raw, only_resample: bool = False, sfreq_new: int = 128, l_freq: float = 0.5, h_freq: float = 40.0
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

        if not only_resample:
            
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

def infer_preictal_interactal(raw: mne.io.Raw, dynamic_preictal: bool = False, SPE: int = 0) -> mne.io.Raw:
    """
    Infer preictal and interictal periods in the Raw object based on seizure annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG with seizure annotations.
    dynamic_preictal : bool, optional
        If True, adjust preictal duration dynamically when overlapping with excluded intervals.
        Default is False.
    SPE : int, optional
        Seizure Prediction Horizon (in seconds). The preictal period will end at (onset - SPE).
        Default is 0.
        
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
        preictal_end = max(0, onset - SPE)

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
            if dynamic_preictal:
                for annot in new_annotations:
                    if annot["onset"] < preictal_start < annot["onset"] + annot["duration"]:
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

def extract_segments_with_labels_bids(
    raw: mne.io.Raw,
    segment_sec: float = 5.0,
    overlap: float = 0.0,
    keep_labels: Set[str] = {"preictal", "interictal"},
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment EEG into fixed-length epochs based on annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording with annotations.
    segment_sec : float
        Length of each segment in seconds.
    overlap : float
        Overlap between consecutive segments in seconds.
    keep_labels : set of str
        Annotation descriptions to keep (e.g., {"preictal", "interictal"}).

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
        Segmented EEG data.
    y : np.ndarray, shape (n_epochs,)
        Labels corresponding to each segment (e.g., "preictal", "interictal").
    group_ids : np.ndarray, shape (n_epochs,)
        IDs for the original annotation interval each segment came from.
        Useful for splitting train/test at the subject or event level.
    """
    X, y, group_ids = [], [], []
    ann_counter = {lab: 0 for lab in keep_labels}  # counter for each label

    for idx, (desc, onset, duration) in enumerate(
        zip(raw.annotations.description,
            raw.annotations.onset,
            raw.annotations.duration)
    ):
        if desc not in keep_labels:
            continue

        # Segment only within this annotation
        segment_raw = raw.copy().crop(tmin=onset, tmax=onset + duration)
        epochs = mne.make_fixed_length_epochs(
            segment_raw,
            duration=segment_sec,
            overlap=overlap,
            preload=True,
            reject_by_annotation=True  # ensures BAD/EDGE are dropped
        )

        # Assign group ID for this block (useful for CV splitting later)
        ann_counter[desc] += 1
        block_id = f"{desc}_{ann_counter[desc]}"

        # Collect
        X.append(epochs.get_data())
        y.extend([desc] * len(epochs))
        group_ids.extend([block_id] * len(epochs))

    if not X:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    group_ids = np.array(group_ids)

    return X, y, group_ids

