import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Tuple, List, Set, Optional
import pandas as pd


def preprocess_chbmit(
    raw: mne.io.Raw,
    sfreq_new: int = 128,
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    apply_ica: bool = True,
    apply_filter: bool = True,
    normalize: Optional[str] = "zscore",
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
        Upper frequency bound for bandpass filter, default = 50 Hz.
    apply_ica: bool
        if true the ica will be applied, default = True.
    apply_filter: bool
        If true the filtering will be applied, default = True.
    normalize : {"zscore", "robust", None}
        Normalization method to apply after resampling. Default is "zscore".

    Returns
    -------
    raw_proc : mne.io.Raw
        The preprocessed Raw object (EEG channels only).
    """
    components_removed = 0
    try:
        raw_proc = raw.copy().pick_types(eeg=True)
        
        # 1. Bandpass filter
        if apply_filter:
            raw_proc.filter(l_freq, h_freq, fir_design="firwin", phase="zero-double")

        # # 2. Notch filter (60 Hz + harmonics)
        # raw_proc.notch_filter(np.arange(60, h_freq, 60), fir_design="firwin")

        # 3. ICA
        if apply_ica:
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
                print(
                    f"Removing {components_removed} ICA components (blink proxies)"
                )
                ica.apply(raw_proc)

        # 4. Downsampling
        raw_proc.resample(sfreq_new, npad="auto")

        # 4. Normalization (channel-wise, on whole continuous data)
        if normalize is not None:
            data = raw_proc.get_data()  # shape (n_channels, n_times)

            if normalize == "zscore":
                mean = data.mean(axis=1, keepdims=True)
                std = data.std(axis=1, keepdims=True)
                std[std == 0] = 1.0
                data = (data - mean) / std

            elif normalize == "robust":
                median = np.median(data, axis=1, keepdims=True)
                q75 = np.percentile(data, 75, axis=1, keepdims=True)
                q25 = np.percentile(data, 25, axis=1, keepdims=True)
                iqr = q75 - q25
                iqr[iqr == 0] = 1.0
                data = (data - median) / iqr

            else:
                raise ValueError(
                    f"Unknown normalization method: {normalize!r}."
                    "Use 'zscore', 'robust', or None."
                )

            # write normalized data back into the Raw object's buffer (preserve dtype)
            raw_buf = raw_proc._data
            raw_buf[...] = data.astype(raw_buf.dtype, copy=False)

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


def infer_preictal_interactal(
    raw: mne.io.Raw,
    preictal_minutes: int = 15,
    postictal_exclude_minutes: int = 120,
    dynamic_preictal: bool = False,
    SPE: int = 0,
) -> mne.io.Raw:
    """
    Infer preictal and interictal periods in the Raw object based on seizure annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG with seizure annotations.
    preictal_minutes : int
        Minutes before seizure onset to mark as preictal. Default = 15.
    postictal_exclude_minutes : int
        Minutes after seizure offset to exclude. Default = 120.
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

    # sfreq = raw.info["sfreq"]
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
        # mark postictal_exclude_minutes mins after offset as excluded
        exclude_start = offset
        ends = [start for start in seizure_onsets if start - offset > 0] + [
            offset + postictal_exclude_minutes * 60,
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
        # mark preictal_minutes mins before onset as preictal if the period is not excluded
        preictal_start = max(0, onset - preictal_minutes  * 60)
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
                    if (
                        annot["onset"]
                        < preictal_start
                        < annot["onset"] + annot["duration"]
                    ):
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

            # exclude the SPE gap
            if SPE > 0 and preictal_end < onset:
                new_annotations.append(
                    {
                        "onset": preictal_end,
                        "duration": onset - preictal_end,
                        "description": "excluded",
                    }
                )

    # mark (postictal_exclude_minutes - preictal_minutes) before each preictal as excluded
    preictal_onsets = [
        annot["onset"]
        for annot in new_annotations
        if annot["description"] == "preictal"
    ]
    for pre_onset in preictal_onsets:
        exclude_start = max(0, pre_onset - (postictal_exclude_minutes - preictal_minutes) * 60)
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
    event_stats = []

    for idx, (desc, onset, duration) in enumerate(
        zip(
            raw.annotations.description, raw.annotations.onset, raw.annotations.duration
        )
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
            reject_by_annotation=True,  # ensures BAD/EDGE are dropped
        )

        # Assign group ID for this block (useful for CV splitting later)
        ann_counter[desc] += 1
        block_id = f"{desc}_{ann_counter[desc]}"

        # Collect
        X.append(epochs.get_data())
        y.extend([desc] * len(epochs))
        group_ids.extend([block_id] * len(epochs))

        # --- Store event-level stats ---
        event_stats.append({
            "event_id": block_id,
            "label": desc,
            "onset_sec": float(onset),
            "duration_sec": float(duration),
            "n_segments": len(epochs),
        })

    if not X:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    group_ids = np.array(group_ids)

    return X, y, group_ids, event_stats


def scale_to_uint16(X: np.ndarray):
    """
    Scale a 3D EEG dataset (samples × channels × time-points)
    to uint16 per sample.

    Parameters
    ----------
    X : np.ndarray
        EEG data of shape (n_samples, n_channels, n_times).

    Returns
    -------
    X_uint16 : np.ndarray
        Scaled EEG data in uint16.
    scales : np.ndarray
        Per-sample (min, max) values used for scaling (shape: n_samples × 2).
    """
    n_samples, n_channels, n_times = X.shape
    X_uint16 = np.zeros_like(X, dtype=np.uint16)
    scales = np.zeros((n_samples, 2), dtype=np.float32)

    for i in range(n_samples):
        x = X[i]
        x_min = x.min()
        x_max = x.max()
        scales[i] = (x_min, x_max)

        if x_max == x_min:  # avoid division by zero
            scaled = np.zeros_like(x)
        else:
            scaled = (x - x_min) / (x_max - x_min) * 65535

        X_uint16[i] = scaled.astype(np.uint16)

    return X_uint16, scales


def invert_uint16_scaling(X_uint16: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Reconstruct float32 EEG data from uint16 scaled values.

    Parameters
    ----------
    X_uint16 : np.ndarray
        EEG data scaled to uint16, shape (n_samples, n_channels, n_times).
    scales : np.ndarray
        Per-sample (min, max) values, shape (n_samples, 2).

    Returns
    -------
    X_reconstructed : np.ndarray
        Reconstructed EEG data as float32, same shape as X_uint16.
    """
    n_samples, n_channels, n_times = X_uint16.shape
    X_reconstructed = np.zeros((n_samples, n_channels, n_times), dtype=np.float32)

    for i in range(n_samples):
        x_uint16 = X_uint16[i].astype(np.float32)
        x_min, x_max = scales[i]
        if x_max == x_min:  # flat signal case
            X_reconstructed[i] = np.full_like(
                x_uint16, fill_value=x_min, dtype=np.float32
            )
        else:
            X_reconstructed[i] = (x_uint16 / 65535.0) * (x_max - x_min) + x_min

    return X_reconstructed
