import numpy as np
import mne
from mne.preprocessing import ICA
from typing import Tuple, List, Set, Optional
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def preprocess_chbmit(
    raw: mne.io.Raw,
    sfreq_new: int = 128,
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    apply_ica: bool = True,
    apply_filter: bool = True,
    filter_type: str = "IIR",
    apply_downsampling: bool = True,
    downsample_method: str = "polyphase",
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
    filter_type : {"IIR", "FIR"}
        Type of bandpass filter to use. Default is "IIR".
    downsample_method : {"polyphase", "fft"}
        Method for downsampling. Default is "polyphase".
    normalize : {"zscore", "robust", None}
        Normalization method to apply after resampling. Default is "zscore".

    Returns
    -------
    raw_proc : mne.io.Raw
        The preprocessed Raw object (EEG channels only).
    """
    components_removed = 0
    try:
        raw_proc = raw.copy().pick(picks="eeg")
        
        # 1. Bandpass filter
        if apply_filter and filter_type == "FIR":
            raw_proc.filter(l_freq, h_freq, fir_design="firwin", phase="zero-double")
        elif apply_filter and filter_type == "IIR":
            fs = raw_proc.info["sfreq"]
            sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=fs, output="sos")
            raw_proc._data = sosfiltfilt(sos, raw_proc._data, axis=1)
        
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
        if apply_downsampling and downsample_method == "fft":
            raw_proc.resample(sfreq_new, npad="auto", method='fft')
        elif apply_downsampling and downsample_method == "polyphase":
            raw_proc.resample(sfreq_new, method='polyphase')

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


def _intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return (a_start < b_end) and (b_start < a_end)


def infer_preictal_interactal(
    raw: mne.io.Raw,
    preictal_minutes: int = 15,
    post_buffer_minutes: int = 60,
    pre_buffer_minutes: int = 45,
) -> mne.io.Raw:
    """
    Infer preictal and interictal periods in the Raw object based on seizure annotations.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG with seizure annotations.
    preictal_minutes : int
        Minutes before seizure onset to mark as preictal. Default = 15.
    post_buffer_minutes : int
        Minutes after seizure offset to exclude. Default = 60.
    pre_buffer_minutes : int
        Minutes before preictal to exclude. Default = 60.

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG with updated annotations for preictal and interictal periods.
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        print("No seizures to infer preictal/interictal periods.")
        return raw

    total_duration = raw.times[-1]

    # collect original annotations to preserve them
    orig_onsets = list(raw.annotations.onset)
    orig_durs = list(raw.annotations.duration)
    orig_descs = list(raw.annotations.description)

    # find seizure annotations (case-insensitive 'seizure')
    seizure_pairs = []
    for o, d, desc in zip(orig_onsets, orig_durs, orig_descs):
        if isinstance(desc, str) and desc.lower() == "seizure":
            seizure_pairs.append((o, o + d))

    if not seizure_pairs:
        return raw

    # sort seizure onsets/offsets by onset
    seizure_pairs = sorted(seizure_pairs, key=lambda x: x[0])
    seizure_onsets = [s for s, _ in seizure_pairs]
    seizure_offsets = [e for _, e in seizure_pairs]

    new_annots = []  # will hold dicts: {"onset":..., "duration":..., "description":...}

    # 1) build post_buffer for each seizure
    for idx, (onset, offset) in enumerate(zip(seizure_onsets, seizure_offsets)):
        # compute proposed end: offset + buffer, but clip to next seizure onset and to recording end
        proposed_end = offset + post_buffer_minutes * 60
        if idx + 1 < len(seizure_onsets):
            next_seizure_onset = seizure_onsets[idx + 1]
            end = min(proposed_end, next_seizure_onset, total_duration)
        else:
            end = min(proposed_end, total_duration)
        start = offset
        if end > start:
            new_annots.append({"onset": start, "duration": end - start, "description": "post_buffer"})

    # helper lists for quickly checking overlaps
    post_buffer_intervals = [(a["onset"], a["onset"] + a["duration"]) for a in new_annots if a["description"] == "post_buffer"]
    seizure_intervals = list(zip(seizure_onsets, seizure_offsets))

    # 2) build preictal (adjust if overlapping with post_buffer or seizure intervals)
    for onset in seizure_onsets:
        pre_start = max(0.0, onset - preictal_minutes * 60)
        pre_end = onset
        # find any exclusion interval that overlaps [pre_start, pre_end)
        # consider post_buffer and seizures as exclusions
        exclusion_intervals = post_buffer_intervals + seizure_intervals
        # If pre_start overlaps any exclusion interval, shift start to the end of that exclusion
        overlapping_ends = [e for (s, e) in exclusion_intervals if _intervals_overlap(s, e, pre_start, pre_end)]
        if overlapping_ends:
            # shift start to the latest overlapping end (the earliest safe start)
            adjusted_start = max(overlapping_ends)
            pre_start = max(pre_start, adjusted_start)
        # finalize
        if pre_end > pre_start:
            new_annots.append({"onset": pre_start, "duration": pre_end - pre_start, "description": "preictal"})

    # update preictal_onsets list (for pre_buffer calculation)
    preictal_intervals = [(a["onset"], a["onset"] + a["duration"]) for a in new_annots if a["description"] == "preictal"]

    # 3) build pre_buffer when positive
    delta_sec = (pre_buffer_minutes) * 60
    if delta_sec > 0:
        for pre_onset, pre_offset in preictal_intervals:
            exclude_end = pre_onset
            exclude_start = max(0.0, pre_onset - delta_sec)

            # If exclude_start overlaps a post_buffer, move start forward
            for pb_start, pb_end in post_buffer_intervals:
                if _intervals_overlap(pb_start, pb_end, exclude_start, exclude_end):
                    exclude_start = max(exclude_start, pb_end)

            # If exclude_start falls inside a seizure, push to end of that seizure + post_buffer_minutes (as in your original attempt)
            for sez_start, sez_end in seizure_intervals:
                if sez_start < exclude_start < sez_end:
                    exclude_start = min(total_duration, sez_end + post_buffer_minutes * 60 + 1)

            if exclude_end > exclude_start:
                new_annots.append({"onset": exclude_start, "duration": exclude_end - exclude_start, "description": "pre_buffer"})
    # 4) Build list of all intervals to compute interictal as complement.
    # Start from original non-boundary annotations (keep them) + seizure intervals + new_annots
    # We will preserve original annotations (including non-seizure like BAD boundary, etc).
    # Keep all annotations in combined lists (including boundaries)
    combined_onsets = list(orig_onsets)
    combined_durs = list(orig_durs)
    combined_descs = list(orig_descs)

    for a in new_annots:
        combined_onsets.append(a["onset"])
        combined_durs.append(a["duration"])
        combined_descs.append(a["description"])

    # ----------------------------------------------------------
    # Build occupied intervals used only for interictal computation
    # Ignore boundaries here
    # ----------------------------------------------------------
    occupied_intervals = []
    for o, d, desc in zip(combined_onsets, combined_durs, combined_descs):
        if desc in ["EDGE boundary", "BAD boundary"]:
            continue
        start = float(o)
        end   = float(o + d)
        occupied_intervals.append((start, end, desc))

    # ----------------------------------------------------------
    # Sort and check for overlaps & adjacency constraints
    # ----------------------------------------------------------
    occupied_intervals.sort(key=lambda x: x[0])

    for (s1, e1, d1), (s2, e2, d2) in zip(occupied_intervals, occupied_intervals[1:]):
        if e1 > s2:
            raise ValueError(f"Overlap detected: {d1} [{s1},{e1}) and {d2} [{s2},{e2})")
        if d1 == "seizure" and d2 != "post_buffer":
            raise ValueError(f"Expected post_buffer after seizure ending at {e1}, found {d2}")
        if d1 == "preictal" and d2 != "seizure":
            raise ValueError(f"Expected seizure after preictal ending at {e1}, found {d2}")
        if d1 == "pre_buffer" and d2 != "preictal":
            raise ValueError(f"Expected preictal after pre_buffer ending at {e1}, found {d2}")
        if (d1, d2) in [
            ("seizure", "post_buffer"),
            ("preictal", "seizure"),
            ("pre_buffer", "preictal")
        ]:
            if e1 != s2:
                raise ValueError(
                    f"Expected {d2} to start immediately after {d1}: "
                    f"{d1} ends at {e1}, {d2} starts at {s2}"
                )

    # ----------------------------------------------------------
    # Compute interictal as complement
    # ----------------------------------------------------------
    interictal_intervals = []
    cur = 0.0
    for start, end, desc in occupied_intervals:
        if cur < start:
            interictal_intervals.append((cur, start))
        cur = end
    if cur < total_duration:
        interictal_intervals.append((cur, total_duration))

    # add interictal intervals to combined lists
    for s, e in interictal_intervals:
        combined_onsets.append(s)
        combined_durs.append(e - s)
        combined_descs.append("interictal")

    # ----------------------------------------------------------
    # FINAL GLOBAL CHECK: ensure no gaps and no overlaps
    # except for EDGE/BAD boundaries (ignored in checks)
    # ----------------------------------------------------------

    # Build full intervals including interictal, but skip boundaries in checking
    full = []
    for o, d, desc in zip(combined_onsets, combined_durs, combined_descs):
        if desc in ["EDGE boundary", "BAD boundary"]:
            continue
        full.append((o, o + d, desc))

    # Sort
    full_sorted = sorted(full, key=lambda x: x[0])

    # Check no overlap and no gap
    for (s1, e1, d1), (s2, e2, d2) in zip(full_sorted, full_sorted[1:]):
        # Overlap check
        if e1 > s2:
            raise ValueError(
                f"Overlap detected between intervals ({s1},{e1}) and ({s2},{e2}). "
                f"Inspect annotation logic."
            )

        # Gap check
        if e1 != s2:
            raise ValueError(
                f"Gap detected: interval {d1} ends at {e1}, "
                f"but next interval {d2} starts at {s2}."
            )
        
    # ----------------------------------------------------------
    # Final write-back: includes ALL original + new + boundaries
    # ----------------------------------------------------------
    raw.set_annotations(mne.Annotations(onset=combined_onsets, duration=combined_durs, description=combined_descs))
    return raw


def extract_segments_with_labels_bids(
    raw: mne.io.Raw,
    segment_sec: float = 5.0,
    keep_labels: Set[str] = {"seizure", "preictal", "interictal", "post_buffer", "pre_buffer"},
    preictal_oversample_factor: float = 1.0,
    seizure_oversample_factor: float = 1.0,
):
    """
    Segment an annotated MNE Raw object into fixed-length EEG epochs, produce
    per-epoch metadata, and optionally oversample specific annotation types
    (preictal or seizure) using internal overlap.

    Parameters
    ----------
    raw : mne.io.Raw
        Annotated Raw EEG data.
    segment_sec : float
        Length of each EEG segment in seconds.
    keep_labels : Set[str]
        Set of annotation labels to extract segments from.  Default includes
        'seizure', 'preictal', 'interictal', 'post_buffer', 'pre_buffer'.
    preictal_oversample_factor : float
        Oversampling factor for 'preictal' segments. Default = 1.0 (no oversampling).
    seizure_oversample_factor : float
        Oversampling factor for 'seizure' segments. Default = 1.0 (no oversampling).

    Returns
    -------
    X : np.ndarray
        EEG segments of shape (n_segments, n_channels, n_times).
    y : np.ndarray
        Labels for each segment of shape (n_segments,).
    meta_df : pd.DataFrame
        Metadata DataFrame with per-segment information.
    event_stats : List[Dict]
        Summary statistics for each annotated event.

    Notes
    -----
    - Segments overlapping annotation boundaries are excluded.
    - Augmented segments (due to oversampling) are flagged in metadata.
    
    """

    import numpy as np
    import pandas as pd
    import mne

    X_list = []
    y_list = []
    meta_list = []

    ann_counter = {lab: 0 for lab in keep_labels}
    event_stats = []
    sfreq = raw.info["sfreq"]
    global_id = 0 

    for desc, onset, duration in zip(
        raw.annotations.description,
        raw.annotations.onset,
        raw.annotations.duration,
    ):
        if desc not in keep_labels:
            continue

        # ------------------------------------------------------------------
        # Determine internal overlap from oversampling factor
        # ------------------------------------------------------------------
        if desc.lower() == "preictal" and preictal_oversample_factor > 1.0:
            overlap_ratio = 1.0 - 1.0 / preictal_oversample_factor
            internal_overlap = min(segment_sec * overlap_ratio, segment_sec - 0.01)
        elif desc.lower() == "seizure" and seizure_oversample_factor > 1.0:
            overlap_ratio = 1.0 - 1.0 / seizure_oversample_factor
            internal_overlap = min(segment_sec * overlap_ratio, segment_sec - 0.01)
        else:
            internal_overlap = 0.0

        # ------------------------------------------------------------------
        # Crop annotation region and segment into epochs
        # ------------------------------------------------------------------
        segment_raw = raw.copy().crop(tmin=onset, tmax=onset + duration)

        epochs = mne.make_fixed_length_epochs(
            segment_raw,
            duration=segment_sec,
            overlap=internal_overlap,
            preload=True,
            reject_by_annotation=True,
        )

        if len(epochs) == 0:
            continue

        # ------------------------------------------------------------------
        # Event bookkeeping
        # ------------------------------------------------------------------
        ann_counter[desc] += 1
        event_id = f"{desc}_{ann_counter[desc]}"
        n_segs = len(epochs)

        # Epoch start times relative to event onset
        starts = (epochs.events[:, 0]) / sfreq
        # ------------------------------------------------------------------
        # Augmentation flag
        # ------------------------------------------------------------------
        baseline = np.arange(0, duration + 1e-6, segment_sec)

        def is_augmented(t):
            return not np.any(np.isclose(t, baseline, atol=1e-3))

        aug_flags = np.array([is_augmented(t) for t in starts], dtype=int)

        # ------------------------------------------------------------------
        # Noise features
        # ------------------------------------------------------------------
        data = epochs.get_data()  # (N, C, T)
        pp = data.max(axis=2) - data.min(axis=2)
        sd = data.std(axis=2)

        pp_mean = pp.mean(axis=1)
        pp_max = pp.max(axis=1)
        sd_mean = sd.mean(axis=1)
        sd_max = sd.max(axis=1)

        # ------------------------------------------------------------------
        # Collect epoch data & metadata
        # ------------------------------------------------------------------
        for i in range(n_segs):
            X_list.append(data[i])
            y_list.append(desc)

            meta_list.append({
                "event_id": event_id,
                "label": desc,
                "epoch_index_within_event": i,
                "global_epoch_id": global_id,
                "n_segments_in_event": n_segs,
                "start_time_in_event": float(starts[i]),
                "augmented": int(aug_flags[i]),
                "pp_mean": float(pp_mean[i]),
                "pp_max": float(pp_max[i]),
                "sd_mean": float(sd_mean[i]),
                "sd_max": float(sd_max[i]),
                "onset_sec": float(onset),
                "duration_sec": float(duration),
            })
            global_id += 1

        # ------------------------------------------------------------------
        # Event summary
        # ------------------------------------------------------------------
        event_stats.append({
            "event_id": event_id,
            "label": desc,
            "onset_sec": float(onset),
            "duration_sec": float(duration),
            "n_segments": n_segs,
            "applied_overlap_sec": float(internal_overlap),
        })

    # ----------------------------------------------------------------------
    # Final assembly
    # ----------------------------------------------------------------------
    if not X_list:
        return (
            np.empty((0,)),
            np.empty((0,)),
            pd.DataFrame(),
            event_stats,
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    meta_df = pd.DataFrame(meta_list)

    return X, y, meta_df, event_stats



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
