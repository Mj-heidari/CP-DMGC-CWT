"""
Unified feature extraction and visualization pipeline for the CHB‑MIT dataset.


Features are derived using the `compute_features` function defined in
`CP‑DMGC‑CWT‑main/feature_extraction.py`, which follows the guidance
from the "What features matter" document.  By default, all 20
features are extracted (mean amplitude, skewness, kurtosis, RMS,
line length, zero crossing rate, Hjorth parameters, sample entropy,
Higuchi fractal dimension, delta/theta/alpha/beta/gamma band power,
spectral entropy, intensity‑weighted mean frequency, spectral edge
frequency, peak frequency and mean absolute correlation).  You can
restrict the extraction to a subset of these features using the
`--features`, `--num_features` and `--remove_features` options.

Usage examples:

    # Extract all features for the first two subjects and save to Feature folder
    python feature_pipeline.py \
        --dataset_dir /path/to/BIDS_CHB-MIT \
        --num_subjects 2 \
        --save_dir Feature

    # Extract only selected features for subjects 03 and 05 and show the
    # interactive plots for each session
    python feature_pipeline.py \
        --dataset_dir /path/to/BIDS_CHB-MIT \
        --subjects 03 05 \
        --features mean_amp rms beta_power alpha_power \
        --visualize

    # Extract all but remove a specific feature and save static PNG
    python feature_pipeline.py \
        --dataset_dir /path/to/BIDS_CHB-MIT \
        --num_subjects 1 \
        --remove_features peak_frequency \
        --save_plots /tmp/plots

Options
-------
``--dataset_dir``
    Path to the root of the CHB‑MIT dataset in BIDS format.  The
    script expects to find subdirectories like ``sub-01/ses-01/eeg``
    containing ``processed_segments.npz`` files.
``--subjects``
    Optional list of subject identifiers (e.g. ``01 02 05``).  Only
    these subjects will be processed.  If omitted, all subjects
    found under ``dataset_dir`` are considered.
``--num_subjects``
    Optional limit on the number of subjects to process.  When set,
    only the first ``N`` subjects (lexicographically sorted) are
    processed.  If ``--subjects`` is also provided, this limit
    applies after filtering by subjects.
``--features``
    Optional list of specific feature names to compute.  See
    `feature_extraction.py` for the full list of available features.
    When provided, only these features will be extracted.  Use
    underscores between words (e.g., ``delta_power``) as shown in
    the default feature list.  This option overrides
    ``--num_features`` if both are given.
``--num_features``
    Optional integer specifying how many of the default features to
    extract.  The features are ordered as they appear in the
    dictionary returned by `compute_features`.  If unspecified, all
    features are used.
``--remove_features``
    Optional list of feature names to omit from extraction.  This
    allows manual exclusion of features without enumerating the
    entire list.  Removal is applied after applying
    ``--features``/``--num_features``.
``--save_dir``
    Directory within the repository where the output feature files
    (``*.npz``) will be saved.  Defaults to ``Feature``.  The script
    creates the directory if it does not exist.
``--visualize``
    Flag to enable interactive visualisation with MNE's viewer.  When
    set, the script displays the raw EEG segments with appended
    feature channels for each session.  Close the viewer window to
    proceed to the next session.
``--save_plots``
    Optional directory in which to save static PNG plots for each
    session instead of opening interactive viewers.  Each file will
    be named ``<subject>_<session>_plot.png``.  If not provided and
    ``--visualize`` is also not set, no plots are generated.

The script is intended to be executed from the root of the cloned
repository so that ``CP‑DMGC‑CWT‑main`` is discoverable.  It
modifies ``sys.path`` at runtime to import the ``compute_features``
function from ``feature_extraction.py``.

"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting imports; available only when visualisation is requested
try:
    import mne  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MNE = True
except Exception:
    _HAVE_MNE = False


def _import_compute_features(repo_root: str) -> Tuple[callable, List[str]]:
    """Dynamically import compute_features and obtain the default feature order.

    This helper appends the path to the CP‑DMGC‑CWT repository to
    sys.path, imports the ``compute_features`` function and then calls
    it on a dummy segment to discover the list of feature names.  The
    dummy call is cheap since the segment contains only zeros.

    Parameters
    ----------
    repo_root : str
        Path to the directory containing ``CP‑DMGC‑CWT‑main``.

    Returns
    -------
    (compute_fn, feature_names)
        A tuple containing the imported ``compute_features`` function
        and the list of default feature names in the order returned by
        the function.
    """
    # Locate feature_extraction.py relative to repo root
    fe_path = os.path.join(repo_root, "CP-DMGC-CWT-main", "feature_extraction.py")
    if not os.path.isfile(fe_path):
        raise FileNotFoundError(
            f"Could not find feature_extraction.py at expected location: {fe_path}."
        )
    # Add repo root and module path to sys.path
    sys.path.insert(0, os.path.join(repo_root, "CP-DMGC-CWT-main"))
    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cp_feat", fe_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load feature_extraction module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    compute_fn = getattr(module, "compute_features")
    # Determine default feature order by calling compute_fn on dummy data
    dummy_seg = np.zeros((1, 128))  # 1 channel, 128 samples
    dummy_features = compute_fn(dummy_seg, fs=128.0)
    default_names = list(dummy_features.keys())
    # Clean up sys.path to avoid side effects
    sys.path.pop(0)
    return compute_fn, default_names


def _discover_sessions(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """Find all processed_segments.npz files under dataset_dir.

    This function walks the dataset directory for patterns
    ``sub-*/ses-*/eeg/processed_segments.npz`` and returns a list of
    tuples containing (subject_id, session_id, npz_path).

    Parameters
    ----------
    dataset_dir : str
        Root directory of the BIDS CHB‑MIT dataset.

    Returns
    -------
    List[Tuple[str, str, str]]
        Sorted list of (subject_id, session_id, npz_path).
    """
    sessions = []
    # Use os.walk to search for processed_segments.npz
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname == "processed_segments.npz":
                rel = os.path.relpath(root, dataset_dir)
                # Expect rel like sub-01/ses-01/eeg
                parts = rel.split(os.sep)
                if len(parts) >= 3:
                    sub = parts[0]
                    ses = parts[1]
                    sessions.append((sub, ses, os.path.join(root, fname)))
    # Sort by subject then session
    return sorted(sessions, key=lambda x: (x[0], x[1]))


def _filter_sessions(
    sessions: List[Tuple[str, str, str]],
    subjects: Optional[List[str]] = None,
    num_subjects: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """Filter the list of sessions by subject list and/or number of subjects.

    Parameters
    ----------
    sessions : list of tuples
        The full list of (subject_id, session_id, npz_path) tuples.
    subjects : list of str, optional
        If provided, only sessions with subject identifiers in this
        list are kept.
    num_subjects : int, optional
        If provided, limit the number of distinct subjects processed
        to the first ``num_subjects`` (sorted).  Applied after
        filtering by ``subjects``.

    Returns
    -------
    List[Tuple[str, str, str]]
        The filtered list of sessions.
    """
    if subjects:
        subjects_set = set(subjects)
        sessions = [s for s in sessions if s[0].split("-")[-1] in subjects_set]
    if num_subjects is not None and num_subjects > 0:
        seen = set()
        filtered = []
        for sub, ses, path in sessions:
            key = sub.split("-")[-1]
            if key not in seen:
                if len(seen) >= num_subjects:
                    break
                seen.add(key)
            filtered.append((sub, ses, path))
        sessions = filtered
    return sessions


def _select_feature_names(
    all_names: List[str],
    include_names: Optional[List[str]] = None,
    num_features: Optional[int] = None,
    remove_names: Optional[List[str]] = None,
) -> List[str]:
    """Determine which feature names to extract.

    The logic is as follows:

    1. Start with ``include_names`` if provided; otherwise use
       ``all_names``.
    2. If ``num_features`` is provided and no explicit ``include_names``
       were given, truncate the list to the first ``num_features``.
    3. Remove any names specified in ``remove_names``.

    Parameters
    ----------
    all_names : list of str
        The complete list of feature names returned by compute_features.
    include_names : list of str, optional
        Explicit list of feature names to include.  If provided,
        ``num_features`` is ignored.
    num_features : int, optional
        Number of features to include from the start of ``all_names``.
    remove_names : list of str, optional
        Names of features to exclude from the final list.

    Returns
    -------
    List[str]
        The final list of feature names to use.
    """
    # Step 1: start with include_names or all_names
    if include_names:
        names = [n for n in include_names if n in all_names]
    else:
        names = list(all_names)
        # Step 2: apply num_features
        if num_features is not None and num_features > 0:
            names = names[:num_features]
    # Step 3: remove names
    if remove_names:
        names = [n for n in names if n not in remove_names]
    return names


def _compute_feature_matrix(
    X: np.ndarray,
    fs: float,
    compute_fn,
    feature_names: List[str],
) -> np.ndarray:
    """Compute selected features for each segment.

    Parameters
    ----------
    X : ndarray
        Array of shape (n_segments, n_channels, n_times).
    fs : float
        Sampling frequency of the segments.
    compute_fn : callable
        Function that computes a dict of all features from a single segment.
    feature_names : list of str
        Names of features to extract from the dict returned by ``compute_fn``.

    Returns
    -------
    ndarray
        Array of shape (n_segments, len(feature_names)) containing the
        selected feature values for each segment.
    """
    n_segments = X.shape[0]
    features_matrix = np.empty((n_segments, len(feature_names)), dtype=float)
    for i in range(n_segments):
        feats = compute_fn(X[i], fs)
        # Fill row with selected features in order
        for j, fname in enumerate(feature_names):
            features_matrix[i, j] = feats.get(fname, np.nan)
    return features_matrix


def _save_features(
    sub: str,
    ses: str,
    save_dir: str,
    F: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> str:
    """Save features for one session to a compressed npz file.

    The file will be named ``<sub>_<ses>_features.npz`` and contain
    arrays ``features``, ``feature_names``, ``labels`` and ``group_ids``.

    Parameters
    ----------
    sub, ses : str
        Subject and session identifiers.
    save_dir : str
        Directory where the file should be saved.  Created if needed.
    F : ndarray
        Feature matrix of shape (n_segments, n_features).
    feature_names : list of str
        Names of the features (in order) corresponding to the columns of ``F``.
    labels : ndarray
        Array of labels for each segment.
    group_ids : ndarray
        Array of group identifiers for each segment.

    Returns
    -------
    str
        Path to the saved .npz file.
    """
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{sub}_{ses}_features.npz"
    out_path = os.path.join(save_dir, fname)
    np.savez_compressed(
        out_path,
        features=F,
        feature_names=np.array(feature_names, dtype=object),
        labels=labels,
        group_ids=group_ids,
    )
    return out_path


def _visualize_session(
    sub: str,
    ses: str,
    X: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    F: np.ndarray,
    feature_names: List[str],
    fs: float,
    ch_names: Optional[List[str]] = None,
    plot_dir: Optional[str] = None,
) -> None:
    """Create an interactive or static visualisation of EEG + features.

    Parameters
    ----------
    sub, ses : str
        Subject and session identifiers.
    X : ndarray
        Raw EEG data of shape (n_segments, n_channels, n_times).
    labels : ndarray
        Array of labels (strings) for each segment.
    group_ids : ndarray
        Array of group identifiers (strings) for each segment.
    F : ndarray
        Feature matrix of shape (n_segments, n_features).
    feature_names : list of str
        Names of the features corresponding to columns of ``F``.
    fs : float
        Sampling frequency.
    ch_names : list of str, optional
        Names of the EEG channels.  If ``None``, generic names are
        generated.
    plot_dir : str, optional
        If provided, save a static plot to this directory instead of
        opening an interactive viewer.  Each file will be named
        ``<sub>_<ses>_plot.png``.  The static plot shows the first
        epoch (segment) with up to 16 channels (including features).
    """
    if not _HAVE_MNE:
        print("[WARN] MNE or matplotlib not available; skipping visualisation.")
        return
    n_segments, n_channels, n_times = X.shape
    n_features = F.shape[1]
    if ch_names is None or len(ch_names) != n_channels:
        ch_names = [f"EEG{idx+1:03d}" for idx in range(n_channels)]
    eeg_types = ["eeg"] * n_channels
    feat_types = ["misc"] * n_features
    feat_ch_names = [f"feat_{nm}" for nm in feature_names]
    all_ch_names = ch_names + feat_ch_names
    all_types = eeg_types + feat_types
    # Replicate features across time for each epoch
    F_rep = np.repeat(F[:, :, np.newaxis], n_times, axis=2)
    data_combined = np.concatenate([X, F_rep], axis=1)  # (n_segments, n_channels+n_features, n_times)
    info = mne.create_info(ch_names=all_ch_names, sfreq=fs, ch_types=all_types)
    epochs = mne.EpochsArray(data_combined, info, verbose=False)
    epochs.metadata = pd.DataFrame({"label": labels, "group_id": group_ids})
    tag = f"{sub}_{ses}"
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        # Plot only the first epoch and limit channels for readability
        ep0 = epochs[0].get_data()[0]  # shape (n_ch, n_times)
        plt.figure(figsize=(10, 6))
        max_plot = min(16, ep0.shape[0])
        t = np.arange(n_times) / fs
        offset = 0.0
        for ch in range(max_plot):
            plt.plot(t, ep0[ch] + offset)
            offset += 5.0
        plt.title(f"{tag} - epoch 1 (EEG + selected features)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (offset)")
        out_png = os.path.join(plot_dir, f"{tag}_plot.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[OK] Saved plot: {out_png}")
    else:
        print(f"[INFO] Opening interactive viewer for {tag}. Close the window to continue...")
        epochs.plot(scalings="auto")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the root of the CHB‑MIT dataset (BIDS structure).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of subject identifiers (e.g., 01 02) to process."
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=None,
        help="Optional maximum number of subjects to process (after subject filtering)."
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional list of feature names to extract. If provided, overrides --num_features."
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=None,
        help="Optional limit on the number of default features to extract. Ignored if --features is provided."
    )
    parser.add_argument(
        "--remove_features",
        nargs="*",
        default=None,
        help="Optional list of feature names to exclude from extraction."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="Feature",
        help="Directory to save the per-session feature files (relative to current working directory)."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, open interactive MNE viewers for each session after saving features."
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        default=None,
        help="If provided, save static PNG plots for each session to this directory instead of opening interactive viewers."
    )
    args = parser.parse_args()

    # Determine repository root relative to this script location.  We assume
    # the script resides at the project root alongside `CP-DMGC-CWT-main`.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = script_dir
    # Import compute_features from the repository
    compute_fn, default_feature_names = _import_compute_features(repo_root)

    # Determine which feature names to extract
    feature_names = _select_feature_names(
        all_names=default_feature_names,
        include_names=args.features,
        num_features=args.num_features,
        remove_names=args.remove_features,
    )
    if not feature_names:
        raise ValueError("No features selected for extraction. Check --features and --remove_features options.")

    # Discover all sessions
    sessions = _discover_sessions(args.dataset_dir)
    # Filter sessions by subjects and limit number of subjects
    sessions = _filter_sessions(sessions, subjects=args.subjects, num_subjects=args.num_subjects)
    if not sessions:
        print("[WARN] No sessions found matching the specified criteria.")
        return
    # Process each session
    for idx, (sub, ses, npz_path) in enumerate(sessions, start=1):
        print(f"\n[{idx}/{len(sessions)}] Processing {sub} {ses} -> {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        group_ids = data["group_ids"]
        n_segments, n_channels, n_times = X.shape
        # Infer sampling frequency assuming 5‑second segments
        fs = n_times / 5.0 if n_times % 5 == 0 else 128.0
        print(f"    Segments={n_segments}, Channels={n_channels}, Samples/seg={n_times}, fs≈{fs:.2f} Hz")
        # Compute feature matrix
        F = _compute_feature_matrix(X, fs, compute_fn, feature_names)
        # Save features
        out_path = _save_features(sub, ses, args.save_dir, F, feature_names, y, group_ids)
        print(f"    [OK] Features saved to {out_path} (shape {F.shape})")
        # Visualisation
        if args.visualize or args.save_plots:
            # Determine channel names from optional metadata JSON if available
            ch_names = None
            # Attempt to find matching JSON in METADATA_JSON list if provided
            # Not part of CLI to avoid complexity; user can extend this script as needed
            _plot_dir = args.save_plots if args.save_plots else None
            _visualize_session(sub, ses, X, y, group_ids, F, feature_names, fs, ch_names, _plot_dir)

    print("\n[DONE] Feature extraction complete.")


if __name__ == "__main__":
    main()
