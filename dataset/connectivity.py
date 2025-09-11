import mne
from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import numpy as np


def split_epoch(epoch, sub_duration=1.0):
    """Split a single Epoch into smaller sub-epochs."""
    sfreq = epoch.info["sfreq"]
    n_channels, n_times = epoch.get_data().shape[1:]
    data = epoch.get_data()[0]

    sub_len = int(sub_duration * sfreq)
    n_subs = n_times // sub_len

    sub_epochs_data = np.array(
        [data[:, i * sub_len : (i + 1) * sub_len] for i in range(n_subs)]
    )

    info = epoch.info.copy()
    sub_epochs = mne.EpochsArray(sub_epochs_data, info, verbose=False)
    return sub_epochs


def coh_connectivity(epoch, fmin=8.0, fmax=12.0):
    con = spectral_connectivity_epochs(
        epoch,
        method=["coh"],
        mode="multitaper",
        sfreq=epoch.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,  # average over frequencies in band
        n_jobs=1,
        verbose=False,
    )
    return con.get_data(output="dense").squeeze()


def plv_connectivity(epoch, fmin=8.0, fmax=12.0, sub_duration=0.5):
    sub_epochs = split_epoch(epochs, sub_duration)
    con = spectral_connectivity_epochs(
        sub_epochs,
        method=["plv"],
        mode="multitaper",
        sfreq=epoch.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,  # average over frequencies in band
        n_jobs=1,
        verbose=False,
    )
    return con.get_data(output="dense").squeeze()


def aec_connectivity(epoch, fmin=8.0, fmax=12.0):
    epoch_band = (
        epoch.copy().filter(fmin, fmax, verbose=False).apply_hilbert(envelope=True)
    )
    data = epoch_band.get_data()
    aec = envelope_correlation(data, orthogonalize="pairwise")
    return aec.get_data(output="dense").squeeze()


def make_masks(n_channels=18):
    """Create boolean masks for intra-left, intra-right, inter-LR, and global."""
    mask = np.zeros((n_channels, n_channels), dtype=bool)

    idx_left = slice(0, 8)
    # idx_mid = slice(8, 10)
    idx_right = slice(10, 18)

    intra_left = np.zeros_like(mask)
    intra_left[idx_left, idx_left] = True
    intra_left = np.tril(intra_left, k=-1).astype(bool)

    intra_right = np.zeros_like(mask)
    intra_right[idx_right, idx_right] = True
    intra_right = np.tril(intra_right, k=-1).astype(bool)

    inter_lr = np.zeros_like(mask)
    inter_lr[idx_right, idx_left] = True  # only left→right (upper block)

    global_mask = np.tril(np.ones_like(mask, dtype=bool), k=-1)

    return intra_left, intra_right, inter_lr, global_mask


def summarize_connectivity(con_matrix, masks=None):
    """Summarize connectivity by applying masks."""
    if masks is None:
        masks = make_masks(con_matrix.shape[0])

    intra_left, intra_right, inter_lr, global_mask = masks

    return {
        "intra_left": np.mean(con_matrix[intra_left]),
        "intra_right": np.mean(con_matrix[intra_right]),
        "inter_left_right": np.mean(con_matrix[inter_lr]),
        "global_avg": np.mean(con_matrix[global_mask]),
    }


if __name__ == "__main__":
    raw_file_path = "data\BIDS_CHB-MIT\sub-01\ses-01\eeg\sub-01_ses-01_task-szMonitoring_run-00_eeg.edf"
    raw = mne.io.read_raw_edf(raw_file_path, preload=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=5,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    masks = make_masks()
    for mask in masks:
        plt.imshow(mask)
        plt.yticks(range(0, 18), raw.ch_names)
        plt.xticks(range(0, 18), raw.ch_names, rotation=90)
        plt.show()

    coh = coh_connectivity(epochs[0])
    summary = summarize_connectivity(coh, masks)
    print(summary)
    plt.imshow(coh)
    plt.show()
    fig, ax = plt.subplots(
        figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
    )
    plot_connectivity_circle(
        coh,
        node_names=raw.ch_names,
        title="All-to-All Connectivity (COH)",
        ax=ax,
    )

    plv = plv_connectivity(epochs[0])
    summary = summarize_connectivity(plv, masks)
    print(summary)
    plt.imshow(plv)
    plt.show()
    fig, ax = plt.subplots(
        figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
    )
    plot_connectivity_circle(
        plv,
        node_names=raw.ch_names,
        title="All-to-All Connectivity (PLV)",
        ax=ax,
    )

    aec = aec_connectivity(epochs[0])
    summary = summarize_connectivity(aec, masks)
    print(summary)
    plt.imshow(aec)
    plt.show()
    fig, ax = plt.subplots(
        figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
    )
    plot_connectivity_circle(
        aec,
        node_names=raw.ch_names,
        title="All-to-All Connectivity (AEC)",
        ax=ax,
    )
