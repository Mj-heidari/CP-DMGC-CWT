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
    epoch_band = epoch.copy().filter(fmin, fmax, verbose=False).apply_hilbert(envelope=True)
    data = epoch_band.get_data()
    aec = envelope_correlation(data, orthogonalize="pairwise")
    return aec.get_data(output="dense").squeeze()


if __name__ == "__main__":
    raw_file_path = "data\BIDS_CHB-MIT\sub-01\ses-01\eeg\sub-01_ses-01_task-szMonitoring_run-00_eeg.edf"
    raw = mne.io.read_raw_edf(raw_file_path, preload=True, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=5,
        overlap=0.0,
        preload=True,
        reject_by_annotation=True, 
        verbose= False
    )

    coh = coh_connectivity(epochs[0])
    fig, ax = plt.subplots(
        figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
    )
    plot_connectivity_circle(
        coh,
        node_names=[str(i) for i in range(18)],
        title="All-to-All Connectivity Condition (COH)",
        ax=ax,
    )

    plv = plv_connectivity(epochs[0])
    fig, ax = plt.subplots(
        figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
    )
    plot_connectivity_circle(
        plv,
        node_names=[str(i) for i in range(18)],
        title="All-to-All Connectivity Condition (PLV)",
        ax=ax,
    )

    aec = aec_connectivity(epochs[0])
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    plot_connectivity_circle(
        aec,
        node_names= [str(i) for i in range(18)],
        title="All-to-All Connectivity Condition (AEC)",
        ax=ax,
    )
