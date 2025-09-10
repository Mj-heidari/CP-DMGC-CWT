import mne
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def read_raw_with_locs(edf_file_path):
    """
    Read raw EEG file and add locations to channel based on 10-20 systems.
    since the channels are bipolar their locations are set to the middle of two endpoints.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]

    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    pairs: list[tuple[str, str]] = [ch.split("-") for ch in raw.ch_names]

    for ch_name, (anode, cathode) in zip(raw.ch_names, pairs):
        anode = anode.capitalize()
        cathode = cathode.capitalize()
        if anode in pos_dict and cathode in pos_dict:
            pa, pc = pos_dict[anode], pos_dict[cathode]
            midpoint = (pa + pc) / 2.0
            ch_idx = raw.ch_names.index(ch_name)
            raw.info["chs"][ch_idx]["loc"][:3] = midpoint

    positions_3d = np.array([ch["loc"][:3] for ch in raw.info["chs"]])

    return raw, positions_3d


def euclidean_dist(positions):
    """compute Euclidean distance matrix"""
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix


def inverse_mean_threshold_adjacency(dist_matrix):
    """
    Build adjacency matrix using inverse mean threshold rule:
    If distance < mean distance for a node, connect with weight 1/distance.
    Based on: "Dynamic Multi-Graph Convolution-Based Channel-Weighted Transformer
    Feature Fusion Network for Epileptic Seizure Prediction"
    """
    N = dist_matrix.shape[0]
    adj_imt = np.zeros((N, N))

    for i in range(N):
        U = dist_matrix[i]
        mean = np.mean(np.delete(U, i))  # exclude self-distance
        for j in range(N):
            if i != j and U[j] < mean:
                adj_imt[i, j] = 1.0 / (U[j] + 1e-6)
        adj_imt[i, i] = np.mean((U < mean) * (U))

    # Make symmetric
    adj_imt = np.maximum(adj_imt, adj_imt.T)
    return adj_imt


def gaussian_kernel_adjacency(dist_matrix):
    """Gaussian kernel (weights closer nodes higher)"""
    sigma = np.mean(dist_matrix)  # scale parameter, tune this
    adj_matrix = np.exp(-(dist_matrix**2) / (2 * sigma**2))
    return adj_matrix


def knn_adjacency(dist_matrix, k=4):
    adj_knn = np.zeros_like(dist_matrix)
    for i in range(dist_matrix.shape[0]):
        idx = np.argsort(dist_matrix[i])[1 : k + 1]  # skip self (index 0)
        adj_knn[i, idx] = 1
        adj_knn[idx, i] = 1
    return adj_knn


def mne_adjacency(raw):
    adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type="eeg")
    return adjacency.toarray()


def get_2d_positions(raw):
    """
    Get 2D positions for channels using MNE's layout projection.
    This matches what raw.plot_sensors(show_names=True) uses.
    """
    layout = mne.channels.make_eeg_layout(raw.info)
    pos_dict = dict(zip(layout.names, layout.pos[:, :2]))
    # Match raw channel order
    pos_2d = np.array([pos_dict[ch] for ch in raw.ch_names if ch in pos_dict])
    return pos_2d


def save_graph(adj_matrix, raw, out_file):
    """
    Save EEG adjacency matrix as a graph with true node positions.
    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix (n_channels x n_channels).
    raw : mne.io.Raw
        Raw object containing channel names.
    positions_3d : np.ndarray
        Channel positions (n_channels x 3).
    out_file : str
        Path to save the figure (e.g. 'graph.png').
    """
    # Project 3D positions to 2D (x, y)
    pos_2d = get_2d_positions(raw)

    # Build graph
    G = nx.Graph()
    for i, ch in enumerate(raw.ch_names):
        G.add_node(ch, pos=pos_2d[i])

    n_nodes = len(raw.ch_names)
    edge_widths = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj_matrix[i, j] > 0:  # edge exists
                G.add_edge(raw.ch_names[i], raw.ch_names[j], weight=adj_matrix[i, j])
                edge_widths.append(adj_matrix[i, j])

    # Plot
    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos=nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color="gray",
        width=edge_widths,
        font_size=8,
    )
    plt.axis("equal")
    plt.title("EEG Graph")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    raw_file_path = "data\BIDS_CHB-MIT\sub-01\ses-01\eeg\sub-01_ses-01_task-szMonitoring_run-00_eeg.edf"
    # Read eeg file and assign location to each channel
    raw, positions_3d = read_raw_with_locs(edf_file_path=raw_file_path)
    raw.plot_sensors(show_names=True)
    plt.savefig("./dataset/output/channels_locations.png")

    # Calculate the distance matrix based on the channels' locations
    dist_matrix = euclidean_dist(positions_3d)
    plt.imsave("./dataset/output/distance_matrix.png", dist_matrix)

    # Calculate different adjacency matrix
    guassian_adj = gaussian_kernel_adjacency(dist_matrix)
    plt.imsave("./dataset/output/gaussian_adjacency.png", guassian_adj)
    save_graph(guassian_adj, raw, "./dataset/output/gaussian_graph.png")

    knn_adj = knn_adjacency(dist_matrix)
    plt.imsave("./dataset/output/knn_adjacency_k4.png", knn_adj)
    save_graph(knn_adj, raw, "./dataset/output/knn_graph.png")

    mne_adj = mne_adjacency(raw)
    plt.imsave("./dataset/output/mne_adjacency.png", mne_adj)
    save_graph(mne_adj, raw, "./dataset/output/mne_graph.png")

    imt_adj = inverse_mean_threshold_adjacency(dist_matrix)
    plt.imsave("./dataset/output/imt_adjacency.png", mne_adj)
    save_graph(mne_adj, raw, "./dataset/output/imt_graph.png")
