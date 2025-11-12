import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mne

from captum.attr import IntegratedGradients
from models.EEGWaveNet import EEGWaveNet
from dataset.dataset import UnderSampledDataLoader, CHBMITDataset, make_cv_splitter
from transforms.signal.wavletfilterbank import WaveletFilterBank


# ---------------- Prepare dataset ----------------
filter_bank = WaveletFilterBank(fs=128, combine_mode="concat_time")
offline_transforms = [filter_bank]

dataset = CHBMITDataset(
    "data/BIDS_CHB-MIT",
    use_uint16=True,
    offline_transforms=[],
    online_transforms=offline_transforms,
    suffix="zscore_F_T",
    subject_id="01"
)

train_val_dataset, test_dataset = next(
    make_cv_splitter(dataset, "leave_one_preictal", method="balanced_shuffled")
)
dataloader = UnderSampledDataLoader(test_dataset, batch_size=320)

all_samples, all_labels = next(iter(dataloader))
all_samples = all_samples.to(torch.float32)
all_labels = all_labels.numpy()

# ---------------- Load model ----------------
run_dir = "runs/run305_20251112_115427"
model = EEGWaveNet()
model.load_state_dict(torch.load(run_dir + '/checkpoints/best_model_outer1_inner1.pth', map_location='cpu'))
model.eval()

# ---------------- Integrated Gradients ----------------
component_lengths = [320, 160, 80, 40, 40]
n_components = len(component_lengths)

# Pick a sample to explain

# Define a wrapper for model output we want to explain
def model_forward(x):
    # return probabilities for class 1 (or logits if you prefer)
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    return probs[:, 1]  # explain class 1

# Initialize Integrated Gradients
ig = IntegratedGradients(model_forward)

# Compute attributions
class0_attr = []
class1_attr = []

for i in range(len(all_samples)):
    x = all_samples[i:i+1].requires_grad_(True)
    label = all_labels[i]

    attr, _ = ig.attribute(x, baselines=torch.zeros_like(x), return_convergence_delta=True)
    attr = attr.detach().cpu().numpy()[0]  # (C, T)

    if label == 0:
        class0_attr.append(attr)
    else:
        class1_attr.append(attr)


class0_attr = np.mean(np.stack(class0_attr), axis=0) if class0_attr else None
class1_attr = np.mean(np.stack(class1_attr), axis=0) if class1_attr else None

def normalize_attr(attr):
    attr = np.abs(attr)
    attr /= attr.max() + 1e-8
    return attr


if class0_attr is not None:
    class0_attr = normalize_attr(class0_attr)
if class1_attr is not None:
    class1_attr = normalize_attr(class1_attr)

# class1_attr = (class1_attr + class0_attr) / 2
class1_attr = class0_attr
# ---------------- Aggregate per wavelet component ----------------
component_means = []
component_stds = []
start = 0

for length in component_lengths:
    end = start + length
    # mean and std over channels and time within this component
    comp_attr = np.abs(class1_attr[:, start:end])
    component_means.append(np.mean(comp_attr))
    component_stds.append(np.std(comp_attr))
    start = end

components = ["D1", "D2", "D3", "D4", "A4"]

# ---------------- Plot ----------------
plt.figure(figsize=(6, 4))
plt.bar(components, component_means, yerr=component_stds, capsize=5, color="skyblue", edgecolor="black")
plt.title("Integrated Gradients per Wavelet Component (Class 1)")
plt.xlabel("Wavelet Component")
plt.ylabel("Mean ± Std of |Attribution|")
plt.tight_layout()
plt.show()

print("Component means:", component_means)
print("Component stds:", component_stds)

# ---------------- Aggregate per wavelet component and channel ----------------
channel_component_importances = []  # will be (C, n_components)
start = 0
for length in component_lengths:
    end = start + length
    # mean over time only, keep channels separate
    comp_attr_per_channel = np.mean(np.abs(class1_attr[:, start:end]), axis=1)  # shape: (C,)
    channel_component_importances.append(comp_attr_per_channel)
    start = end

# Convert to numpy array: shape (C, n_components)
channel_component_importances = np.stack(channel_component_importances, axis=1)  # (C, n_components)

# ---------------- Plot per-channel contributions ----------------
channels = [
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2"
]

fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(channel_component_importances, aspect='auto', cmap='viridis')
ax.set_xticks(np.arange(n_components))
ax.set_xticklabels(components)
ax.set_yticks(np.arange(len(channels)))
ax.set_yticklabels(channels)
ax.set_xlabel("Wavelet Component")
ax.set_ylabel("Channel")
ax.set_title("Integrated Gradients: Channel × Component")
fig.colorbar(im, ax=ax, label="Mean |Attribution|")
plt.show()


# ---------------- Prepare channel info ----------------
# Manually add remaining 6 channels with offsets
positions= np.array([[3.13782141e-01, 6.78939002e-01],
 [2.84336102e-01, 4.75428638e-01],
 [2.77791973e-01, 2.60762849e-01],
 [3.14951391e-01, 4.61489878e-02],
 [2.30150900e-01, 7.13858956e-01],
 [9.31017962e-02, 4.73175785e-01],
 [1.05112229e-01, 1.99319851e-01],
 [2.47329596e-01, 1.38777878e-17],
 [4.72116339e-01, 4.79726557e-01],
 [4.72157148e-01, 2.75127744e-01],
 [6.33446501e-01, 6.85022078e-01],
 [6.64730884e-01, 4.79078388e-01],
 [6.72248071e-01, 2.61854146e-01],
 [6.32955694e-01, 4.73929673e-02],
 [7.19980886e-01, 7.21167564e-01],
 [8.58418221e-01, 4.80440936e-01],
 [8.40870463e-01, 2.02433784e-01],
 [6.97209315e-01, 1.38196188e-03]])

# ---------------- Normalize positions to [-1,1] ----------------
pos_min = positions.min(axis=0)
pos_max = positions.max(axis=0)
positions_norm = 2 * (positions - pos_min) / (pos_max - pos_min) - 1
positions_norm = positions_norm / 14

channel_attr = np.mean(np.abs(class1_attr[:,0:320]), axis=1)  # mean over time per channel
mne.viz.plot_topomap(channel_attr, pos=positions_norm, names=channels, outlines='head', size=6)
plt.title("Integrated Gradients per EEG Channel (Conceptual Layout)")

channel_attr = np.mean(np.abs(class1_attr[:,320+180:]), axis=1)  # mean over time per channel
mne.viz.plot_topomap(channel_attr, pos=positions_norm, names=channels, outlines='head', size=6)
plt.title("Integrated Gradients per EEG Channel (Conceptual Layout)")