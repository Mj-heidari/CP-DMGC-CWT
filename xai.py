import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import mne
from tqdm import tqdm

from captum.attr import IntegratedGradients
from models.EEGWaveNet import EEGWaveNet
from dataset.dataset import UnderSampledDataLoader, CHBMITDataset, make_cv_splitter
from transforms.signal.wavletfilterbank import WaveletFilterBank


class XAI_Analyzer:
    def __init__(self, run_dir: Path):
        self.component_lengths = [320, 160, 80, 40, 40]
        self.components = ["D1", "D2", "D3", "D4", "A4"]
        self.n_components = len(self.component_lengths)
        self.dir = run_dir
        self.subj_id = self.get_subj_id()
        self.splits = self.prepare_subject_dataset(self.subj_id)

        self.channels = [
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FZ-CZ", "CZ-PZ",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2"
        ]

        positions = np.array([
            [3.13782141e-01, 6.78939002e-01],
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
            [6.97209315e-01, 1.38196188e-03]
        ])
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        positions_norm = 2 * (positions - pos_min) / (pos_max - pos_min) - 1
        self.positions_norm = positions_norm / 14

    def get_subj_id(self):
        with open(self.dir / "config.json", "r") as f:
            config_data = json.load(f)
        return config_data["arguments"]["subject_id"]

    def prepare_subject_dataset(self, subj_id):
        filter_bank = WaveletFilterBank(fs=128, combine_mode="concat_time")
        offline_transforms = [filter_bank]

        dataset = CHBMITDataset(
            "data/BIDS_CHB-MIT",
            use_uint16=True,
            offline_transforms=[],
            online_transforms=offline_transforms,
            suffix="zscore_F_T",
            subject_id=subj_id,
        )

        splits = make_cv_splitter(dataset, "leave_one_preictal", method="balanced_shuffled")
        return splits

    def load_model(self,fold):
        model = EEGWaveNet()
        model_path = self.dir / f"checkpoints/best_model_outer{fold}_inner1.pth"
        model.load_state_dict(torch.load(model_path, map_location="cuda"))
        model.eval()
        return model.cuda()

    def compute_intergrated_gradient(self, normalize=True):
        attrs = []
        for fold, (_, test_dataset) in enumerate(self.splits):
            try:
                self.model = self.load_model(fold+1)
            except Exception as e:
                print("error loading model:", e)
                break
            
            def model_forward(x):
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                return probs[:, 1]  # class 1 probability

            ig = IntegratedGradients(model_forward)
            dataloader = UnderSampledDataLoader(test_dataset, batch_size=64)

            for X, y in tqdm(dataloader, desc=f"Fold {fold+1} IG Computation"):
                X = X.to(torch.float32).cuda()
                y = y.cuda()
                interictal_segments = X[y==0]

                for i in range(len(X)):
                    if y[i] == 0:
                        continue  # only compute IG for class 1

                    j = torch.randint(len(interictal_segments), (1,)).item()
                    x = X[i:i+1].requires_grad_(True)
                    baseline = interictal_segments[j:j+1]

                    attr, _ = ig.attribute(
                        x, baselines=baseline, return_convergence_delta=True
                    )
                    attrs.append(attr[0].detach().cpu().numpy())
    
        attrs = np.array(attrs)

        start = 0
        component_attrs = []
        for length in self.component_lengths:
            end = start + length
            component = attrs[:, :, start:end]
            component_attrs.append(np.mean(component, axis=(0, 2)))  # (C,)
            start = end

        component_attrs = np.array(component_attrs)  # (n_components, C)
        component_attrs = np.abs(component_attrs)
        component_mean_attr = np.transpose(component_attrs, (1, 0))  # (C, n_components)  

        if normalize:
            component_mean_attr = ((component_mean_attr - component_mean_attr.min()) 
                                   / (component_mean_attr.max() - component_mean_attr.min() + 1e-8))

        return component_mean_attr

    def plot_band_attribution(self, components_mean):
        plt.figure(figsize=(6, 4))
        plt.bar(self.components, components_mean.mean(axis=0))
        plt.title("Integrated Gradients per Wavelet Component")
        plt.xlabel("Wavelet Component")
        plt.ylabel("Attribution")
        plt.tight_layout()
        plt.savefig(self.dir / "band_attributes.png", dpi=300)
        plt.close()

    def plot_pre_channel_band_attribution(self, components_mean):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(components_mean, aspect="auto", cmap="viridis")
        ax.set_xticks(range(self.n_components))
        ax.set_xticklabels(self.components)
        ax.set_yticks(range(len(self.channels)))
        ax.set_yticklabels(self.channels)
        ax.set_title("Channel × Component Attribution")
        fig.colorbar(im, ax=ax)
        plt.savefig(self.dir / "band_channels_attributions.png", dpi=300)
        plt.close()

    def plot_topo_band_attribution(self, components_mean):
        # ---------------- LOW FREQUENCY ----------------
        low_attr = np.mean(np.abs(components_mean[:, 2:]), axis=1)

        fig, ax = plt.subplots(figsize=(6, 6))
        im, cn = mne.viz.plot_topomap(
            low_attr,
            pos=self.positions_norm,
            names=self.channels,
            outlines="head",
            cmap='jet',
            axes=ax,
            show=False
        )
        ax.set_title("Topomap Low-frequency")
        fig.savefig(self.dir / "topograph_low_freq.png", dpi=300)
        plt.close(fig)

        # ---------------- HIGH FREQUENCY ----------------
        high_attr = np.mean(np.abs(components_mean[:, :2]), axis=1)

        fig, ax = plt.subplots(figsize=(6, 6))
        im, cn = mne.viz.plot_topomap(
            high_attr,
            pos=self.positions_norm,
            names=self.channels,
            outlines="head",
            cmap='jet',
            axes=ax,
            show=False
        )
        ax.set_title("Topomap High-frequency")
        fig.savefig(self.dir / "topograph_high_freq.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Path to runs directory")
    args = parser.parse_args()

    root_dir = Path(args.root)

    # loop over run folders
    for run_dir in root_dir.iterdir():
        if not run_dir.is_dir():
            continue

        print(f"Processing: {run_dir}")

        analyzer = XAI_Analyzer(run_dir)
        components_means = analyzer.compute_intergrated_gradient()

        analyzer.plot_band_attribution(components_means)
        analyzer.plot_pre_channel_band_attribution(components_means)
        analyzer.plot_topo_band_attribution(components_means)
        