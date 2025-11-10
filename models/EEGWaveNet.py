import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletBranch(nn.Module):
    """Processes a single wavelet component (detail or approximation)."""
    def __init__(self, n_channels, kernel_t_len, n_output_features=16, pool_kernel=4, pool_stride=4):
        super().__init__()
        self.temp_conv = nn.Conv2d(
            1, 16, kernel_size=(1, kernel_t_len), padding=(0, kernel_t_len//2), bias=False
        )
        self.channel_conv = nn.Conv2d(
            16, n_output_features, kernel_size=(n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_output_features)
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, C, T)
        """
        x = x.unsqueeze(1)                   # (B, 1, C, T)
        x = self.temp_conv(x)                # (B, 16, C, T)
        x = self.channel_conv(x)             # (B, n_output_features, 1, T)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)                     # (B, n_output_features, T_out)
        return x


class EEGWaveletEmbeddingNet(nn.Module):
    """Handles multi-resolution wavelet components and fuses their embeddings."""
    def __init__(self, n_channels=18, component_lengths=(320, 160, 80, 40, 40),
                 kernel_t_lens=None, n_output_features=16):
        super().__init__()
        self.n_channels = n_channels
        self.component_lengths = component_lengths
        self.n_components = len(component_lengths)

        if kernel_t_lens is None:
            kernel_t_lens = [7] * self.n_components

        assert len(kernel_t_lens) == self.n_components

        # One branch per wavelet component
        self.branches = nn.ModuleList([
            WaveletBranch(n_channels, kernel_t_len=kernel_t_lens[i],
                          n_output_features=n_output_features)
            for i in range(self.n_components)
        ])

        # Fuse features across scales (time dimension collapsed)
        self.fuse_conv = nn.Conv2d(n_output_features, 32, kernel_size=(1, self.n_components))

    def forward(self, x):
        """
        x: concatenated components (B, C, sum(T_i))
        """
        B, C, total_T = x.shape
        assert total_T == sum(self.component_lengths), \
            f"Expected total time {sum(self.component_lengths)}, got {total_T}"

        # Split input into wavelet components
        splits = torch.split(x, self.component_lengths, dim=2)
        branch_feats = []

        for i, branch in enumerate(self.branches):
            f = branch(splits[i])           # (B, n_output_features, T_out)
            branch_feats.append(f)

        # Match temporal length across all components
        target_T = min(f.shape[-1] for f in branch_feats)
        pooled_feats = [F.adaptive_avg_pool1d(f, target_T) for f in branch_feats]

        # Stack them along a new "component" dimension
        feats = torch.stack(pooled_feats, dim=-1)     # (B, n_output_features, T_out, n_components)

        # Fuse across components (treat each as a "frequency channel")
        fused = self.fuse_conv(feats)                 # (B, 32, T_out, 1)
        fused = fused.mean(dim=2).squeeze(-1)         # (B, 32)
        return fused

class EEGWaveNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.embedding = EEGWaveletEmbeddingNet(
            n_channels=18,
            component_lengths=(320, 160, 80, 40, 40),
            n_output_features=16
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(32),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        return self.classifier(emb)


if __name__ == "__main__":

    from network_debugger import NetworkDebugger
    from torchinfo import summary

    torch.manual_seed(0)
    model = EEGWaveNet().cuda()

    debugger = NetworkDebugger(model)
    debugger.register_hooks()

    # Fake EEG input: (B, C, Bn, T)
    x = torch.randn(2, 18, 640).cuda()
    y = model(x)

    print("\nOutput shape:", y.shape)

    # Backward pass
    target = torch.randint(0, 2, (2,)).cuda()
    loss = F.cross_entropy(y, target)
    loss.backward()

    debugger.remove_hooks()

    summary(model, (2, 18, 640))
