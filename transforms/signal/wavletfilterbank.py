import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class WaveletFilterBank(nn.Module):
    """
    Daubechies-4 Wavelet Filter Bank implemented using torch convolutions
    (same structure as MultiBandSpectralConv). Keeps output length equal to input.
    
    combine_mode: 'upsample' | 'concat_time' | 'mean_pool'
    """

    def __init__(self, fs: int, combine_mode: str = "concat_time"):
        super().__init__()
        assert combine_mode in ["upsample", "concat_time", "mean_pool"]
        self.fs = fs
        self.combine_mode = combine_mode
        self.levels = max(1, math.floor(math.log2(fs)) - 3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Daubechies-4 coefficients
        h = torch.tensor([
            -0.010597401785069032,
             0.032883011666982945,
             0.030841381835560763,
            -0.18703481171888114,
            -0.02798376941698385,
             0.6308807679295904,
             0.7148465705529154,
             0.2303778133088964
        ], dtype=torch.float32).to(self.device)

        g = torch.tensor([( (-1)**k ) * v for k, v in enumerate(h.flip(0))], dtype=torch.float32).to(self.device)

        self.register_buffer("db4_low", h)
        self.register_buffer("db4_high", g)
        

    def _conv_groups(self, x, filt):
        """Apply depthwise conv1d with reflect padding, keeping signal length."""
        B, C, S = x.shape
        K = filt.shape[0]
        w = filt.view(1, 1, K).repeat(C, 1, 1)
        x_p = F.pad(x, (K - 1, 0), mode='reflect')
        out = F.conv1d(x_p, w, bias=None, stride=1, groups=C)
        return out  # (B, C, S)

    def forward(self, eeg: np.ndarray) -> torch.Tensor:
        # eeg: (B, C, S) or (C, S)
        with torch.no_grad():
            eeg = torch.tensor(eeg, dtype=torch.float).to(self.device)
            is_batched = True
            if eeg.ndim == 2:  # (C, S) unbatched
                eeg = eeg.unsqueeze(0)  # add batch dim
                is_batched = False

            B, C, S = eeg.shape

            approx = eeg
            details = []
            approximations = []

            for _ in range(1, self.levels + 1):
                low = self._conv_groups(approx, self.db4_low)
                high = self._conv_groups(approx, self.db4_high)

                low_ds = low[..., ::2]
                high_ds = high[..., ::2]

                details.append(high_ds)
                approximations.append(low_ds)
                approx = low_ds

            # collect all wavelet components
            components = details[-self.levels:] + [approximations[-1]] 

            # combine
            if self.combine_mode == "upsample":
                components = [F.interpolate(b, size=S, mode='linear', align_corners=False).unsqueeze(-2) for b in components]
                out = torch.cat(components, dim=-2)
            elif self.combine_mode == "concat_time":
                out = torch.cat(components, dim=2)
            elif self.combine_mode == "mean_pool":
                min_len = min(b.shape[-1] for b in components)
                comps = [F.interpolate(b, min_len, mode='linear').unsqueeze(-2) for b in components]
                out = torch.cat(comps, dim=-2)
            else:
                raise ValueError(f"Invalid combine_mode: {self.combine_mode}")

            
            if not is_batched:
                out = out.cpu().squeeze(0)  # remove batch dim if input was unbatched

            return out.cpu()


if __name__ == "__main__":
    # Example: batch of EEG-like signals
    x = np.random.randn(8, 640)  # (batch, channels, time)

    # Create transform
    transform = WaveletFilterBank(fs=128, combine_mode="concat_time").cuda()

    # Apply
    y = transform(x)
    print(y.shape)  # depends on mode