import numpy as np
from scipy.signal import welch
from .base import FeatureTransform


class BandPowerTransform(FeatureTransform):
    def __init__(self, fmin, fmax, sampling_rate=128, nperseg=256):
        self.fmin, self.fmax = fmin, fmax
        self.sampling_rate, self.nperseg = sampling_rate, nperseg

    def opt(self, f, Pxx) -> float:
        idx = np.logical_and(f >= self.fmin, f <= self.fmax)
        return float(np.trapz(Pxx[idx], f[idx]))

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        out = []
        for ch in eeg:
            f, Pxx = welch(ch, fs=self.sampling_rate, nperseg=self.nperseg)
            out.append([self.opt(f, Pxx)])
        return np.array(out)


class DeltaPower(BandPowerTransform):
    def __init__(self, sampling_rate=128): super().__init__(0.5, 4, sampling_rate)


class ThetaPower(BandPowerTransform):
    def __init__(self, sampling_rate=128): super().__init__(4, 8, sampling_rate)


class AlphaPower(BandPowerTransform):
    def __init__(self, sampling_rate=128): super().__init__(8, 13, sampling_rate)


class BetaPower(BandPowerTransform):
    def __init__(self, sampling_rate=128): super().__init__(13, 30, sampling_rate)


class GammaPower(BandPowerTransform):
    def __init__(self, sampling_rate=128): super().__init__(30, 45, sampling_rate)

if __name__ == "__main__":
    eeg = np.random.randn(4, 512)  # 4 channels, 512 samples

    transforms = [
        DeltaPower(), ThetaPower(), AlphaPower(), BetaPower(), GammaPower()
    ]

    for t in transforms:
        result = t(eeg)
        print(f"{t.__class__.__name__}: {result.shape}\n{result}")