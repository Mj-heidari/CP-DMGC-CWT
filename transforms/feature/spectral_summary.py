import numpy as np
from scipy.signal import welch
from .base import FeatureTransform


class SpectralStatTransform(FeatureTransform):
    def __init__(self, sampling_rate=128, nperseg=256):
        self.sampling_rate, self.nperseg = sampling_rate, nperseg

    def opt(self, f, Pxx) -> float:
        raise NotImplementedError

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        out = []
        for ch in eeg:
            f, Pxx = welch(ch, fs=self.sampling_rate, nperseg=self.nperseg)
            out.append([self.opt(f, Pxx)])
        return np.array(out)


class SpectralEntropy(SpectralStatTransform):
    def opt(self, f, Pxx):
        psd_norm = Pxx / (np.sum(Pxx) + 1e-8)
        return float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))


class IntensityWeightedMeanFrequency(SpectralStatTransform):
    def opt(self, f, Pxx):
        return float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-8))


class SpectralEdgeFrequency(SpectralStatTransform):
    def opt(self, f, Pxx):
        cum = np.cumsum(Pxx) / (np.sum(Pxx) + 1e-8)
        return float(f[np.searchsorted(cum, 0.95)])


class PeakFrequency(SpectralStatTransform):
    def opt(self, f, Pxx):
        return float(f[np.argmax(Pxx)])

if __name__ == "__main__":
    eeg = np.random.randn(4, 512)

    transforms = [
        SpectralEntropy(),
        IntensityWeightedMeanFrequency(),
        SpectralEdgeFrequency(),
        PeakFrequency()
    ]

    for t in transforms:
        result = t(eeg)
        print(f"{t.__class__.__name__}: {result.shape}\n{result}")