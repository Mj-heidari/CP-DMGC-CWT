import numpy as np
from .base import FeatureTransform


class BasicStatTransform(FeatureTransform):
    def opt(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([[self.opt(ch)] for ch in eeg])


class MeanAmplitude(BasicStatTransform):
    def opt(self, x): return float(np.mean(x))


class StandardDeviation(BasicStatTransform):
    def opt(self, x): return float(np.std(x))


class Skewness(BasicStatTransform):
    def opt(self, x): return float(((x - np.mean(x))**3).mean() / (np.std(x)**3 + 1e-8))


class Kurtosis(BasicStatTransform):
    def opt(self, x): return float(((x - np.mean(x))**4).mean() / (np.std(x)**4 + 1e-8))


class RootMeanSquare(BasicStatTransform):
    def opt(self, x): return float(np.sqrt(np.mean(x**2)))


class LineLength(BasicStatTransform):
    def opt(self, x): return float(np.sum(np.abs(np.diff(x))))


class ZeroCrossingRate(BasicStatTransform):
    def opt(self, x): return float(np.mean(np.diff(np.sign(x)) != 0))


class HjorthActivity(BasicStatTransform):
    def opt(self, x): return float(np.var(x))


class HjorthMobility(BasicStatTransform):
    def opt(self, x):
        return float(np.sqrt(np.var(np.diff(x)) / (np.var(x) + 1e-8)))


class HjorthComplexity(BasicStatTransform):
    def opt(self, x):
        num = np.sqrt(np.var(np.diff(np.diff(x))) / (np.var(np.diff(x)) + 1e-8))
        den = np.sqrt(np.var(np.diff(x)) / (np.var(x) + 1e-8))
        return float(num / (den + 1e-8))

if __name__ == "__main__":
    eeg = np.random.randn(4, 128)  # 4 channels, 128 samples

    transforms = [
        MeanAmplitude(), StandardDeviation(), Skewness(), Kurtosis(),
        RootMeanSquare(), LineLength(), ZeroCrossingRate(),
        HjorthActivity(), HjorthMobility(), HjorthComplexity()
    ]

    for t in transforms:
        result = t(eeg)
        print(f"{t.__class__.__name__}: {result.shape}\n{result}")