import numpy as np
from .base import FeatureTransform


class ConnectivityTransform(FeatureTransform):
    def opt(self, corr: np.ndarray) -> float:
        raise NotImplementedError

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        corr = np.corrcoef(eeg)
        val = self.opt(corr)
        return np.full((eeg.shape[0], 1), val)


class MeanAbsCorrelation(ConnectivityTransform):
    def opt(self, corr):
        n = corr.shape[0]
        iu = np.triu_indices(n, k=1)
        return float(np.mean(np.abs(corr[iu])))
