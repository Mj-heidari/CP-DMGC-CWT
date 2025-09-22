import numpy as np


class FeatureTransform:
    """Base class for all feature transforms."""

    def __call__(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return self.apply(eeg, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def aggregate(self, features: np.ndarray, agg_fn=np.mean, axis=0) -> np.ndarray:
        """
        Aggregate features across channels.
        Default is mean over axis=0 (channels).
        """
        return agg_fn(features, axis=axis, keepdims=True)

if __name__ == "__main__":
    from base import FeatureTransform

    class Dummy(FeatureTransform):
        def __call__(self, eeg, **kwargs):
            return eeg

    eeg = np.random.randn(2, 100)
    print("FeatureTransform subclass works:", Dummy()(eeg).shape)