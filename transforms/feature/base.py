import numpy as np
from ..base import BaseTransform

class FeatureTransform(BaseTransform):
    """Signal → Feature transforms."""
    def __call__(self, eeg, **kwargs):
        return self.apply(eeg, **kwargs)

    def apply(self, eeg, **kwargs):
        raise NotImplementedError

    def aggregate(self, features: np.ndarray, agg_fn=np.mean, axis=0) -> np.ndarray:
        """
        Aggregate features across channels/trials.
        Default: mean over channels.
        """
        return agg_fn(features, axis=axis)