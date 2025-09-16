from ..base import BaseTransform

class SignalTransform(BaseTransform):
    """Signal → Signal transforms (filtering, noise, fft, etc.)."""
    def __call__(self, eeg, **kwargs):
        raise NotImplementedError