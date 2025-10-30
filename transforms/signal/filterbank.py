from typing import Dict, Tuple
from scipy.signal import butter, filtfilt
import numpy as np

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


class FilterBank:
    def __init__(
        self,
        sampling_rate: int = 128,
        order: int = 4,
        band_dict: Dict[str, Tuple[float, float]] = None,
        axis: int = -1,  # which axis is time
    ):
        if band_dict is None:
            band_dict = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 14),
                "beta": (14, 30),
                "gamma": (30, 48),
            }

        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict
        self.axis = axis

        # Precompute filter coefficients
        self.filters_parameters = {
            name: butter_bandpass(low, high, sampling_rate, order)
            for name, (low, high) in band_dict.items()
        }

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        return self.apply(eeg)

    def apply(self, eeg: np.ndarray) -> np.ndarray:
        """Apply all band-pass filters to EEG array.
        Args:
            eeg: np.ndarray, shape (..., n_samples)
        Returns:
            np.ndarray, shape (..., n_bands, n_samples)
        """
        band_list = []
        for b, a in self.filters_parameters.values():
            filtered = filtfilt(b, a, eeg, axis=self.axis)
            band_list.append(filtered.astype(np.float32))
        return np.stack(band_list, axis=-2)  # add band dimension before time

    def __repr__(self):
        return (
            f"FilterBank(fs={self.sampling_rate}, order={self.order}, "
            f"bands={list(self.band_dict.keys())})"
        )


if __name__ == "__main__":
    eeg = np.random.randn(8, 128 * 5)  # (channels, samples)
    fb = FilterBank(sampling_rate=128)
    out = fb(eeg)
    print("Output shape:", out.shape)  # (8, 5, 640)
