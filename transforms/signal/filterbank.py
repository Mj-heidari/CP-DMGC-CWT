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
        normalize_by_lowbands: bool = False,
        scale_dict: Dict[str, float] = None,
    ):
        if band_dict is None:
            band_dict = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 14),
                "beta": (14, 30),
                "gamma": (30, 48),
            }
        
        if scale_dict is None:
            scale_dict = {"beta": 2.0, "gamma": 3.0}

        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict
        self.axis = axis
        self.normalize_by_lowbands = normalize_by_lowbands
        self.scale_dict = scale_dict


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
        band_names = list(self.band_dict.keys())
        band_list = []
        for b, a in self.filters_parameters.values():
            filtered = filtfilt(b, a, eeg, axis=self.axis)
            band_list.append(filtered.astype(np.float32))
            
        bands_out = np.stack(band_list, axis=-2)  # (..., n_bands, n_samples)

        # --- Optional band-based normalization ---
        if self.normalize_by_lowbands:
            if "delta" in band_names and "theta" in band_names:
                # Compute RMS for delta and theta bands
                idx_delta = band_names.index("delta")
                idx_theta = band_names.index("theta")
                delta_power = np.sqrt(np.mean(bands_out[..., idx_delta, :] ** 2, axis=self.axis, keepdims=True))
                theta_power = np.sqrt(np.mean(bands_out[..., idx_theta, :] ** 2, axis=self.axis, keepdims=True))
                ref_power = 0.5 * (delta_power + theta_power) + 1e-8  # avoid division by zero

                # Expand to match (…, n_bands, n_samples)
                while ref_power.ndim < bands_out.ndim:
                    ref_power = np.expand_dims(ref_power, axis=-2)

                # Normalize all bands
                bands_out = bands_out / ref_power

                # Apply optional scaling for higher bands
                for name, scale in self.scale_dict.items():
                    if name in band_names:
                        idx = band_names.index(name)
                        bands_out[..., idx, :] *= scale

        return bands_out

    def __repr__(self):
        return (
            f"FilterBank(fs={self.sampling_rate}, order={self.order}, "
            f"bands={list(self.band_dict.keys())})"
        )


if __name__ == "__main__":
    np.random.seed(0)
    eeg = np.random.randn(8, 128 * 5)  # (channels, samples)
    fb = FilterBank(sampling_rate=128, normalize_by_lowbands=False)
    fb_norm = FilterBank(sampling_rate=128, normalize_by_lowbands=True)

    out_raw = fb(eeg)
    out_norm = fb_norm(eeg)

    print("Raw shape:", out_raw.shape)
    print("Normalized shape:", out_norm.shape)
