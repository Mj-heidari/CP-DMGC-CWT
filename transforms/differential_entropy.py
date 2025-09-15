from typing import Dict, Tuple
from scipy.signal import butter, lfilter
import numpy as np


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


class BandTransform:
    def __init__(
        self,
        sampling_rate: int = 128,
        order: int = 5,
        band_dict: Dict[str, Tuple[int, int]] = {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 14],
            "beta": [14, 31],
            "gamma": [31, 49],
        },
    ):
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict
        self.filters_parameters = {
            key: butter_bandpass(low, high, self.sampling_rate, self.order)
            for key, (low, high) in band_dict.items()
        }

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        band_list = []
        for b, a in self.filters_parameters.values():
            filtered_eeg = lfilter(b, a, eeg)
            band_list.append(self.opt(filtered_eeg))
        return np.stack(band_list, axis=-1)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body,
            **{
                "sampling_rate": self.sampling_rate,
                "order": self.order,
                "band_dict": {...},
            },
        )


class BandDifferentialEntropy(BandTransform):
    r"""
    A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input. It is a widely accepted differential entropy calculation method by the community, which is often applied to the DEAP and DREAMER datasets. It is relatively easy to understand and has a smaller scale and more gradual changes than the :obj:`BandDifferentialEntropyV1` calculated based on average power spectral density.

    - Related Paper: Fdez J, Guttenberg N, Witkowski O, et al. Cross-subject EEG-based emotion recognition through neural networks with stratified normalization[J]. Frontiers in neuroscience, 2021, 15: 626277.
    - Related Project: https://github.com/javiferfer/cross-subject-eeg-emotion-recognition-through-nn/

    - Related Paper: Li D, Xie L, Chai B, et al. Spatial-frequency convolutional self-attention network for EEG emotion recognition[J]. Applied Soft Computing, 2022, 122: 108740.
    - Related Project: https://github.com/qeebeast7/SFCSAN/

    """

    def __call__(self, *args, eeg: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        r"""
        Args:
            eeg (np.ndarray): The input EEG signals in shape of [number of electrodes, number of data points].
        Returns:
            np.ndarray[number of electrodes, number of sub-bands]: The differential entropy of several sub-bands for all electrodes.
        """
        return super().__call__(*args, eeg=eeg, **kwargs)

    def opt(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        return 0.5 * np.log2(2 * np.pi * np.e * np.var(eeg, axis= -1))

if __name__ == "__main__":
    transform = BandDifferentialEntropy(sampling_rate=128, order=5)
    import time
    start = time.time()
    for i in range(100):
        eeg = np.random.randn(18,128*5)
        result = transform(eeg=eeg)
    end = time.time()
    print("elapsed time:", (end-start)*1000)
    print(result.shape)