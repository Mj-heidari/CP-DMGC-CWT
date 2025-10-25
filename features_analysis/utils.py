import numpy as np
from tqdm import tqdm

# --- Imports from the main project structure ---
# Note: These will work when the main script handles the system path.
from transforms.feature.base import FeatureTransform
from transforms.feature.basic_stats import (
    MeanAmplitude, StandardDeviation, Skewness, Kurtosis,
    RootMeanSquare, LineLength, ZeroCrossingRate,
    HjorthActivity, HjorthMobility, HjorthComplexity,
)
from transforms.feature.band_power import (
    BandPowerTransform, DeltaPower, ThetaPower, AlphaPower, BetaPower, GammaPower,
)
from transforms.feature.spectral_summary import (
    SpectralEntropy, IntensityWeightedMeanFrequency,
    SpectralEdgeFrequency, PeakFrequency,
)
from transforms.feature.connectivity import MeanAbsCorrelation, MeanCoh, MeanPLV, MeanImCoh, MeanPLI, MeanWPLI
from transforms.feature.differential_entropy import BandDifferentialEntropy
from dataset.utils import invert_uint16_scaling


# --- Custom Features to reach 30+ ---
class LowAlphaPower(BandPowerTransform):
    """Custom feature for power in the low alpha band."""
    def __init__(self, sampling_rate=128): super().__init__(8, 10, sampling_rate)

class HighAlphaPower(BandPowerTransform):
    """Custom feature for power in the high alpha band."""
    def __init__(self, sampling_rate=128): super().__init__(10, 13, sampling_rate)

class LowBetaPower(BandPowerTransform):
    """Custom feature for power in the low beta band."""
    def __init__(self, sampling_rate=128): super().__init__(13, 20, sampling_rate)

class HighBetaPower(BandPowerTransform):
    """Custom feature for power in the high beta band."""
    def __init__(self, sampling_rate=128): super().__init__(20, 30, sampling_rate)

class AlphaBetaRatio(FeatureTransform):
    """Custom feature for the ratio of alpha to beta band power."""
    def __init__(self, sampling_rate=128):
        self.alpha_power = AlphaPower(sampling_rate)
        self.beta_power = BetaPower(sampling_rate)

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        alpha = self.alpha_power.apply(eeg, **kwargs)
        beta = self.beta_power.apply(eeg, **kwargs)
        ratio = alpha / (beta + 1e-8)
        return ratio


def get_feature_extractors():
    """Returns a list of feature extractor instances and their names."""
    # Add the new MNE-based connectivity features here
    features_to_extract = [
        MeanAmplitude(), StandardDeviation(), Skewness(), Kurtosis(),
        RootMeanSquare(), LineLength(), ZeroCrossingRate(),
        HjorthActivity(), HjorthMobility(), HjorthComplexity(),
        DeltaPower(), ThetaPower(), AlphaPower(), BetaPower(), GammaPower(),
        LowAlphaPower(), HighAlphaPower(), LowBetaPower(), HighBetaPower(),
        AlphaBetaRatio(),
        SpectralEntropy(), IntensityWeightedMeanFrequency(),
        SpectralEdgeFrequency(), PeakFrequency(),
        MeanAbsCorrelation(), # Keep original corrcoef version
        MeanCoh(), # Add MNE Coherence
        MeanPLV(), # Add MNE PLV
        MeanImCoh(),# Add MNE ImCoh
        MeanPLI(), # Add MNE PLI
        MeanWPLI(),# Add MNE WPLI
        BandDifferentialEntropy() # Keep BDE last
    ]
    bde = BandDifferentialEntropy()
    # Update feature_names generation to match the extended list
    feature_names = [f.__class__.__name__ for f in features_to_extract[:-1]] # Exclude BDE
    for band in bde.band_dict.keys():
        feature_names.append(f"BDE_{band}")

    return features_to_extract, bde, feature_names


def extract_all_features(X_data, channel, features_to_extract, bde, feature_names):
    """
    Extracts all features for all segments in the provided data.
    """
    num_segments = X_data.shape[0]
    num_features = len(feature_names)
    all_extracted_features = np.zeros((num_segments, num_features))

    print(f"Extracting {num_features} features from {num_segments} segments...")
    for i in tqdm(range(num_segments), desc="Extracting Features"):
        segment = X_data[i]
        feature_vector = []
        
        for feature in features_to_extract[:-1]:
            result = feature.apply(segment)[channel]
            feature_vector.append(result.item() if hasattr(result, 'item') else result)

        bde_values = bde.apply(eeg=segment)
        feature_vector.extend(bde_values[channel])
        
        all_extracted_features[i, :] = np.array(feature_vector)
    
    print("Full feature extraction complete.")
    return all_extracted_features
