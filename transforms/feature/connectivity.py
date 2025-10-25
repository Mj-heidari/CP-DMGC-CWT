import numpy as np
import mne
import warnings
from .base import FeatureTransform
try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    print("Warning: mne-connectivity not found. MNE-based connectivity features will not work.")
    print("Please install it: pip install mne-connectivity")
    spectral_connectivity_epochs = None

class MNEConnectivityBase(FeatureTransform):
    """ Base class for MNE connectivity measures using spectral_connectivity_epochs """
    def __init__(self, method='coh', fmin=0.5, fmax=50, sfreq=128, mode='multitaper', faverage=True):
        if spectral_connectivity_epochs is None:
             raise ImportError("mne-connectivity is required for this feature but not installed.")
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.sfreq = sfreq
        self.mode = mode
        self.faverage = faverage # Average over frequencies in the band

    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray:
        """
        Args:
            eeg (np.ndarray): EEG data for one segment, shape (n_channels, n_times)
        Returns:
            np.ndarray: Mean connectivity value, shape (n_channels, 1)
                       (Value is duplicated across channels as it's a network metric)
        """
        n_channels, n_times = eeg.shape
        if n_channels <= 1:
             # Connectivity requires at least 2 channels
             return np.full((n_channels, 1), np.nan)

        # spectral_connectivity_epochs expects Epochs object or data array (n_epochs, n_channels, n_times)
        # We treat the single segment as one epoch
        data_mne = eeg[np.newaxis, :, :] # Add epoch dimension -> (1, n_channels, n_times)

        mean_con = np.nan # Default to NaN
        try:
            # Suppress potential MNE warnings during connectivity calculation
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 # Calculate connectivity
                 con = spectral_connectivity_epochs(
                     data_mne,
                     method=self.method,
                     mode=self.mode,
                     sfreq=self.sfreq,
                     fmin=self.fmin,
                     fmax=self.fmax,
                     faverage=self.faverage,
                     n_jobs=1, # Use 1 job for simplicity within feature extraction
                     verbose=False
                 )
            # Get the dense connectivity matrix (n_channels, n_channels)
            # squeeze() removes the frequency dimension if faverage=True
            con_matrix = con.get_data(output='dense').squeeze()

            # Calculate the mean of the upper triangle (excluding diagonal)
            # Check if the output is a valid matrix
            if isinstance(con_matrix, np.ndarray) and con_matrix.ndim == 2 and con_matrix.shape[0] == n_channels and con_matrix.shape[1] == n_channels:
                 iu = np.triu_indices(n_channels, k=1) # Indices for upper triangle (k=1 excludes diagonal)
                 # Use abs value for measures like imcoh, coh. Others like PLV, PLI are often in [0, 1] or [-1, 1]
                 # Taking mean of absolute values provides a general measure of connection strength magnitude
                 con_values = con_matrix[iu]
                 # Only average if there are valid (non-NaN) connectivity values
                 if np.any(np.isfinite(con_values)):
                      mean_con = float(np.nanmean(np.abs(con_values))) # Use nanmean to ignore NaNs
                 else:
                      print(f"Warning: All connectivity values were NaN for method {self.method}")

            else:
                 print(f"Warning: Unexpected connectivity matrix shape {getattr(con_matrix, 'shape', 'N/A')} or type {type(con_matrix)} for method {self.method}. Expected ({n_channels},{n_channels}).")


        except ImportError:
             # This handles the case where mne-connectivity wasn't installed initially
             print(f"Error: mne-connectivity is required for {self.__class__.__name__} but not found.")
             # mean_con remains np.nan
        except Exception as e:
            print(f"Error calculating MNE connectivity ({self.method}) for segment: {e}")
            # mean_con remains np.nan

        # Return the mean value, duplicated for each channel row
        return np.full((n_channels, 1), mean_con, dtype=float)

# --- Original MeanAbsCorrelation using np.corrcoef ---
class MeanAbsCorrelation(FeatureTransform): #
    """ Calculates the mean absolute Pearson correlation between all channel pairs. """
    def apply(self, eeg: np.ndarray, **kwargs) -> np.ndarray: #
        n_channels = eeg.shape[0] #
        if n_channels <= 1: #
             # Correlation requires at least 2 channels
             return np.full((n_channels, 1), np.nan) #

        # Calculate Pearson correlation coefficient matrix
        with warnings.catch_warnings(): # Suppress potential warnings from corrcoef (e.g., constant input)
             warnings.simplefilter("ignore")
             corr = np.corrcoef(eeg) #

        # Handle case where corrcoef might return NaN (e.g., zero variance channels)
        if not np.all(np.isfinite(corr)):
             print("Warning: NaN found in correlation matrix. Returning NaN for MeanAbsCorrelation.")
             return np.full((n_channels, 1), np.nan)

        iu = np.triu_indices(n_channels, k=1) # Indices for upper triangle, excluding diagonal
        if len(corr[iu]) == 0: # Should not happen if n_channels > 1, but safety check
             val = np.nan
        else:
             val = float(np.mean(np.abs(corr[iu]))) # Calculate mean of absolute values

        # Return the single mean value, broadcast to all channels
        return np.full((n_channels, 1), val) #


# --- New Connectivity Features using MNE ---
class MeanCoh(MNEConnectivityBase):
    """ Mean Coherence across all channel pairs. Requires mne-connectivity. """
    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method='coh', fmin=fmin, fmax=fmax, sfreq=sfreq)

class MeanPLV(MNEConnectivityBase):
    """ Mean Phase Locking Value across all channel pairs. Requires mne-connectivity. """
    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method='plv', fmin=fmin, fmax=fmax, sfreq=sfreq)

class MeanImCoh(MNEConnectivityBase):
    """ Mean Imaginary Part of Coherency across all channel pairs. Requires mne-connectivity. """
    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method='imcoh', fmin=fmin, fmax=fmax, sfreq=sfreq)

class MeanPLI(MNEConnectivityBase):
    """ Mean Phase Lag Index across all channel pairs. Requires mne-connectivity. """
    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method='pli', fmin=fmin, fmax=fmax, sfreq=sfreq)

class MeanWPLI(MNEConnectivityBase):
    """ Mean Weighted Phase Lag Index across all channel pairs. Requires mne-connectivity. """
    def __init__(self, fmin=0.5, fmax=50, sfreq=128):
        super().__init__(method='wpli', fmin=fmin, fmax=fmax, sfreq=sfreq)

# Example Usage Block (optional, for testing)
if __name__ == "__main__":
    # Check if mne-connectivity is available before running examples
    if spectral_connectivity_epochs is not None:
        eeg_data = np.random.randn(18, 128*5) # 18 channels, 5 seconds
        sfreq = 128

        print("Testing Connectivity Features:")
        transforms_to_test = [
            MeanAbsCorrelation(),
            MeanCoh(sfreq=sfreq),
            MeanPLV(sfreq=sfreq),
            MeanImCoh(sfreq=sfreq),
            MeanPLI(sfreq=sfreq),
            MeanWPLI(sfreq=sfreq)
        ]

        for t in transforms_to_test:
            try:
                 result = t.apply(eeg_data)
                 # Check if result is valid before accessing
                 if result is not None and result.size > 0 and np.isfinite(result[0,0]):
                      print(f"  {t.__class__.__name__}: {result[0,0]:.4f}") # Print the mean value
                 else:
                      print(f"  {t.__class__.__name__}: Failed or returned NaN")
            except Exception as e:
                 print(f"  Error testing {t.__class__.__name__}: {e}")
    else:
        print("Skipping connectivity examples because mne-connectivity is not installed.")
        # Test MeanAbsCorrelation separately as it doesn't depend on mne-connectivity
        print("\nTesting MeanAbsCorrelation (requires only numpy):")
        eeg_data = np.random.randn(18, 128*5)
        t_corr = MeanAbsCorrelation()
        result_corr = t_corr.apply(eeg_data)
        if result_corr is not None and result_corr.size > 0 and np.isfinite(result_corr[0,0]):
             print(f"  {t_corr.__class__.__name__}: {result_corr[0,0]:.4f}")
        else:
             print(f"  {t_corr.__class__.__name__}: Failed or returned NaN")
