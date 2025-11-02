
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from typing import List, Tuple
import logging


class ProbabilityCalibrator:
    """Base class for probability calibration"""
    
    def __init__(self, method: str = 'percentile', **kwargs):
        self.method = method
        self.params = kwargs
        self.is_fitted = False
        self.calibration_params = {}
    
    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        """Fit calibration parameters on validation data"""
        if self.method == 'percentile':
            self._fit_percentile(val_probs, val_labels)
        elif self.method == 'beta':
            self._fit_beta(val_probs, val_labels)
        elif self.method == 'isotonic':
            self._fit_isotonic(val_probs, val_labels)
        elif self.method == 'temperature':
            self._fit_temperature(val_probs, val_labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibration"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        
        if self.method == 'percentile':
            return self._transform_percentile(probs)
        elif self.method == 'beta':
            return self._transform_beta(probs)
        elif self.method == 'isotonic':
            return self._transform_isotonic(probs)
        elif self.method == 'temperature':
            return self._transform_temperature(probs)
    
    def _fit_percentile(self, val_probs, val_labels):
        """
        Fit sigmoid-like transformation where target_percentile% of preictal 
        samples have probability ABOVE 0.5.
        
        Transformation: sigmoid(a * logit(p) + b)
        """
        target_percentile = self.params.get('target_preictal_percentile', 10)
        preictal_probs = val_probs[val_labels == 1]
        
        if len(preictal_probs) == 0:
            logging.warning("No preictal samples in validation set")
            self.calibration_params = {'a': 1.0, 'b': 0.0}
            return
        
        # To have X% above 0.5, we map the (100-X)th percentile to 0.5
        target_prob = np.percentile(preictal_probs, 100 - target_percentile)
        
        # Use another percentile as anchor for the high end
        high_percentile_value = 100 - max(target_percentile - 40, 1)
        high_prob = np.percentile(preictal_probs, high_percentile_value)
        
        # Solve for a and b
        try:
            target_logit = logit(np.clip(target_prob, 1e-7, 1-1e-7))
            high_logit = logit(np.clip(high_prob, 1e-7, 1-1e-7))
            
            if abs(high_logit - target_logit) < 1e-6:
                a = 1.0
            else:
                a = 2.197 / (high_logit - target_logit)
            
            b = -a * target_logit
            
        except (ValueError, ZeroDivisionError):
            logging.warning("Failed to compute logit transformation, using identity")
            a, b = 1.0, 0.0
        
        self.calibration_params = {'a': a, 'b': b, 'target_prob': target_prob}
        logging.info(f"Percentile calibration: a={a:.4f}, b={b:.4f}, target_prob={target_prob:.4f}")
    
    def _transform_percentile(self, probs):
        """Apply percentile-based transformation"""
        a = self.calibration_params['a']
        b = self.calibration_params['b']
        
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        logit_probs = logit(probs_clipped)
        transformed = expit(a * logit_probs + b)
        
        return transformed
    
    def _fit_beta(self, val_probs, val_labels):
        """
        Fit beta calibration: three parameters a, b, c
        Transformed prob = sigmoid(a + b * log(p) + c * log(1-p))
        """
        val_probs_clipped = np.clip(val_probs, 1e-7, 1-1e-7)
        
        def loss_fn(params):
            a, b, c = params
            log_p = np.log(val_probs_clipped)
            log_1mp = np.log(1 - val_probs_clipped)
            calibrated = expit(a + b * log_p + c * log_1mp)
            
            loss = -np.mean(
                val_labels * np.log(calibrated + 1e-7) + 
                (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )
            return loss
        
        init_params = [0.0, 1.0, 1.0]
        result = minimize(loss_fn, init_params, method='L-BFGS-B')
        
        self.calibration_params = {'a': result.x[0], 'b': result.x[1], 'c': result.x[2]}
        logging.info(f"Beta calibration: a={result.x[0]:.4f}, b={result.x[1]:.4f}, c={result.x[2]:.4f}")
    
    def _transform_beta(self, probs):
        """Apply beta calibration"""
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        
        a = self.calibration_params['a']
        b = self.calibration_params['b']
        c = self.calibration_params['c']
        
        log_p = np.log(probs_clipped)
        log_1mp = np.log(1 - probs_clipped)
        transformed = expit(a + b * log_p + c * log_1mp)
        
        return transformed
    
    def _fit_isotonic(self, val_probs, val_labels):
        """Fit isotonic regression (non-parametric monotonic calibration)"""
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.iso_reg.fit(val_probs, val_labels)
        logging.info("Fitted isotonic regression calibration")
    
    def _transform_isotonic(self, probs):
        """Apply isotonic calibration"""
        return self.iso_reg.predict(probs)
    
    def _fit_temperature(self, val_probs, val_labels):
        """
        Fit temperature scaling with bias
        Transformed = sigmoid((logit(p) - b) / T)
        """
        val_probs_clipped = np.clip(val_probs, 1e-7, 1-1e-7)
        
        def loss_fn(params):
            T, b = params
            if T <= 0:
                return 1e10
            
            logit_p = logit(val_probs_clipped)
            calibrated = expit((logit_p - b) / T)
            
            loss = -np.mean(
                val_labels * np.log(calibrated + 1e-7) + 
                (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )
            return loss
        
        init_params = [1.0, 0.0]
        result = minimize(loss_fn, init_params, method='L-BFGS-B',
                         bounds=[(0.1, 10.0), (-5.0, 5.0)])
        
        self.calibration_params = {'T': result.x[0], 'b': result.x[1]}
        logging.info(f"Temperature calibration: T={result.x[0]:.4f}, b={result.x[1]:.4f}")
    
    def _transform_temperature(self, probs):
        """Apply temperature scaling"""
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        
        T = self.calibration_params['T']
        b = self.calibration_params['b']
        
        logit_p = logit(probs_clipped)
        transformed = expit((logit_p - b) / T)
        
        return transformed


def calibrate_ensemble(
    test_probs_stack: np.ndarray,
    val_probs_list: List[np.ndarray],
    val_labels_list: List[np.ndarray],
    val_aucs: np.ndarray,
    calibration_method: str = 'percentile',
    **calibration_params
) -> Tuple[np.ndarray, List[ProbabilityCalibrator]]:
    """
    Calibrate ensemble predictions
    
    Args:
        test_probs_stack: (n_folds, n_test_samples) test probabilities
        val_probs_list: List of (n_val_samples,) validation probabilities per fold
        val_labels_list: List of (n_val_samples,) validation labels per fold
        val_aucs: (n_folds,) validation AUCs for weighting
        calibration_method: Which calibration method to use
        **calibration_params: Additional parameters (e.g., target_preictal_percentile)
    
    Returns:
        calibrated_probs: Final calibrated probabilities
        calibrators: List of fitted calibrators for each model
    """
    n_models = len(val_probs_list)
    calibrators = []
    calibrated_test_probs = []
    
    for i in range(n_models):
        calibrator = ProbabilityCalibrator(method=calibration_method, **calibration_params)
        calibrator.fit(val_probs_list[i], val_labels_list[i])
        calibrators.append(calibrator)
        
        calibrated_test_probs.append(calibrator.transform(test_probs_stack[i]))
    
    # Stack and ensemble
    calibrated_stack = np.stack(calibrated_test_probs)
    
    # Weighted ensemble based on validation AUC
    weights = val_aucs / val_aucs.sum()
    final_probs = np.tensordot(weights, calibrated_stack, axes=1)
    
    logging.info(f"Calibrated predictions using method: {calibration_method}")
    
    return final_probs, calibrators