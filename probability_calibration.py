import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
import logging


class ProbabilityCalibrator:
    """
    Calibrates probability outputs to optimize sensitivity/FPR trade-off.
    Fits transformation on validation data, applies to test data.
    """
    
    def __init__(self, method='percentile', target_preictal_percentile=10):
        """
        Args:
            method: 'percentile', 'beta', 'isotonic', or 'temperature'
            target_preictal_percentile: For percentile method, what % of preictal 
                                       samples should be around threshold 0.5
        """
        self.method = method
        self.target_percentile = target_preictal_percentile
        self.params = {}
        
    def fit(self, val_probs, val_labels):
        """
        Fit calibration parameters on validation data.
        
        Args:
            val_probs: Validation probabilities (N,)
            val_labels: Validation labels (N,)
        """
        if self.method == 'percentile':
            self._fit_percentile(val_probs, val_labels)
        elif self.method == 'beta':
            self._fit_beta(val_probs, val_labels)
        elif self.method == 'isotonic':
            self._fit_isotonic(val_probs, val_labels)
        elif self.method == 'temperature':
            self._fit_temperature(val_probs, val_labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return self
    
    def transform(self, probs):
        """
        Apply calibration to new probabilities.
        
        Args:
            probs: Probabilities to transform (N,)
        Returns:
            Calibrated probabilities (N,)
        """
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
        samples have probability around 0.5.
        
        Transformation: sigmoid(a * logit(p) + b)
        """
        preictal_probs = val_probs[val_labels == 1]
        
        if len(preictal_probs) == 0:
            logging.warning("No preictal samples in validation set")
            self.params = {'a': 1.0, 'b': 0.0}
            return
        
        # Find the probability value at target percentile
        target_prob = np.percentile(preictal_probs, self.target_percentile)
        
        # We want: sigmoid(a * logit(target_prob) + b) = 0.5
        # This means: a * logit(target_prob) + b = 0
        # So: b = -a * logit(target_prob)
        
        # We also want to preserve some spread. Use another percentile as anchor.
        high_percentile = min(self.target_percentile + 40, 90)
        high_prob = np.percentile(preictal_probs, high_percentile)
        
        # We want high_prob to map to ~0.9
        # sigmoid(a * logit(high_prob) + b) = 0.9
        # a * logit(high_prob) + b = logit(0.9) ≈ 2.197
        
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
        
        self.params = {'a': a, 'b': b, 'target_prob': target_prob}
        logging.info(f"Percentile calibration: a={a:.4f}, b={b:.4f}, target_prob={target_prob:.4f}")
    
    def _transform_percentile(self, probs):
        """Apply percentile-based transformation"""
        a = self.params['a']
        b = self.params['b']
        
        # Avoid numerical issues
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        logit_probs = logit(probs_clipped)
        
        # Apply affine transformation in logit space, then sigmoid back
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
            
            # Binary cross-entropy loss
            loss = -np.mean(
                val_labels * np.log(calibrated + 1e-7) + 
                (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )
            return loss
        
        # Initialize near identity transformation
        init_params = [0.0, 1.0, 1.0]
        
        result = minimize(loss_fn, init_params, method='L-BFGS-B')
        
        self.params = {'a': result.x[0], 'b': result.x[1], 'c': result.x[2]}
        logging.info(f"Beta calibration: a={result.x[0]:.4f}, b={result.x[1]:.4f}, c={result.x[2]:.4f}")
    
    def _transform_beta(self, probs):
        """Apply beta calibration"""
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        
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
            
            # Binary cross-entropy
            loss = -np.mean(
                val_labels * np.log(calibrated + 1e-7) + 
                (1 - val_labels) * np.log(1 - calibrated + 1e-7)
            )
            return loss
        
        init_params = [1.0, 0.0]
        result = minimize(loss_fn, init_params, method='L-BFGS-B',
                         bounds=[(0.1, 10.0), (-5.0, 5.0)])
        
        self.params = {'T': result.x[0], 'b': result.x[1]}
        logging.info(f"Temperature calibration: T={result.x[0]:.4f}, b={result.x[1]:.4f}")
    
    def _transform_temperature(self, probs):
        """Apply temperature scaling"""
        probs_clipped = np.clip(probs, 1e-7, 1-1e-7)
        
        T = self.params['T']
        b = self.params['b']
        
        logit_p = logit(probs_clipped)
        transformed = expit((logit_p - b) / T)
        
        return transformed


def calibrate_ensemble_predictions(
    test_probs_stack,
    val_probs_list,
    val_labels_list,
    val_aucs,
    calibration_method='percentile',
    target_percentile=10
):
    """
    Calibrate each inner fold's predictions on test data using validation data.
    
    Args:
        test_probs_stack: (n_folds, n_test_samples) test probabilities
        val_probs_list: List of (n_val_samples,) validation probabilities per fold
        val_labels_list: List of (n_val_samples,) validation labels per fold
        val_aucs: (n_folds,) validation AUCs for weighting
        calibration_method: Which calibration method to use
        target_percentile: For percentile method
        
    Returns:
        calibrated_probs: (n_test_samples,) weighted ensemble of calibrated predictions
        calibrators: List of fitted calibrator objects
    """
    n_folds = test_probs_stack.shape[0]
    calibrated_test_probs = []
    calibrators = []
    
    for fold_idx in range(n_folds):
        # Fit calibrator on validation data
        calibrator = ProbabilityCalibrator(
            method=calibration_method,
            target_preictal_percentile=target_percentile
        )
        calibrator.fit(val_probs_list[fold_idx], val_labels_list[fold_idx])
        
        # Transform test probabilities
        calibrated = calibrator.transform(test_probs_stack[fold_idx])
        calibrated_test_probs.append(calibrated)
        calibrators.append(calibrator)
    
    # Stack and ensemble
    calibrated_stack = np.stack(calibrated_test_probs)
    
    # Weighted ensemble based on validation AUC
    weights = val_aucs / val_aucs.sum()
    final_probs = np.tensordot(weights, calibrated_stack, axes=1)
    
    logging.info(f"Calibrated predictions using method: {calibration_method}")
    
    return final_probs, calibrators