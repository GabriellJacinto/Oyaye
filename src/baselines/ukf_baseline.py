"""
Unscented Kalman Filter (UKF) Baseline Implementation.

UKF provides better nonlinearity handling compared to EKF through
sigma point sampling. Used as advanced filtering baseline for
comparison with NP-SNN predictions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.linalg import cholesky, LinAlgError
import warnings

from .ekf_baseline import EKFBaseline  # Inherit dynamics


class UKFBaseline(EKFBaseline):
    """
    Unscented Kalman Filter baseline with orbital dynamics.
    
    Extends EKF baseline with sigma point sampling for better
    nonlinearity handling in orbital state estimation.
    """
    
    def __init__(self,
                 include_j2: bool = True,
                 process_noise_std: float = 1e-9,
                 measurement_noise_std: float = 1000.0,
                 integration_step: float = 60.0,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        """
        Initialize UKF baseline.
        
        Args:
            include_j2: Include J2 gravitational perturbations
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation  
            integration_step: Integration time step (seconds)
            alpha: UKF spread parameter (1e-4 <= alpha <= 1)
            beta: Prior knowledge parameter (beta=2 optimal for Gaussian)
            kappa: Secondary scaling parameter
        """
        super().__init__(include_j2, process_noise_std, measurement_noise_std, integration_step)
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # State dimension
        self.n = 6  # [x, y, z, vx, vy, vz]
        
        # Compute UKF weights
        self._compute_ukf_weights()
    
    def _compute_ukf_weights(self):
        """Compute UKF sigma point weights."""
        n = self.n
        
        # Composite scaling parameter
        self.lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        # Number of sigma points
        self.n_sigma = 2 * n + 1
        
        # Weights for mean
        self.Wm = np.zeros(self.n_sigma)
        self.Wm[0] = self.lambda_param / (n + self.lambda_param)
        self.Wm[1:] = 1 / (2 * (n + self.lambda_param))
        
        # Weights for covariance
        self.Wc = self.Wm.copy()
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
    
    def generate_sigma_points(self, 
                             mean: np.ndarray, 
                             covariance: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for UKF.
        
        Args:
            mean: State mean (n,)
            covariance: State covariance (n, n)
            
        Returns:
            Sigma points (2n+1, n)
        """
        n = len(mean)
        sigma_points = np.zeros((self.n_sigma, n))
        
        # Central sigma point
        sigma_points[0] = mean
        
        try:
            # Matrix square root using Cholesky decomposition
            sqrt_matrix = cholesky((n + self.lambda_param) * covariance)
        except LinAlgError:
            # Fallback to SVD if Cholesky fails
            U, S, Vt = np.linalg.svd(covariance)
            sqrt_matrix = U @ np.diag(np.sqrt(S * (n + self.lambda_param)))
        
        # Generate remaining sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_matrix[i]
            sigma_points[i + 1 + n] = mean - sqrt_matrix[i]
        
        return sigma_points
    
    def unscented_transform(self, 
                           sigma_points: np.ndarray, 
                           func: callable,
                           noise_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transform through nonlinear function.
        
        Args:
            sigma_points: Input sigma points (n_sigma, n_x)
            func: Nonlinear function to apply
            noise_cov: Additive noise covariance (optional)
            
        Returns:
            Tuple of (transformed_mean, transformed_covariance)
        """
        n_sigma, n_x = sigma_points.shape
        
        # Apply function to each sigma point
        transformed_points = np.zeros((n_sigma, n_x))
        
        for i, point in enumerate(sigma_points):
            try:
                transformed_points[i] = func(point)
            except Exception as e:
                warnings.warn(f"Function evaluation failed for sigma point {i}: {e}")
                # Use previous point or original point as fallback
                if i > 0:
                    transformed_points[i] = transformed_points[i-1]
                else:
                    transformed_points[i] = point
        
        # Compute weighted mean
        mean = np.sum(self.Wm[:, np.newaxis] * transformed_points, axis=0)
        
        # Compute weighted covariance
        covariance = np.zeros((n_x, n_x))
        for i in range(n_sigma):
            diff = transformed_points[i] - mean
            covariance += self.Wc[i] * np.outer(diff, diff)
        
        # Add process noise if provided
        if noise_cov is not None:
            covariance += noise_cov
        
        return mean, covariance
    
    def predict_step(self, 
                    state: np.ndarray, 
                    covariance: np.ndarray, 
                    dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF prediction step using sigma points.
        
        Args:
            state: Current state estimate
            covariance: Current covariance matrix
            dt: Time step (seconds)
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(state, covariance)
        
        # Define propagation function
        def propagate_func(x):
            return self.integrate_state(x, dt)
        
        # Process noise covariance
        dt2 = dt**2
        dt3 = dt**3 / 3.0
        dt4 = dt**4 / 4.0
        
        Q = np.zeros((6, 6))
        Q[:3, :3] = self.process_noise_std**2 * dt4 * np.eye(3)
        Q[:3, 3:] = self.process_noise_std**2 * dt3 * np.eye(3)
        Q[3:, :3] = self.process_noise_std**2 * dt3 * np.eye(3)
        Q[3:, 3:] = self.process_noise_std**2 * dt2 * np.eye(3)
        
        # Apply unscented transform
        predicted_state, predicted_covariance = self.unscented_transform(
            sigma_points, propagate_func, Q
        )
        
        return predicted_state, predicted_covariance
    
    def update_step(self, 
                   predicted_state: np.ndarray,
                   predicted_covariance: np.ndarray,
                   measurement: np.ndarray,
                   measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """
        UKF update step using sigma points.
        
        Args:
            predicted_state: Predicted state from forecast
            predicted_covariance: Predicted covariance matrix
            measurement: Measurement vector
            measurement_type: Type of measurement
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Generate sigma points from predicted state
        sigma_points = self.generate_sigma_points(predicted_state, predicted_covariance)
        
        # Define measurement function
        if measurement_type == 'position':
            def h_func(x):
                return x[:3]  # Direct position measurement
            
            measurement_dim = 3
            R = self.measurement_noise_std**2 * np.eye(3)
            
        elif measurement_type == 'range':
            def h_func(x):
                return np.array([np.linalg.norm(x[:3])])  # Range measurement
            
            measurement_dim = 1
            R = np.array([[self.measurement_noise_std**2]])
            
        else:
            # Unknown measurement type - no update
            return predicted_state, predicted_covariance
        
        # Apply measurement function to sigma points
        measurement_sigma_points = np.zeros((self.n_sigma, measurement_dim))
        
        for i, point in enumerate(sigma_points):
            measurement_sigma_points[i] = h_func(point)
        
        # Predicted measurement mean and covariance
        predicted_measurement = np.sum(self.Wm[:, np.newaxis] * measurement_sigma_points, axis=0)
        
        # Measurement covariance
        Pyy = np.zeros((measurement_dim, measurement_dim))
        for i in range(self.n_sigma):
            diff = measurement_sigma_points[i] - predicted_measurement
            Pyy += self.Wc[i] * np.outer(diff, diff)
        
        Pyy += R  # Add measurement noise
        
        # Cross-covariance
        Pxy = np.zeros((self.n, measurement_dim))
        for i in range(self.n_sigma):
            state_diff = sigma_points[i] - predicted_state
            meas_diff = measurement_sigma_points[i] - predicted_measurement
            Pxy += self.Wc[i] * np.outer(state_diff, meas_diff)
        
        # Kalman gain
        try:
            K = Pxy @ np.linalg.inv(Pyy)
        except np.linalg.LinAlgError:
            # Singular measurement covariance - skip update
            return predicted_state, predicted_covariance
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # State and covariance update
        updated_state = predicted_state + K @ innovation
        updated_covariance = predicted_covariance - K @ Pyy @ K.T
        
        return updated_state, updated_covariance
    
    def get_name(self) -> str:
        """Return baseline model name."""
        return f"UKF{'_J2' if self.include_j2 else ''}"


def test_ukf_baseline():
    """Test UKF baseline implementation."""
    
    print("🧪 Testing UKF Baseline...")
    
    # Create test initial state
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    # Test times
    times = np.linspace(0, 2, 25)  # hours
    
    # Initialize baseline
    ukf = UKFBaseline(
        include_j2=True,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0
    )
    
    # Create synthetic measurements
    measurement_times = np.array([0.5, 1.0, 1.5])  # hours
    measurements = []
    
    for t_meas in measurement_times:
        t_sec = t_meas * 3600
        true_state = ukf.integrate_state(initial_state, t_sec)
        meas_noise = np.random.normal(0, 500, 3)  # 500m noise
        measurement = true_state[:3] + meas_noise
        measurements.append(measurement)
    
    # Predict trajectory
    predicted_states, uncertainties = ukf.predict_trajectory(
        initial_state, times, measurements, measurement_times
    )
    
    print(f"✅ UKF prediction successful!")
    print(f"   Model: {ukf.get_name()}")
    print(f"   Sigma points: {ukf.n_sigma}")
    print(f"   UKF parameters: α={ukf.alpha}, β={ukf.beta}, κ={ukf.kappa}")
    print(f"   Initial position magnitude: {np.linalg.norm(initial_state[:3])/1000:.1f} km")
    print(f"   Final position magnitude: {np.linalg.norm(predicted_states[-1, :3])/1000:.1f} km")
    print(f"   Final position uncertainty: ±{uncertainties[-1, 0]/1000:.1f} km")
    print(f"   Final velocity uncertainty: ±{uncertainties[-1, 3]:.1f} m/s")
    
    # Compare with EKF
    ekf = EKFBaseline(include_j2=True)
    ekf_states, ekf_uncertainties = ekf.predict_trajectory(
        initial_state, times, measurements, measurement_times
    )
    
    # Position difference
    pos_diff = np.linalg.norm(predicted_states[-1, :3] - ekf_states[-1, :3])
    print(f"   UKF vs EKF position difference: {pos_diff/1000:.1f} km")
    
    return predicted_states, uncertainties


if __name__ == "__main__":
    test_ukf_baseline()