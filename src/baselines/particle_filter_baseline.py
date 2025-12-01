"""
Particle Filter Baseline Implementation.

Particle Filter provides non-parametric state estimation
through particle-based probability distribution representation.
Used as advanced baseline for highly nonlinear orbital dynamics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .ekf_baseline import EKFBaseline  # Inherit dynamics


class ParticleFilterBaseline(EKFBaseline):
    """
    Particle Filter baseline with orbital dynamics.
    
    Extends EKF baseline with particle-based state estimation
    for handling arbitrary probability distributions and
    severe nonlinearities in orbital prediction.
    """
    
    def __init__(self,
                 include_j2: bool = True,
                 process_noise_std: float = 1e-9,
                 measurement_noise_std: float = 1000.0,
                 integration_step: float = 60.0,
                 n_particles: int = 1000,
                 resample_threshold: float = 0.5,
                 initialization_std: float = 10000.0):
        """
        Initialize Particle Filter baseline.
        
        Args:
            include_j2: Include J2 gravitational perturbations
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
            integration_step: Integration time step (seconds)
            n_particles: Number of particles
            resample_threshold: Effective sample size threshold for resampling
            initialization_std: Standard deviation for particle initialization
        """
        super().__init__(include_j2, process_noise_std, measurement_noise_std, integration_step)
        
        # Particle Filter parameters
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.initialization_std = initialization_std
        
        # Particle states and weights
        self.particles = None
        self.weights = None
        
        # State dimension
        self.n = 6  # [x, y, z, vx, vy, vz]
    
    def initialize_particles(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        """
        Initialize particles around initial state.
        
        Args:
            initial_state: Initial state estimate (6,)
            initial_covariance: Initial covariance matrix (6, 6)
        """
        self.particles = np.zeros((self.n_particles, self.n))
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Sample particles from multivariate normal distribution
        try:
            # Use provided covariance if available
            self.particles = np.random.multivariate_normal(
                initial_state, initial_covariance, self.n_particles
            )
        except np.linalg.LinAlgError:
            # Fallback to diagonal covariance
            std_vec = np.sqrt(np.diag(initial_covariance))
            for i in range(self.n):
                self.particles[:, i] = np.random.normal(
                    initial_state[i], std_vec[i], self.n_particles
                )
    
    def predict_particles(self, dt: float):
        """
        Predict particles forward in time.
        
        Args:
            dt: Time step (seconds)
        """
        # Propagate each particle
        for i in range(self.n_particles):
            # Add process noise before propagation
            noise = np.random.normal(0, self.process_noise_std * np.sqrt(dt), self.n)
            
            # For position noise, scale by dt^2
            noise[:3] *= dt
            
            # Propagate with noise
            try:
                self.particles[i] = self.integrate_state(self.particles[i] + noise, dt)
            except Exception as e:
                # If integration fails, keep previous state
                warnings.warn(f"Particle {i} propagation failed: {e}")
                pass
    
    def update_weights(self, measurement: np.ndarray, measurement_type: str = 'position'):
        """
        Update particle weights based on measurement likelihood.
        
        Args:
            measurement: Measurement vector
            measurement_type: Type of measurement
        """
        log_likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            particle = self.particles[i]
            
            # Compute predicted measurement
            if measurement_type == 'position':
                predicted_measurement = particle[:3]
                measurement_noise_var = self.measurement_noise_std**2
                
                # Gaussian likelihood
                diff = measurement - predicted_measurement
                log_likelihood = -0.5 * np.sum(diff**2) / measurement_noise_var
                log_likelihood -= 1.5 * np.log(2 * np.pi * measurement_noise_var)
                
            elif measurement_type == 'range':
                predicted_range = np.linalg.norm(particle[:3])
                measured_range = measurement[0] if len(measurement) == 1 else np.linalg.norm(measurement)
                
                # Gaussian likelihood for range
                diff = measured_range - predicted_range
                log_likelihood = -0.5 * diff**2 / self.measurement_noise_std**2
                log_likelihood -= 0.5 * np.log(2 * np.pi * self.measurement_noise_std**2)
                
            else:
                # Unknown measurement type - uniform weights
                log_likelihood = 0.0
            
            log_likelihoods[i] = log_likelihood
        
        # Convert to weights and normalize
        # Subtract max for numerical stability
        max_log_likelihood = np.max(log_likelihoods)
        weights = np.exp(log_likelihoods - max_log_likelihood)
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            self.weights = weights / weight_sum
        else:
            # All weights are zero - reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def effective_sample_size(self) -> float:
        """
        Compute effective sample size.
        
        Returns:
            Effective sample size (ESS)
        """
        return 1.0 / np.sum(self.weights**2)
    
    def resample_particles(self):
        """
        Resample particles using systematic resampling.
        """
        # Systematic resampling
        indices = np.zeros(self.n_particles, dtype=int)
        
        # Cumulative sum of weights
        cumulative_sum = np.cumsum(self.weights)
        
        # Starting point
        u = np.random.random() / self.n_particles
        
        # Systematic sampling
        i, j = 0, 0
        while i < self.n_particles:
            if u <= cumulative_sum[j]:
                indices[i] = j
                u += 1.0 / self.n_particles
                i += 1
            else:
                j += 1
        
        # Update particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted state estimate and covariance.
        
        Returns:
            Tuple of (state_estimate, covariance_estimate)
        """
        # Weighted mean
        state_estimate = np.sum(self.weights[:, np.newaxis] * self.particles, axis=0)
        
        # Weighted covariance
        covariance_estimate = np.zeros((self.n, self.n))
        for i in range(self.n_particles):
            diff = self.particles[i] - state_estimate
            covariance_estimate += self.weights[i] * np.outer(diff, diff)
        
        return state_estimate, covariance_estimate
    
    def predict_step_particles(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Particle filter prediction step.
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Predict particles
        self.predict_particles(dt)
        
        # Get state estimate
        return self.get_state_estimate()
    
    def update_step_particles(self, 
                             measurement: np.ndarray,
                             measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """
        Particle filter update step.
        
        Args:
            measurement: Measurement vector
            measurement_type: Type of measurement
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Update weights
        self.update_weights(measurement, measurement_type)
        
        # Check if resampling is needed
        ess = self.effective_sample_size()
        if ess < self.resample_threshold * self.n_particles:
            self.resample_particles()
        
        # Get state estimate
        return self.get_state_estimate()
    
    def predict_trajectory(self,
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          measurements: Optional[List[np.ndarray]] = None,
                          measurement_times: Optional[np.ndarray] = None,
                          measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict orbital trajectory using Particle Filter.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz] (m, m/s)
            times: Time points for predictions (hours)
            measurements: List of measurements for updates
            measurement_times: Times of measurements (hours)
            measurement_type: Type of measurements
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        times_sec = times * 3600  # Convert to seconds
        n_times = len(times)
        
        # Initialize outputs
        predicted_states = np.zeros((n_times, 6))
        uncertainties = np.zeros((n_times, 6))
        
        # Initialize particles
        initial_cov = np.eye(6) * self.initialization_std**2
        initial_cov[3:, 3:] *= 1e-6  # Smaller velocity uncertainty
        
        self.initialize_particles(initial_state, initial_cov)
        
        # Track measurement index
        measurement_idx = 0
        measurement_times_sec = measurement_times * 3600 if measurement_times is not None else None
        
        current_time = 0.0
        
        for i, target_time in enumerate(times_sec):
            # Time step
            dt = target_time - current_time
            
            if dt > 0:
                # Prediction step
                predicted_state, predicted_covariance = self.predict_step_particles(dt)
            else:
                # Get current estimate
                predicted_state, predicted_covariance = self.get_state_estimate()
            
            # Check for measurements at this time
            if (measurements is not None and 
                measurement_times_sec is not None and 
                measurement_idx < len(measurements)):
                
                # Check if measurement is close to current time
                if abs(target_time - measurement_times_sec[measurement_idx]) < 300:  # Within 5 minutes
                    # Update step
                    updated_state, updated_covariance = self.update_step_particles(
                        measurements[measurement_idx], measurement_type
                    )
                    predicted_state = updated_state
                    predicted_covariance = updated_covariance
                    measurement_idx += 1
            
            # Store results
            predicted_states[i] = predicted_state
            uncertainties[i] = np.sqrt(np.diag(predicted_covariance))
            
            current_time = target_time
        
        return predicted_states, uncertainties
    
    def get_name(self) -> str:
        """Return baseline model name."""
        return f"PF{self.n_particles}{'_J2' if self.include_j2 else ''}"


def test_particle_filter_baseline():
    """Test Particle Filter baseline implementation."""
    
    print("🧪 Testing Particle Filter Baseline...")
    
    # Create test initial state
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    # Test times
    times = np.linspace(0, 2, 25)  # hours
    
    # Initialize baseline
    pf = ParticleFilterBaseline(
        include_j2=True,
        n_particles=500,
        resample_threshold=0.5
    )
    
    # Create synthetic measurements
    measurement_times = np.array([0.5, 1.0, 1.5])  # hours
    measurements = []
    
    for t_meas in measurement_times:
        t_sec = t_meas * 3600
        true_state = pf.integrate_state(initial_state, t_sec)
        meas_noise = np.random.normal(0, 500, 3)  # 500m noise
        measurement = true_state[:3] + meas_noise
        measurements.append(measurement)
    
    # Predict trajectory
    predicted_states, uncertainties = pf.predict_trajectory(
        initial_state, times, measurements, measurement_times
    )
    
    print(f"✅ Particle Filter prediction successful!")
    print(f"   Model: {pf.get_name()}")
    print(f"   Particles: {pf.n_particles}")
    print(f"   Final ESS: {pf.effective_sample_size():.1f}")
    print(f"   Initial position magnitude: {np.linalg.norm(initial_state[:3])/1000:.1f} km")
    print(f"   Final position magnitude: {np.linalg.norm(predicted_states[-1, :3])/1000:.1f} km")
    print(f"   Final position uncertainty: ±{uncertainties[-1, 0]/1000:.1f} km")
    print(f"   Final velocity uncertainty: ±{uncertainties[-1, 3]:.1f} m/s")
    
    return predicted_states, uncertainties


if __name__ == "__main__":
    test_particle_filter_baseline()