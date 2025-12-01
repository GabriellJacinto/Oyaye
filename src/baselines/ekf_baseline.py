"""
Extended Kalman Filter (EKF) Baseline Implementation.

Classical state estimation using EKF with 2-body + J2 dynamics
for comparison with NP-SNN predictions. Represents the standard
filtering approach used in operational tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import warnings

from .sgp4_baseline import BaselineModel


class EKFBaseline(BaselineModel):
    """
    Extended Kalman Filter baseline with orbital dynamics.
    
    Implements classical EKF state estimation with 2-body + J2 
    gravitational dynamics for satellite state prediction.
    """
    
    def __init__(self,
                 include_j2: bool = True,
                 process_noise_std: float = 1e-9,
                 measurement_noise_std: float = 1000.0,
                 integration_step: float = 60.0):
        """
        Initialize EKF baseline.
        
        Args:
            include_j2: Include J2 gravitational perturbations
            process_noise_std: Process noise standard deviation (m/s^2)
            measurement_noise_std: Measurement noise standard deviation (m)
            integration_step: Integration time step (seconds)
        """
        self.include_j2 = include_j2
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.integration_step = integration_step
        
        # Earth gravitational parameters
        self.mu = 3.986004418e14  # m^3/s^2
        self.R_earth = 6.378137e6  # m
        self.J2 = 1.08262668e-3   # J2 coefficient
        
        # Initialize state covariance
        self.initial_pos_uncertainty = 1000.0  # m
        self.initial_vel_uncertainty = 1.0     # m/s
    
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Orbital dynamics function (2-body + J2).
        
        Args:
            t: Time (not used for autonomous system)
            state: State vector [x, y, z, vx, vy, vz]
            
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        r = state[:3]
        v = state[3:]
        
        r_mag = np.linalg.norm(r)
        
        # Two-body acceleration
        a_2body = -self.mu * r / r_mag**3
        
        # J2 perturbation
        if self.include_j2 and r_mag > 0:
            z = r[2]
            r2 = r_mag**2
            
            # J2 acceleration components
            common_factor = 1.5 * self.J2 * self.mu * self.R_earth**2 / r_mag**5
            
            a_j2_x = common_factor * r[0] * (5 * z**2 / r2 - 1)
            a_j2_y = common_factor * r[1] * (5 * z**2 / r2 - 1)  
            a_j2_z = common_factor * z * (5 * z**2 / r2 - 3)
            
            a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z])
        else:
            a_j2 = np.zeros(3)
        
        # Total acceleration
        a_total = a_2body + a_j2
        
        # State derivative
        state_dot = np.concatenate([v, a_total])
        
        return state_dot
    
    def compute_state_transition_matrix(self, 
                                      state: np.ndarray, 
                                      dt: float) -> np.ndarray:
        """
        Compute linearized state transition matrix (STM).
        
        Args:
            state: Current state [x, y, z, vx, vy, vz]
            dt: Time step (seconds)
            
        Returns:
            6x6 state transition matrix
        """
        r = state[:3]
        r_mag = np.linalg.norm(r)
        
        if r_mag < 1000:  # Avoid singularity
            return np.eye(6)
        
        # Gravity gradient matrix (2-body)
        mu_r3 = self.mu / r_mag**3
        mu_r5 = self.mu / r_mag**5
        
        # Second-order gravity gradient
        A_rr = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    A_rr[i, j] = -mu_r3 + 3 * mu_r5 * r[i]**2
                else:
                    A_rr[i, j] = 3 * mu_r5 * r[i] * r[j]
        
        # J2 perturbation gradient (simplified)
        if self.include_j2:
            z = r[2]
            r2 = r_mag**2
            
            j2_factor = 1.5 * self.J2 * self.mu * self.R_earth**2 / r_mag**7
            
            # Add J2 contributions to gravity gradient (simplified)
            # Full implementation would require more complex derivatives
            A_rr += j2_factor * np.outer(r, r)
        
        # Construct continuous-time A matrix
        A_cont = np.zeros((6, 6))
        A_cont[:3, 3:] = np.eye(3)  # Position -> Velocity
        A_cont[3:, :3] = A_rr       # Position -> Acceleration
        
        # Discrete-time STM using matrix exponential
        STM = expm(A_cont * dt)
        
        return STM
    
    def integrate_state(self, 
                       state: np.ndarray, 
                       t_span: float, 
                       n_steps: int = None) -> np.ndarray:
        """
        Integrate state using orbital dynamics.
        
        Args:
            state: Initial state [x, y, z, vx, vy, vz]
            t_span: Integration time span (seconds)
            n_steps: Number of integration steps
            
        Returns:
            Final integrated state
        """
        if n_steps is None:
            n_steps = max(1, int(t_span / self.integration_step))
        
        # Use scipy's ODE solver
        try:
            sol = solve_ivp(
                self.dynamics,
                [0, t_span],
                state,
                method='DOP853',  # High-order Runge-Kutta
                rtol=1e-11,
                atol=1e-12,
                max_step=self.integration_step
            )
            
            if sol.success:
                return sol.y[:, -1]  # Final state
            else:
                warnings.warn(f"Integration failed: {sol.message}")
                return state  # Return initial state if integration fails
                
        except Exception as e:
            warnings.warn(f"Integration error: {e}")
            return state
    
    def predict_step(self, 
                    state: np.ndarray, 
                    covariance: np.ndarray, 
                    dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF prediction step.
        
        Args:
            state: Current state estimate
            covariance: Current covariance matrix
            dt: Time step (seconds)
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Propagate state
        predicted_state = self.integrate_state(state, dt)
        
        # Compute state transition matrix
        STM = self.compute_state_transition_matrix(state, dt)
        
        # Process noise covariance (simple model)
        dt2 = dt**2
        dt3 = dt**3 / 3.0
        dt4 = dt**4 / 4.0
        
        # Process noise affects acceleration, propagates to position/velocity
        Q = np.zeros((6, 6))
        
        # Position-position coupling
        Q[:3, :3] = self.process_noise_std**2 * dt4 * np.eye(3)
        
        # Position-velocity coupling  
        Q[:3, 3:] = self.process_noise_std**2 * dt3 * np.eye(3)
        Q[3:, :3] = self.process_noise_std**2 * dt3 * np.eye(3)
        
        # Velocity-velocity coupling
        Q[3:, 3:] = self.process_noise_std**2 * dt2 * np.eye(3)
        
        # Propagate covariance
        predicted_covariance = STM @ covariance @ STM.T + Q
        
        return predicted_state, predicted_covariance
    
    def update_step(self, 
                   predicted_state: np.ndarray,
                   predicted_covariance: np.ndarray,
                   measurement: np.ndarray,
                   measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """
        EKF update step with measurements.
        
        Args:
            predicted_state: Predicted state from forecast
            predicted_covariance: Predicted covariance matrix
            measurement: Measurement vector
            measurement_type: Type of measurement ('position', 'range', etc.)
            
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        if measurement_type == 'position':
            # Direct position measurement
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)  # Measure position directly
            
            # Measurement noise
            R = self.measurement_noise_std**2 * np.eye(3)
            
        elif measurement_type == 'range':
            # Range-only measurement
            r_pred = predicted_state[:3]
            r_mag = np.linalg.norm(r_pred)
            
            if r_mag > 0:
                H = np.zeros((1, 6))
                H[0, :3] = r_pred / r_mag  # Range measurement Jacobian
                R = np.array([[self.measurement_noise_std**2]])
            else:
                # Degenerate case
                return predicted_state, predicted_covariance
        else:
            # Unknown measurement type - no update
            return predicted_state, predicted_covariance
        
        # Innovation
        if measurement_type == 'position':
            innovation = measurement - predicted_state[:3]
        elif measurement_type == 'range':
            predicted_range = np.linalg.norm(predicted_state[:3])
            innovation = measurement - predicted_range
        
        # Innovation covariance
        S = H @ predicted_covariance @ H.T + R
        
        # Kalman gain
        try:
            K = predicted_covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular innovation covariance - skip update
            return predicted_state, predicted_covariance
        
        # State update
        updated_state = predicted_state + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        updated_covariance = I_KH @ predicted_covariance @ I_KH.T + K @ R @ K.T
        
        return updated_state, updated_covariance
    
    def predict_trajectory(self, 
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          measurements: Optional[List[np.ndarray]] = None,
                          measurement_times: Optional[np.ndarray] = None,
                          **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict trajectory using EKF.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            times: Time array in hours from initial epoch
            measurements: List of measurements (optional)
            measurement_times: Times of measurements in hours (optional)
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        times_seconds = times * 3600.0  # Convert hours to seconds
        n_points = len(times)
        
        # Initialize state and covariance
        state = initial_state.copy()
        
        # Initial covariance
        P = np.diag([
            self.initial_pos_uncertainty**2,  # x
            self.initial_pos_uncertainty**2,  # y  
            self.initial_pos_uncertainty**2,  # z
            self.initial_vel_uncertainty**2,  # vx
            self.initial_vel_uncertainty**2,  # vy
            self.initial_vel_uncertainty**2   # vz
        ])
        
        # Storage
        predicted_states = np.zeros((n_points, 6))
        uncertainties = np.zeros((n_points, 6))
        
        # Process measurements if provided
        measurement_dict = {}
        if measurements is not None and measurement_times is not None:
            for i, meas_time in enumerate(measurement_times):
                # Find closest time index
                time_idx = np.argmin(np.abs(times - meas_time))
                measurement_dict[time_idx] = measurements[i]
        
        # Initial values
        predicted_states[0] = state
        uncertainties[0] = np.sqrt(np.diag(P))
        
        # Propagate through time
        for i in range(1, n_points):
            dt = times_seconds[i] - times_seconds[i-1]
            
            # Prediction step
            state, P = self.predict_step(state, P, dt)
            
            # Update step if measurement available
            if i in measurement_dict:
                measurement = measurement_dict[i]
                state, P = self.update_step(state, P, measurement)
            
            # Store results
            predicted_states[i] = state
            uncertainties[i] = np.sqrt(np.diag(P))
        
        return predicted_states, uncertainties
    
    def get_name(self) -> str:
        """Return baseline model name."""
        return f"EKF{'_J2' if self.include_j2 else ''}"


def test_ekf_baseline():
    """Test EKF baseline implementation."""
    
    print("🧪 Testing EKF Baseline...")
    
    # Create test initial state (LEO orbit)
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    # Test times (0 to 2 hours)
    times = np.linspace(0, 2, 25)  # hours
    
    # Initialize baseline
    ekf = EKFBaseline(include_j2=True)
    
    # Create some synthetic measurements (every 30 minutes)
    measurement_times = np.array([0.5, 1.0, 1.5])  # hours
    measurements = []
    
    # Generate measurements from true trajectory (with noise)
    for t_meas in measurement_times:
        # Simple propagation for "true" position
        t_sec = t_meas * 3600
        true_state = ekf.integrate_state(initial_state, t_sec)
        
        # Add measurement noise
        meas_noise = np.random.normal(0, 500, 3)  # 500m noise
        measurement = true_state[:3] + meas_noise
        measurements.append(measurement)
    
    # Predict trajectory
    predicted_states, uncertainties = ekf.predict_trajectory(
        initial_state, times, measurements, measurement_times
    )
    
    print(f"✅ EKF prediction successful!")
    print(f"   Model: {ekf.get_name()}")
    print(f"   Initial position magnitude: {np.linalg.norm(initial_state[:3])/1000:.1f} km")
    print(f"   Final position magnitude: {np.linalg.norm(predicted_states[-1, :3])/1000:.1f} km")
    print(f"   Final position uncertainty: ±{uncertainties[-1, 0]/1000:.1f} km")
    print(f"   Final velocity uncertainty: ±{uncertainties[-1, 3]:.1f} m/s")
    
    # Check energy conservation
    initial_energy = 0.5 * np.sum(initial_state[3:]**2) - ekf.mu / np.linalg.norm(initial_state[:3])
    final_energy = 0.5 * np.sum(predicted_states[-1, 3:]**2) - ekf.mu / np.linalg.norm(predicted_states[-1, :3])
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
    
    print(f"   Energy conservation: {energy_drift:.2e} relative drift")
    
    return predicted_states, uncertainties


if __name__ == "__main__":
    test_ekf_baseline()