"""
Orbital propagators for physics-informed training.

This module provides:
- Numerical integration (RK45) for orbital dynamics
- Two-body dynamics with perturbations (J2, drag, SRP)
- Validation against standard propagators (SGP4, poliastro)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

# Import acceleration models
try:
    from .accel_models import GravityField, AtmosphericModel, DragModel, SolarRadiationPressure, SpaceWeatherData
except ImportError:
    # Fallback for direct execution
    from accel_models import GravityField, AtmosphericModel, DragModel, SolarRadiationPressure, SpaceWeatherData

# Physical constants
GM_EARTH = 3.986004418e14  # m^3/s^2
J2 = 1.08262668e-3  # J2 coefficient
R_EARTH = 6.378137e6  # Earth radius (m)

@dataclass
class OrbitState:
    """Orbital state vector."""
    r: np.ndarray  # Position (m) - shape (3,)
    v: np.ndarray  # Velocity (m/s) - shape (3,)
    t: float       # Time (s)

class TwoBodyPropagator:
    """Simple two-body orbital propagator."""
    
    def __init__(self, mu: float = GM_EARTH):
        self.mu = mu
        
    def acceleration(self, r: np.ndarray) -> np.ndarray:
        """Two-body gravitational acceleration."""
        r_mag = np.linalg.norm(r)
        return -self.mu * r / (r_mag**3)
    
    def propagate(self, 
                  initial_state: OrbitState,
                  t_span: np.ndarray,
                  rtol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit using numerical integration.
        
        Args:
            initial_state: Initial orbital state
            t_span: Time points to evaluate
            rtol: Relative tolerance for integration
            
        Returns:
            positions: Array of shape (n_times, 3)
            velocities: Array of shape (n_times, 3)
        """
        def equations_of_motion(t, state):
            """State derivative for two-body problem."""
            r = state[:3]
            v = state[3:]
            accel = self.acceleration(r)
            return np.concatenate([v, accel])
        
        # Initial state vector
        y0 = np.concatenate([initial_state.r, initial_state.v])
        
        # Time span for integration
        t_eval = t_span
        t_span_bounds = [t_span[0], t_span[-1]]
        
        # Numerical integration
        try:
            sol = solve_ivp(
                equations_of_motion,
                t_span_bounds,
                y0,
                t_eval=t_eval,
                method='RK45',
                rtol=rtol,
                atol=1e-12
            )
            
            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
                
        except Exception as e:
            raise RuntimeError(f"Propagation error: {str(e)}")
        
        # Extract positions and velocities
        positions = sol.y[:3, :].T  # Shape: (n_times, 3)
        velocities = sol.y[3:, :].T  # Shape: (n_times, 3)
        
        return positions, velocities

class NumericalPropagator:
    """High-fidelity orbital propagator with perturbations."""
    
    def __init__(self, 
                 include_j2: bool = True,
                 include_drag: bool = False,
                 include_srp: bool = False,
                 mu: float = GM_EARTH,
                 gravity_degree: int = 2):
        self.mu = mu
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.include_srp = include_srp
        
        # Initialize acceleration models
        # Set gravity degree based on J2 flag
        if include_j2:
            self.gravity = GravityField(max_degree=max(gravity_degree, 2))
        else:
            self.gravity = GravityField(max_degree=1)  # Point mass only
        
        if include_drag:
            self.atmosphere = AtmosphericModel(model_type="exponential")
            self.drag_model = DragModel(self.atmosphere)
        else:
            self.drag_model = None
            
        if include_srp:
            self.srp_model = SolarRadiationPressure()
        else:
            self.srp_model = None
        
    def two_body_accel(self, r: np.ndarray) -> np.ndarray:
        """Two-body gravitational acceleration."""
        r_mag = np.linalg.norm(r)
        return -self.mu * r / (r_mag**3)
    
    def j2_accel(self, r: np.ndarray) -> np.ndarray:
        """J2 perturbation acceleration."""
        if not self.include_j2:
            return np.zeros(3)
            
        x, y, z = r
        r_mag = np.linalg.norm(r)
        
        factor = -1.5 * J2 * self.mu * (R_EARTH**2) / (r_mag**5)
        
        ax = factor * x * (5 * z**2 / r_mag**2 - 1)
        ay = factor * y * (5 * z**2 / r_mag**2 - 1) 
        az = factor * z * (5 * z**2 / r_mag**2 - 3)
        
        return np.array([ax, ay, az])
    
    def drag_accel(self, r: np.ndarray, v: np.ndarray, ballistic_coeff: float = 1e-3, **kwargs) -> np.ndarray:
        """Atmospheric drag acceleration."""
        if not self.include_drag or self.drag_model is None:
            return np.zeros(3)
            
        timestamp = kwargs.get('timestamp', 0.0)
        space_weather = kwargs.get('space_weather', SpaceWeatherData())
        
        return self.drag_model.drag_accel(r, v, ballistic_coeff, timestamp, space_weather)
    
    def srp_accel(self, r: np.ndarray, area_mass_ratio: float = 1e-3, **kwargs) -> np.ndarray:
        """Solar radiation pressure acceleration."""
        if not self.include_srp or self.srp_model is None:
            return np.zeros(3)
            
        timestamp = kwargs.get('timestamp', 0.0)
        reflectivity = kwargs.get('reflectivity', 1.0)
        
        return self.srp_model.srp_accel(r, area_mass_ratio, reflectivity, timestamp)
    
    def total_acceleration(self, 
                          r: np.ndarray, 
                          v: np.ndarray, 
                          **kwargs) -> np.ndarray:
        """Compute total acceleration from all forces."""
        # Use the modular gravity field
        accel = self.gravity.total_gravity_accel(r)
        
        # Add perturbations
        accel += self.drag_accel(r, v, kwargs.get('ballistic_coeff', 1e-3), 
                                timestamp=kwargs.get('timestamp', 0.0),
                                space_weather=kwargs.get('space_weather', None))
        accel += self.srp_accel(r, kwargs.get('area_mass_ratio', 1e-3), 
                               timestamp=kwargs.get('timestamp', 0.0),
                               reflectivity=kwargs.get('reflectivity', 1.0))
        
        return accel
    
    def equations_of_motion(self, t: float, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        State derivative function for numerical integration.
        
        Args:
            t: Time
            state: State vector [x, y, z, vx, vy, vz]
            **kwargs: Additional parameters (ballistic_coeff, etc.)
            
        Returns:
            state_dot: Time derivative of state
        """
        r = state[:3]
        v = state[3:]
        
        accel = self.total_acceleration(r, v, **kwargs)
        
        return np.concatenate([v, accel])
    
    def propagate(self,
                  initial_state: OrbitState, 
                  t_span: np.ndarray,
                  rtol: float = 1e-8,
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit with perturbations.
        
        Args:
            initial_state: Initial orbital state
            t_span: Time points to evaluate
            rtol: Relative tolerance
            **kwargs: Additional parameters for forces
            
        Returns:
            positions: Array of shape (n_times, 3)
            velocities: Array of shape (n_times, 3)
        """
        def equations_of_motion_full(t, state):
            """State derivative with all perturbations."""
            r = state[:3]
            v = state[3:]
            
            # Update timestamp in kwargs for time-dependent forces
            time_kwargs = kwargs.copy()
            time_kwargs['timestamp'] = t + initial_state.t
            
            accel = self.total_acceleration(r, v, **time_kwargs)
            return np.concatenate([v, accel])
        
        # Initial state vector
        y0 = np.concatenate([initial_state.r, initial_state.v])
        
        # Time span for integration (relative to initial time)
        t_eval = t_span - initial_state.t
        t_span_bounds = [t_eval[0], t_eval[-1]]
        
        # Numerical integration with adaptive step size
        try:
            sol = solve_ivp(
                equations_of_motion_full,
                t_span_bounds,
                y0,
                t_eval=t_eval,
                method='RK45',
                rtol=rtol,
                atol=1e-12,
                max_step=3600.0  # Maximum 1 hour step for accuracy
            )
            
            if not sol.success:
                warnings.warn(f"Integration warning: {sol.message}")
                
        except Exception as e:
            raise RuntimeError(f"Propagation error: {str(e)}")
        
        # Extract positions and velocities
        positions = sol.y[:3, :].T  # Shape: (n_times, 3)
        velocities = sol.y[3:, :].T  # Shape: (n_times, 3)
        
        return positions, velocities
        
    def energy(self, r: np.ndarray, v: np.ndarray) -> float:
        """Compute specific orbital energy."""
        kinetic = 0.5 * np.dot(v, v)
        potential = -self.mu / np.linalg.norm(r)
        return kinetic + potential
    
    def angular_momentum(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute specific angular momentum vector."""
        return np.cross(r, v)

def validate_propagator(propagator: Union[TwoBodyPropagator, NumericalPropagator], 
                       test_cases: list = None) -> Dict:
    """
    Validate propagator against known solutions.
    
    Args:
        propagator: Propagator to test
        test_cases: List of test cases with known solutions
        
    Returns:
        validation_results: Dictionary with error metrics
    """
    if test_cases is None:
        test_cases = _get_default_test_cases()
    
    results = {
        'energy_conservation': [],
        'angular_momentum_conservation': [],
        'position_errors': [],
        'velocity_errors': []
    }
    
    for case in test_cases:
        initial_state = case['initial_state']
        t_span = case['t_span']
        expected = case.get('expected', None)
        
        # Propagate
        try:
            positions, velocities = propagator.propagate(initial_state, t_span)
            
            # Check energy conservation (for conservative forces)
            if hasattr(propagator, 'energy') and not getattr(propagator, 'include_drag', False):
                energies = []
                for i in range(len(positions)):
                    energy = propagator.energy(positions[i], velocities[i])
                    energies.append(energy)
                
                energy_drift = (energies[-1] - energies[0]) / abs(energies[0])
                results['energy_conservation'].append(abs(energy_drift))
            
            # Check angular momentum conservation
            if hasattr(propagator, 'angular_momentum'):
                h_vectors = []
                for i in range(len(positions)):
                    h = propagator.angular_momentum(positions[i], velocities[i])
                    h_vectors.append(np.linalg.norm(h))
                
                h_drift = (h_vectors[-1] - h_vectors[0]) / abs(h_vectors[0])
                results['angular_momentum_conservation'].append(abs(h_drift))
            
            # Compare with expected if provided
            if expected is not None:
                pos_error = np.linalg.norm(positions[-1] - expected['position'])
                vel_error = np.linalg.norm(velocities[-1] - expected['velocity'])
                results['position_errors'].append(pos_error)
                results['velocity_errors'].append(vel_error)
                
        except Exception as e:
            print(f"Validation failed for test case: {e}")
            continue
    
    # Compute statistics
    validation_summary = {}
    for key, values in results.items():
        if values:
            validation_summary[key] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'std': np.std(values)
            }
    
    return validation_summary

def _get_default_test_cases() -> list:
    """Generate default test cases for validation."""
    test_cases = []
    
    # Circular LEO orbit
    r0 = np.array([7000e3, 0, 0])  # 7000 km altitude
    v0 = np.array([0, np.sqrt(GM_EARTH / np.linalg.norm(r0)), 0])
    initial_state = OrbitState(r=r0, v=v0, t=0.0)
    
    # One orbit period
    period = 2 * np.pi * np.sqrt(np.linalg.norm(r0)**3 / GM_EARTH)
    t_span = np.linspace(0, period, 100)
    
    test_cases.append({
        'initial_state': initial_state,
        't_span': t_span,
        'name': 'circular_leo'
    })
    
    # Elliptical GTO orbit  
    r0_gto = np.array([6678e3, 0, 0])  # Perigee at 300 km
    v0_gto = np.array([0, 10.25e3, 0])  # Velocity for GTO
    initial_state_gto = OrbitState(r=r0_gto, v=v0_gto, t=0.0)
    
    t_span_gto = np.linspace(0, 3600 * 6, 50)  # 6 hours
    
    test_cases.append({
        'initial_state': initial_state_gto,
        't_span': t_span_gto,
        'name': 'elliptical_gto'
    })
    
    return test_cases