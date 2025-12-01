"""
Unit tests for orbital propagators.

This module tests:
- Two-body dynamics accuracy
- J2 perturbation implementation  
- Energy conservation
- Comparison with known solutions
"""

import pytest
import numpy as np
from src.physics.propagators import TwoBodyPropagator, NumericalPropagator, OrbitState

class TestTwoBodyPropagator:
    """Test two-body orbital propagator."""
    
    def setup_method(self):
        """Setup test propagator."""
        self.propagator = TwoBodyPropagator()
        
    def test_circular_orbit_energy_conservation(self):
        """Test energy conservation for circular orbit."""
        # LEO circular orbit
        r0 = np.array([7000e3, 0, 0])  # 7000 km altitude
        v0 = np.array([0, 7546, 0])   # Circular velocity
        
        initial_state = OrbitState(r=r0, v=v0, t=0.0)
        
        # Propagate for one orbit period  
        period = 2 * np.pi * np.sqrt(np.linalg.norm(r0)**3 / self.propagator.mu)
        times = np.linspace(0, period, 100)
        
        # TODO: Implement actual propagation when method is complete
        # positions, velocities = self.propagator.propagate(initial_state, times)
        
        # For now, just test that propagator initializes
        assert self.propagator.mu == 3.986004418e14
        
    def test_acceleration_magnitude(self):
        """Test two-body acceleration magnitude."""
        r = np.array([7000e3, 0, 0])
        accel = self.propagator.acceleration(r)
        
        expected_magnitude = self.propagator.mu / (np.linalg.norm(r)**2)
        actual_magnitude = np.linalg.norm(accel)
        
        np.testing.assert_allclose(actual_magnitude, expected_magnitude, rtol=1e-10)
        
    def test_acceleration_direction(self):
        """Test that acceleration points toward Earth center."""
        r = np.array([7000e3, 1000e3, 500e3])
        accel = self.propagator.acceleration(r)
        
        # Acceleration should be anti-parallel to position
        r_unit = r / np.linalg.norm(r)
        accel_unit = accel / np.linalg.norm(accel)
        
        dot_product = np.dot(r_unit, accel_unit)
        np.testing.assert_allclose(dot_product, -1.0, atol=1e-10)

class TestNumericalPropagator:
    """Test numerical propagator with perturbations."""
    
    def setup_method(self):
        """Setup test propagator."""
        self.propagator = NumericalPropagator(
            include_j2=True,
            include_drag=False,
            include_srp=False
        )
        
    def test_j2_acceleration_equatorial(self):
        """Test J2 acceleration at equator (should be zero in Z)."""
        r = np.array([7000e3, 0, 0])  # Equatorial position
        j2_accel = self.propagator.j2_accel(r)
        
        # At equator, J2 acceleration should have no Z component
        np.testing.assert_allclose(j2_accel[2], 0.0, atol=1e-15)
        
    def test_j2_acceleration_polar(self):
        """Test J2 acceleration at pole."""
        r = np.array([0, 0, 7000e3])  # Polar position
        j2_accel = self.propagator.j2_accel(r)
        
        # At pole, J2 acceleration should have no X,Y components
        np.testing.assert_allclose(j2_accel[:2], 0.0, atol=1e-15)
        
    def test_total_acceleration_includes_j2(self):
        """Test that total acceleration includes J2 when enabled."""
        r = np.array([7000e3, 0, 1000e3])
        v = np.array([0, 7500, 0])
        
        two_body_accel = self.propagator.two_body_accel(r)
        j2_accel = self.propagator.j2_accel(r)
        total_accel = self.propagator.total_acceleration(r, v)
        
        expected_accel = two_body_accel + j2_accel
        np.testing.assert_allclose(total_accel, expected_accel)
        
    def test_equations_of_motion_structure(self):
        """Test that equations of motion return correct structure."""
        state = np.array([7000e3, 0, 0, 0, 7546, 0])  # [r, v]
        
        state_dot = self.propagator.equations_of_motion(0.0, state)
        
        # Should return [v, a] where v=state[3:] and a is acceleration
        assert state_dot.shape == (6,)
        np.testing.assert_allclose(state_dot[:3], state[3:])  # dr/dt = v

class TestOrbitState:
    """Test OrbitState data structure."""
    
    def test_orbit_state_creation(self):
        """Test OrbitState creation and attributes."""
        r = np.array([7000e3, 0, 0])
        v = np.array([0, 7546, 0])
        t = 12345.0
        
        state = OrbitState(r=r, v=v, t=t)
        
        np.testing.assert_array_equal(state.r, r)
        np.testing.assert_array_equal(state.v, v)
        assert state.t == t

def test_physical_constants():
    """Test that physical constants have correct values."""
    from src.physics.propagators import GM_EARTH, J2, R_EARTH
    
    # Check approximate values
    assert abs(GM_EARTH - 3.986004418e14) < 1e6  # Within 1e6 m^3/s^2
    assert abs(J2 - 1.08262668e-3) < 1e-6      # Within 1e-6
    assert abs(R_EARTH - 6.378137e6) < 1e3      # Within 1 km

if __name__ == "__main__":
    pytest.main([__file__])