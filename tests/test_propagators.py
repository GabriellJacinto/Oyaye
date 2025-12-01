"""
Unit tests for orbital propagators.

Tests include:
- Two-body propagation accuracy
- Energy and angular momentum conservation
- J2 perturbation effects
- Comparison with analytical solutions
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics.propagators import TwoBodyPropagator, NumericalPropagator, OrbitState, validate_propagator
from physics.propagators import GM_EARTH, R_EARTH

class TestTwoBodyPropagator:
    """Test cases for two-body propagator."""
    
    def setup_method(self):
        """Setup for each test."""
        self.propagator = TwoBodyPropagator()
        
        # Circular LEO orbit
        self.r0_circular = np.array([7000e3, 0, 0])  # 7000 km altitude
        self.v0_circular = np.array([0, np.sqrt(GM_EARTH / np.linalg.norm(self.r0_circular)), 0])
        self.state_circular = OrbitState(r=self.r0_circular, v=self.v0_circular, t=0.0)
        
        # Elliptical orbit
        self.r0_elliptical = np.array([6678e3, 0, 0])  # Perigee at 300 km
        self.v0_elliptical = np.array([0, 10.25e3, 0])  # High velocity for elliptical
        self.state_elliptical = OrbitState(r=self.r0_elliptical, v=self.v0_elliptical, t=0.0)
    
    def test_circular_orbit_closure(self):
        """Test that circular orbit returns to starting position."""
        # Orbital period
        period = 2 * np.pi * np.sqrt(np.linalg.norm(self.r0_circular)**3 / GM_EARTH)
        t_span = np.linspace(0, period, 100)
        
        positions, velocities = self.propagator.propagate(self.state_circular, t_span)
        
        # Check closure (should return to starting point)
        pos_error = np.linalg.norm(positions[-1] - positions[0])
        vel_error = np.linalg.norm(velocities[-1] - velocities[0])
        
        # Allow for numerical errors
        assert pos_error < 1e3, f"Position closure error: {pos_error:.3e} m"
        assert vel_error < 1e0, f"Velocity closure error: {vel_error:.3e} m/s"
    
    def test_energy_conservation(self):
        """Test energy conservation for two-body problem."""
        # Short propagation
        t_span = np.linspace(0, 3600, 50)  # 1 hour
        
        positions, velocities = self.propagator.propagate(self.state_circular, t_span)
        
        # Compute energies
        energies = []
        for i in range(len(positions)):
            kinetic = 0.5 * np.dot(velocities[i], velocities[i])
            potential = -GM_EARTH / np.linalg.norm(positions[i])
            energies.append(kinetic + potential)
        
        # Check conservation
        energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        assert energy_drift < 1e-7, f"Energy drift: {energy_drift:.3e}"
    
    def test_angular_momentum_conservation(self):
        """Test angular momentum conservation."""
        t_span = np.linspace(0, 3600, 50)  # 1 hour
        
        positions, velocities = self.propagator.propagate(self.state_elliptical, t_span)
        
        # Compute angular momentum magnitudes
        h_magnitudes = []
        for i in range(len(positions)):
            h = np.cross(positions[i], velocities[i])
            h_magnitudes.append(np.linalg.norm(h))
        
        # Check conservation
        h_drift = abs(h_magnitudes[-1] - h_magnitudes[0]) / abs(h_magnitudes[0])
        assert h_drift < 1e-7, f"Angular momentum drift: {h_drift:.3e}"
    
    def test_kepler_third_law(self):
        """Test Kepler's third law for circular orbits."""
        radii = [7000e3, 8000e3, 10000e3]  # Different orbital radii
        
        for radius in radii:
            r0 = np.array([radius, 0, 0])
            v0 = np.array([0, np.sqrt(GM_EARTH / radius), 0])
            state = OrbitState(r=r0, v=v0, t=0.0)
            
            # Compute period numerically
            expected_period = 2 * np.pi * np.sqrt(radius**3 / GM_EARTH)
            
            # Propagate for expected period
            t_span = np.linspace(0, expected_period, 100)
            positions, _ = self.propagator.propagate(state, t_span)
            
            # Check return to starting point
            pos_error = np.linalg.norm(positions[-1] - positions[0])
            assert pos_error < 1e3, f"Kepler's law violation for r={radius/1e3:.1f} km"

class TestNumericalPropagator:
    """Test cases for numerical propagator with perturbations."""
    
    def setup_method(self):
        """Setup for each test."""
        self.propagator_j2 = NumericalPropagator(include_j2=True, include_drag=False, include_srp=False)
        self.propagator_clean = NumericalPropagator(include_j2=False, include_drag=False, include_srp=False)
        
        # Circular LEO orbit
        self.r0 = np.array([7000e3, 0, 0])  # 7000 km altitude  
        self.v0 = np.array([0, np.sqrt(GM_EARTH / np.linalg.norm(self.r0)), 0])
        self.state = OrbitState(r=self.r0, v=self.v0, t=0.0)
    
    def test_j2_perturbation_effect(self):
        """Test that J2 perturbation causes precession."""
        # Long propagation to see J2 effects
        t_span = np.linspace(0, 86400, 200)  # 1 day
        
        # Propagate with and without J2
        pos_j2, vel_j2 = self.propagator_j2.propagate(self.state, t_span)
        pos_clean, vel_clean = self.propagator_clean.propagate(self.state, t_span)
        
        # J2 should cause differences
        final_pos_diff = np.linalg.norm(pos_j2[-1] - pos_clean[-1])
        assert final_pos_diff > 1e3, f"J2 effect too small: {final_pos_diff:.1f} m"
        
        # But should still be bounded (J2 can cause ~1-2 km difference per day in LEO)
        assert final_pos_diff < 5e6, f"J2 effect too large: {final_pos_diff:.1f} m"
    
    def test_j2_regression_along_track(self):
        """Test J2 causes regression of ascending node."""
        # Inclined orbit to see nodal regression
        inclination = np.radians(45)  # 45 degree inclination
        r0_inclined = np.array([7000e3, 0, 0])
        v0_inclined = np.array([0, 
                               np.sqrt(GM_EARTH / np.linalg.norm(r0_inclined)) * np.cos(inclination),
                               np.sqrt(GM_EARTH / np.linalg.norm(r0_inclined)) * np.sin(inclination)])
        
        state_inclined = OrbitState(r=r0_inclined, v=v0_inclined, t=0.0)
        
        # Multiple orbits to accumulate J2 effects
        n_orbits = 5
        period = 2 * np.pi * np.sqrt(np.linalg.norm(r0_inclined)**3 / GM_EARTH)
        t_span = np.linspace(0, n_orbits * period, 100)
        
        positions, _ = self.propagator_j2.propagate(state_inclined, t_span)
        
        # Check that orbit precesses (position at same orbital phase should drift)
        # This is a qualitative test - J2 should cause measurable drift
        orbit_spacing = len(positions) // n_orbits
        phase_positions = positions[::orbit_spacing]
        
        if len(phase_positions) >= 3:
            # Check that positions at same phase drift due to precession
            drift_1_2 = np.linalg.norm(phase_positions[1] - phase_positions[0])
            drift_2_3 = np.linalg.norm(phase_positions[2] - phase_positions[1])
            
            # J2 should cause systematic drift
            assert drift_1_2 > 1e2, "No significant J2 precession detected"
    
    def test_energy_conservation_with_j2(self):
        """Test energy conservation with J2 (should be preserved)."""
        t_span = np.linspace(0, 7200, 100)  # 2 hours
        
        positions, velocities = self.propagator_j2.propagate(self.state, t_span)
        
        # Compute total energy (kinetic + potential with J2)
        energies = []
        for i in range(len(positions)):
            energy = self.propagator_j2.energy(positions[i], velocities[i])
            energies.append(energy)
        
        # J2 is conservative, so energy should be conserved (allow for small numerical drift)
        energy_drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        assert energy_drift < 1e-5, f"Energy not conserved with J2: {energy_drift:.3e}"

class TestPropagatorValidation:
    """Test the validation framework."""
    
    def test_validation_two_body(self):
        """Test validation framework on two-body propagator."""
        propagator = TwoBodyPropagator()
        
        results = validate_propagator(propagator)
        
        # Check that validation ran successfully
        assert 'energy_conservation' in results
        
        # Two-body should have excellent conservation
        if 'energy_conservation' in results:
            assert results['energy_conservation']['max'] < 1e-9
    
    def test_validation_numerical(self):
        """Test validation framework on numerical propagator."""
        propagator = NumericalPropagator(include_j2=True)
        
        results = validate_propagator(propagator)
        
        # Check that validation ran
        assert isinstance(results, dict)
        
        # Should have some conservation metrics
        if 'energy_conservation' in results:
            # J2 is conservative so should still conserve energy well
            assert results['energy_conservation']['max'] < 1e-6

@pytest.mark.integration
class TestIntegrationPropagators:
    """Integration tests comparing different propagators."""
    
    def test_two_body_vs_numerical_clean(self):
        """Compare two-body and numerical (no perturbations) propagators."""
        propagator_2body = TwoBodyPropagator()
        propagator_num = NumericalPropagator(include_j2=False, include_drag=False, include_srp=False)
        
        # Circular orbit
        r0 = np.array([7000e3, 0, 0])
        v0 = np.array([0, np.sqrt(GM_EARTH / np.linalg.norm(r0)), 0])
        state = OrbitState(r=r0, v=v0, t=0.0)
        
        t_span = np.linspace(0, 3600, 50)  # 1 hour
        
        pos_2body, vel_2body = propagator_2body.propagate(state, t_span)
        pos_num, vel_num = propagator_num.propagate(state, t_span)
        
        # Should be very similar
        pos_diff = np.linalg.norm(pos_2body[-1] - pos_num[-1])
        vel_diff = np.linalg.norm(vel_2body[-1] - vel_num[-1])
        
        assert pos_diff < 1e1, f"Two-body vs numerical position difference: {pos_diff:.3e} m"
        assert vel_diff < 1e-2, f"Two-body vs numerical velocity difference: {vel_diff:.3e} m/s"

if __name__ == "__main__":
    # Run basic tests
    print("Testing orbital propagators...")
    
    # Test two-body propagator
    test_2body = TestTwoBodyPropagator()
    test_2body.setup_method()
    
    try:
        test_2body.test_circular_orbit_closure()
        print("✓ Two-body circular orbit closure")
        
        test_2body.test_energy_conservation()
        print("✓ Two-body energy conservation")
        
        test_2body.test_angular_momentum_conservation()
        print("✓ Two-body angular momentum conservation")
        
    except Exception as e:
        print(f"✗ Two-body test failed: {e}")
    
    # Test numerical propagator
    test_num = TestNumericalPropagator()
    test_num.setup_method()
    
    try:
        test_num.test_j2_perturbation_effect()
        print("✓ J2 perturbation effect")
        
        test_num.test_energy_conservation_with_j2()
        print("✓ Energy conservation with J2")
        
    except Exception as e:
        print(f"✗ Numerical propagator test failed: {e}")
    
    print("Propagator tests completed!")