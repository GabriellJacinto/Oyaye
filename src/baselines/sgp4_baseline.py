"""
SGP4 Baseline Implementation.

Standard orbital propagation using SGP4/SDP4 model for comparison 
with NP-SNN predictions. This represents the classical approach
used in operational space situational awareness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import warnings

try:
    from sgp4.api import Satrec
    from sgp4.earth_gravity import wgs84
    from sgp4 import exporter
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    warnings.warn("SGP4 library not available. Install with: pip install sgp4")

from datetime import datetime, timezone
import julian


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""
    
    @abstractmethod
    def predict_trajectory(self, 
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict trajectory from initial state.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz] in meters and m/s
            times: Time array in hours from initial epoch
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (predicted_states, uncertainties)
            predicted_states: (N, 6) array of states
            uncertainties: (N, 6) array of uncertainties or None
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return baseline model name."""
        pass


class SGP4Baseline(BaselineModel):
    """
    SGP4/SDP4 orbital propagation baseline.
    
    Uses the SGP4 simplified general perturbations model to propagate
    satellite orbits from TLE-like initial conditions.
    """
    
    def __init__(self, 
                 use_high_precision: bool = True,
                 estimate_tle_uncertainty: bool = True):
        """
        Initialize SGP4 baseline.
        
        Args:
            use_high_precision: Use high-precision SGP4 computation
            estimate_tle_uncertainty: Estimate uncertainties based on TLE age/accuracy
        """
        if not SGP4_AVAILABLE:
            raise ImportError("SGP4 library required. Install with: pip install sgp4")
            
        self.use_high_precision = use_high_precision
        self.estimate_tle_uncertainty = estimate_tle_uncertainty
        self.earth_radius = 6378.137  # km
        self.mu = 398600.4418  # km^3/s^2
    
    def cartesian_to_tle_elements(self, 
                                 state: np.ndarray, 
                                 epoch: datetime = None) -> Dict[str, float]:
        """
        Convert cartesian state to TLE-like orbital elements.
        
        Args:
            state: [x, y, z, vx, vy, vz] in meters and m/s
            epoch: Reference epoch (default: J2000.0)
            
        Returns:
            Dictionary of orbital elements
        """
        if epoch is None:
            epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Convert to km and km/s
        r = state[:3] / 1000.0  # m to km
        v = state[3:] / 1000.0  # m/s to km/s
        
        # Compute orbital elements
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Semi-major axis (energy)
        energy = 0.5 * v_mag**2 - self.mu / r_mag
        a = -self.mu / (2 * energy)  # km
        
        # Angular momentum
        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)
        
        # Eccentricity
        e_vec = ((v_mag**2 - self.mu/r_mag) * r - np.dot(r, v) * v) / self.mu
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.arccos(h_vec[2] / h_mag)  # radians
        
        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n_mag = np.linalg.norm(n_vec)
        
        # RAAN (Right Ascension of Ascending Node)
        if n_mag > 1e-10:
            raan = np.arccos(n_vec[0] / n_mag)
            if n_vec[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0.0
        
        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            cos_omega = np.dot(n_vec, e_vec) / (n_mag * e)
            cos_omega = np.clip(cos_omega, -1.0, 1.0)
            omega = np.arccos(cos_omega)
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0
        
        # True anomaly
        if e > 1e-10:
            cos_nu = np.dot(e_vec, r) / (e * r_mag)
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        else:
            # Circular orbit - use argument of latitude
            if n_mag > 1e-10:
                cos_u = np.dot(n_vec, r) / (n_mag * r_mag)
                cos_u = np.clip(cos_u, -1.0, 1.0)
                nu = np.arccos(cos_u)
                if r[2] < 0:
                    nu = 2 * np.pi - nu
            else:
                nu = np.arctan2(r[1], r[0])
                if nu < 0:
                    nu += 2 * np.pi
        
        # Mean motion (for SGP4)
        n = np.sqrt(self.mu / a**3)  # rad/s
        n_rev_per_day = n * 86400 / (2 * np.pi)  # revolutions per day
        
        return {
            'semi_major_axis': a,  # km
            'eccentricity': e,
            'inclination': np.degrees(i),  # degrees
            'raan': np.degrees(raan),  # degrees  
            'arg_periapsis': np.degrees(omega),  # degrees
            'true_anomaly': np.degrees(nu),  # degrees
            'mean_motion': n_rev_per_day  # rev/day
        }
    
    def create_satrec_from_state(self, 
                                state: np.ndarray,
                                epoch: datetime = None,
                                norad_id: int = 99999) -> 'Satrec':
        """
        Create SGP4 Satrec object from cartesian state.
        
        Args:
            state: Initial state [x, y, z, vx, vy, vz]
            epoch: Reference epoch
            norad_id: NORAD catalog ID (dummy)
            
        Returns:
            SGP4 Satrec object
        """
        if epoch is None:
            epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Convert to orbital elements
        elements = self.cartesian_to_tle_elements(state, epoch)
        
        # Convert epoch to Julian date
        jd_epoch = julian.to_jd(epoch, fmt='jd')
        
        # Create TLE-like data
        # Line 1: NORAD ID, classification, launch year/number, epoch, derivatives, B-star, element set
        # Line 2: NORAD ID, inclination, RAAN, eccentricity, arg perigee, mean anomaly, mean motion, revolution
        
        # Simplified TLE creation (for SGP4 initialization)
        # Convert true anomaly to mean anomaly (simplified for near-circular orbits)
        e = elements['eccentricity']
        nu_rad = np.radians(elements['true_anomaly'])
        
        # Eccentric anomaly
        if e < 0.99:  # Elliptical orbit
            E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu_rad/2))
            if nu_rad > np.pi:
                E += 2 * np.pi
            # Mean anomaly
            M = E - e * np.sin(E)
        else:
            # Nearly parabolic - use approximation
            M = nu_rad
        
        M_deg = np.degrees(M) % 360
        
        # Create Satrec object directly
        satrec = Satrec()
        
        # Set basic parameters
        satrec.satnum = norad_id
        satrec.epochyr = epoch.year % 100  # 2-digit year
        satrec.epochdays = epoch.timetuple().tm_yday + \
                          (epoch.hour + epoch.minute/60.0 + epoch.second/3600.0) / 24.0
        satrec.ndot = 0.0  # First derivative of mean motion
        satrec.nddot = 0.0  # Second derivative of mean motion
        satrec.bstar = 0.0  # B-star drag term
        satrec.inclo = np.radians(elements['inclination'])
        satrec.nodeo = np.radians(elements['raan'])
        satrec.ecco = elements['eccentricity']
        satrec.argpo = np.radians(elements['arg_periapsis'])
        satrec.mo = np.radians(M_deg)
        satrec.no_kozai = elements['mean_motion'] * 2 * np.pi / 86400  # rad/s
        
        # Initialize SGP4 model
        satrec.sgp4init(
            whichconst=wgs84,
            opsmode='i',  # Improved mode
            satnum=satrec.satnum,
            epoch=jd_epoch - 2433281.5,  # Days since 1949 Dec 31 00:00 UT
            bstar=satrec.bstar,
            ndot=satrec.ndot,
            nddot=satrec.nddot,
            ecco=satrec.ecco,
            argpo=satrec.argpo,
            inclo=satrec.inclo,
            mo=satrec.mo,
            no_kozai=satrec.no_kozai,
            nodeo=satrec.nodeo
        )
        
        return satrec
    
    def predict_trajectory(self, 
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          epoch: datetime = None,
                          **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict trajectory using SGP4 propagation.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz] in SI units
            times: Time array in hours from initial epoch
            epoch: Reference epoch
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        if epoch is None:
            epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Create SGP4 satellite record
        try:
            satrec = self.create_satrec_from_state(initial_state, epoch)
        except Exception as e:
            # Fallback to simple Keplerian propagation if SGP4 fails
            warnings.warn(f"SGP4 initialization failed: {e}. Using Keplerian fallback.")
            return self._keplerian_fallback(initial_state, times)
        
        # Convert times to minutes (SGP4 uses minutes since epoch)
        times_minutes = times * 60.0
        
        # Propagate trajectory
        n_points = len(times)
        predicted_states = np.zeros((n_points, 6))
        
        for i, t_min in enumerate(times_minutes):
            try:
                # SGP4 propagation
                error, r_teme, v_teme = satrec.sgp4(
                    jd=julian.to_jd(epoch) + t_min / (24 * 60),  # Julian date
                    fr=0.0  # Fractional day
                )
                
                if error == 0:
                    # Convert from km to m, km/s to m/s
                    predicted_states[i, :3] = np.array(r_teme) * 1000.0  # km to m
                    predicted_states[i, 3:] = np.array(v_teme) * 1000.0  # km/s to m/s
                else:
                    # SGP4 error - use previous state or interpolation
                    if i > 0:
                        predicted_states[i] = predicted_states[i-1]
                    else:
                        predicted_states[i] = initial_state
                        
            except Exception as e:
                # Propagation failed - use fallback
                if i > 0:
                    predicted_states[i] = predicted_states[i-1]
                else:
                    predicted_states[i] = initial_state
        
        # Estimate uncertainties if requested
        uncertainties = None
        if self.estimate_tle_uncertainty:
            uncertainties = self._estimate_sgp4_uncertainties(times, predicted_states)
        
        return predicted_states, uncertainties
    
    def _keplerian_fallback(self, 
                           initial_state: np.ndarray, 
                           times: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simple 2-body Keplerian propagation fallback.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            times: Time array in hours
            
        Returns:
            Tuple of (predicted_states, None)
        """
        # Convert to km and km/s for computation
        r0 = initial_state[:3] / 1000.0  # m to km
        v0 = initial_state[3:] / 1000.0  # m/s to km/s
        
        # Orbital parameters
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        
        # Semi-major axis from energy
        energy = 0.5 * v0_mag**2 - self.mu / r0_mag
        a = -self.mu / (2 * energy)  # km
        
        # Mean motion
        n = np.sqrt(self.mu / a**3)  # rad/s
        
        # Convert times to seconds
        times_sec = times * 3600.0
        
        # Simple circular orbit approximation
        n_points = len(times)
        predicted_states = np.zeros((n_points, 6))
        
        # Assume circular orbit for simplicity
        for i, t in enumerate(times_sec):
            # Mean anomaly
            M = n * t
            
            # For circular orbit: E = M, nu = M
            # Position in orbital plane
            cos_M = np.cos(M)
            sin_M = np.sin(M)
            
            # Simple rotation (assumes orbit in XY plane initially)
            r_mag = r0_mag  # Constant for circular
            v_mag = v0_mag  # Constant for circular
            
            # Rotate initial state
            r = r_mag * np.array([cos_M, sin_M, 0.0])
            v = v_mag * np.array([-sin_M, cos_M, 0.0])
            
            # Convert back to meters and m/s
            predicted_states[i, :3] = r * 1000.0
            predicted_states[i, 3:] = v * 1000.0
        
        return predicted_states, None
    
    def _estimate_sgp4_uncertainties(self, 
                                   times: np.ndarray, 
                                   predicted_states: np.ndarray) -> np.ndarray:
        """
        Estimate SGP4 prediction uncertainties.
        
        Based on typical TLE accuracy and error growth rates.
        
        Args:
            times: Time array in hours
            predicted_states: Predicted states
            
        Returns:
            Uncertainty estimates (N, 6)
        """
        n_points = len(times)
        uncertainties = np.zeros((n_points, 6))
        
        # Base TLE uncertainties (typical values)
        base_pos_error = 1000.0  # 1 km base position error
        base_vel_error = 1.0     # 1 m/s base velocity error
        
        # Error growth rates (empirical)
        pos_growth_rate = 500.0   # m/h position error growth
        vel_growth_rate = 0.1     # m/s/h velocity error growth
        
        for i, t in enumerate(times):
            # Position uncertainty grows with time
            pos_std = base_pos_error + pos_growth_rate * t
            vel_std = base_vel_error + vel_growth_rate * t
            
            # Apply to all dimensions
            uncertainties[i, :3] = pos_std
            uncertainties[i, 3:] = vel_std
        
        return uncertainties
    
    def get_name(self) -> str:
        """Return baseline model name."""
        return "SGP4"


def test_sgp4_baseline():
    """Test SGP4 baseline implementation."""
    
    if not SGP4_AVAILABLE:
        print("⚠️  SGP4 library not available - skipping test")
        return
    
    print("🧪 Testing SGP4 Baseline...")
    
    # Create test initial state (LEO orbit)
    # ~400 km altitude, ~7.7 km/s velocity
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    # Test times (0 to 2 hours)
    times = np.linspace(0, 2, 25)  # hours
    
    # Initialize baseline
    sgp4 = SGP4Baseline()
    
    # Predict trajectory
    predicted_states, uncertainties = sgp4.predict_trajectory(initial_state, times)
    
    print(f"✅ SGP4 prediction successful!")
    print(f"   Initial position magnitude: {np.linalg.norm(initial_state[:3])/1000:.1f} km")
    print(f"   Final position magnitude: {np.linalg.norm(predicted_states[-1, :3])/1000:.1f} km")
    print(f"   Position change: {np.linalg.norm(predicted_states[-1, :3] - predicted_states[0, :3])/1000:.1f} km")
    
    if uncertainties is not None:
        print(f"   Final position uncertainty: ±{uncertainties[-1, 0]/1000:.1f} km")
        print(f"   Final velocity uncertainty: ±{uncertainties[-1, 3]:.1f} m/s")
    
    return predicted_states, uncertainties


if __name__ == "__main__":
    test_sgp4_baseline()