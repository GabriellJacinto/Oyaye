"""
Modular acceleration models for orbital mechanics.

This module provides:
- Individual acceleration models (gravity, drag, SRP, third-body)
- Atmospheric density models 
- Solar flux and space weather interfaces
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import math

# Constants
GM_EARTH = 3.986004418e14  # m^3/s^2
GM_SUN = 1.32712442018e20  # m^3/s^2
GM_MOON = 4.9048695e12     # m^3/s^2
R_EARTH = 6.378137e6       # m
R_SUN = 6.96e8             # m
AU = 1.495978707e11        # m (1 AU)
SOLAR_FLUX = 1361.0        # W/m^2 (solar constant)
SPEED_OF_LIGHT = 299792458 # m/s

@dataclass
class SpaceWeatherData:
    """Space weather parameters."""
    f107: float = 150.0      # F10.7 solar flux (sfu)
    f107_avg: float = 150.0  # 81-day average F10.7
    ap: float = 15.0         # Ap geomagnetic index
    timestamp: float = 0.0   # Time (for temporal variations)

class GravityField:
    """Earth gravity field with spherical harmonics."""
    
    def __init__(self, max_degree: int = 2):
        self.max_degree = max_degree
        self.mu = GM_EARTH
        self.R_earth = R_EARTH
        
        # Store commonly used coefficients
        self.J2 = 1.08262668e-3
        self.J3 = -2.53265648e-6
        self.J4 = -1.61962159e-6
    
    def central_body_accel(self, r: np.ndarray) -> np.ndarray:
        """Point mass gravitational acceleration."""
        r_mag = np.linalg.norm(r)
        return -self.mu * r / (r_mag**3)
    
    def j2_accel(self, r: np.ndarray) -> np.ndarray:
        """J2 zonal harmonic acceleration."""
        x, y, z = r
        r_mag = np.linalg.norm(r)
        
        factor = -1.5 * self.J2 * self.mu * (self.R_earth**2) / (r_mag**5)
        
        ax = factor * x * (5 * z**2 / r_mag**2 - 1)
        ay = factor * y * (5 * z**2 / r_mag**2 - 1)
        az = factor * z * (5 * z**2 / r_mag**2 - 3)
        
        return np.array([ax, ay, az])
    
    def j3_accel(self, r: np.ndarray) -> np.ndarray:
        """J3 zonal harmonic acceleration."""
        # TODO: Implement J3 acceleration
        return np.zeros(3)
    
    def j4_accel(self, r: np.ndarray) -> np.ndarray:
        """J4 zonal harmonic acceleration.""" 
        # TODO: Implement J4 acceleration
        return np.zeros(3)
    
    def total_gravity_accel(self, r: np.ndarray) -> np.ndarray:
        """Total gravitational acceleration."""
        accel = self.central_body_accel(r)
        
        if self.max_degree >= 2:
            accel += self.j2_accel(r)
        if self.max_degree >= 3:
            accel += self.j3_accel(r)
        if self.max_degree >= 4:
            accel += self.j4_accel(r)
            
        return accel

class AtmosphericModel:
    """Atmospheric density model."""
    
    def __init__(self, model_type: str = "exponential"):
        self.model_type = model_type
        
        # Exponential model parameters
        self.rho0 = 1.225        # kg/m^3 (sea level density)
        self.h0 = 0.0            # m (reference altitude)
        self.H = 8500.0          # m (scale height)
    
    def exponential_density(self, altitude: float) -> float:
        """Simple exponential atmosphere model."""
        return self.rho0 * math.exp(-(altitude - self.h0) / self.H)
    
    def nrlmsise00_density(self, 
                          r: np.ndarray, 
                          timestamp: float,
                          space_weather: SpaceWeatherData) -> float:
        """
        NRLMSISE-00 atmospheric density model.
        
        TODO: This is a placeholder - real implementation would
        require the NRLMSISE-00 Fortran code or Python wrapper.
        """
        altitude = np.linalg.norm(r) - R_EARTH
        
        if altitude < 0:
            return self.rho0  # Surface density
        
        # Simplified density with space weather effects
        base_density = self.exponential_density(altitude)
        
        # Scale by space weather (very simplified)
        f107_factor = space_weather.f107 / 150.0
        ap_factor = 1.0 + 0.01 * space_weather.ap
        
        return base_density * f107_factor * ap_factor
    
    def get_density(self, 
                   r: np.ndarray,
                   timestamp: float = 0.0,
                   space_weather: Optional[SpaceWeatherData] = None) -> float:
        """Get atmospheric density at position."""
        if space_weather is None:
            space_weather = SpaceWeatherData()
            
        if self.model_type == "exponential":
            altitude = np.linalg.norm(r) - R_EARTH
            return self.exponential_density(altitude)
        elif self.model_type == "nrlmsise00":
            return self.nrlmsise00_density(r, timestamp, space_weather)
        else:
            raise ValueError(f"Unknown atmospheric model: {self.model_type}")

class DragModel:
    """Atmospheric drag acceleration."""
    
    def __init__(self, 
                 atmosphere: AtmosphericModel,
                 drag_coeff: float = 2.2):
        self.atmosphere = atmosphere
        self.Cd = drag_coeff
    
    def drag_accel(self,
                   r: np.ndarray,
                   v: np.ndarray,
                   ballistic_coeff: float,  # Cd * A / m (m^2/kg)
                   timestamp: float = 0.0,
                   space_weather: Optional[SpaceWeatherData] = None) -> np.ndarray:
        """
        Atmospheric drag acceleration.
        
        Args:
            r: Position vector (m)
            v: Velocity vector (m/s)  
            ballistic_coeff: Ballistic coefficient Cd*A/m (m^2/kg)
            timestamp: Current time
            space_weather: Space weather parameters
            
        Returns:
            drag acceleration vector (m/s^2)
        """
        # Get atmospheric density
        rho = self.atmosphere.get_density(r, timestamp, space_weather)
        
        # Relative velocity (assuming co-rotating atmosphere)
        # TODO: Add Earth rotation effects
        v_rel = v
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag < 1e-6:
            return np.zeros(3)
        
        # Drag acceleration = -0.5 * rho * Cd * A/m * v_rel * |v_rel|
        drag_accel = -0.5 * rho * ballistic_coeff * v_rel * v_rel_mag
        
        return drag_accel

class SolarRadiationPressure:
    """Solar radiation pressure model."""
    
    def __init__(self, 
                 solar_flux: float = SOLAR_FLUX,
                 speed_of_light: float = SPEED_OF_LIGHT):
        self.Phi = solar_flux
        self.c = speed_of_light
        
    def sun_position_ecliptic(self, timestamp: float) -> np.ndarray:
        """
        Simplified Sun position in Earth-centered frame.
        
        Args:
            timestamp: Time since J2000 epoch (seconds)
            
        Returns:
            Sun position vector (m)
        """
        # Very simplified - assumes circular orbit
        days = timestamp / 86400.0  # Convert to days
        mean_anomaly = 2 * math.pi * days / 365.25
        
        sun_distance = AU
        sun_pos = np.array([
            sun_distance * math.cos(mean_anomaly),
            sun_distance * math.sin(mean_anomaly), 
            0.0
        ])
        
        return sun_pos
    
    def shadow_function(self, r_sat: np.ndarray, r_sun: np.ndarray) -> float:
        """
        Compute shadow function (0 = umbra, 1 = sunlight).
        
        Args:
            r_sat: Satellite position
            r_sun: Sun position
            
        Returns:
            Shadow function value [0, 1]
        """
        # TODO: Implement proper shadow function with penumbra
        # Simple umbra check for now
        
        # Vector from Earth to satellite
        sat_unit = r_sat / np.linalg.norm(r_sat)
        
        # Vector from Earth to Sun
        sun_unit = -r_sun / np.linalg.norm(r_sun)
        
        # Check if satellite is in Earth's shadow
        dot_product = np.dot(sat_unit, sun_unit)
        
        if dot_product > 0:
            return 1.0  # In sunlight
        
        # Check if Earth blocks the Sun
        earth_angular_radius = R_EARTH / np.linalg.norm(r_sat)
        sun_sat_angle = math.acos(abs(dot_product))
        
        if sun_sat_angle < earth_angular_radius:
            return 0.0  # In umbra
        else:
            return 1.0  # In sunlight
    
    def srp_accel(self,
                  r: np.ndarray,
                  area_mass_ratio: float,  # A/m (m^2/kg)
                  reflectivity: float = 1.0,
                  timestamp: float = 0.0) -> np.ndarray:
        """
        Solar radiation pressure acceleration.
        
        Args:
            r: Satellite position (m)
            area_mass_ratio: Area-to-mass ratio (m^2/kg)
            reflectivity: Surface reflectivity [0, 2] (1 = absorbing, 2 = perfect mirror)
            timestamp: Current time
            
        Returns:
            SRP acceleration vector (m/s^2)
        """
        # Sun position
        r_sun = self.sun_position_ecliptic(timestamp)
        
        # Shadow function
        shadow = self.shadow_function(r, r_sun)
        
        if shadow < 1e-6:
            return np.zeros(3)  # In shadow
        
        # Unit vector from Sun to satellite
        sun_to_sat = r - r_sun
        sun_to_sat_unit = sun_to_sat / np.linalg.norm(sun_to_sat)
        
        # Distance to Sun
        sun_distance = np.linalg.norm(sun_to_sat)
        
        # Solar flux at satellite distance
        flux_at_sat = self.Phi * (AU / sun_distance)**2
        
        # SRP acceleration
        pressure = flux_at_sat / self.c
        srp_accel = shadow * pressure * area_mass_ratio * reflectivity * sun_to_sat_unit
        
        return srp_accel

class ThirdBodyPerturbations:
    """Third-body gravitational perturbations (Sun, Moon)."""
    
    def __init__(self):
        self.mu_sun = GM_SUN
        self.mu_moon = GM_MOON
    
    def third_body_accel(self,
                        r_sat: np.ndarray,
                        r_body: np.ndarray,
                        mu_body: float) -> np.ndarray:
        """
        Third-body gravitational acceleration.
        
        Args:
            r_sat: Satellite position relative to Earth
            r_body: Third body position relative to Earth
            mu_body: Third body gravitational parameter
            
        Returns:
            Third-body acceleration
        """
        # Vector from third body to satellite
        r_rel = r_sat - r_body
        
        # Accelerations
        direct_accel = -mu_body * r_rel / (np.linalg.norm(r_rel)**3)
        indirect_accel = mu_body * r_body / (np.linalg.norm(r_body)**3)
        
        return direct_accel + indirect_accel
    
    def moon_position(self, timestamp: float) -> np.ndarray:
        """Simplified Moon position."""
        # TODO: Implement proper lunar ephemeris
        # Very simplified circular orbit for now
        days = timestamp / 86400.0
        moon_period = 27.3  # days
        angle = 2 * math.pi * days / moon_period
        
        moon_distance = 3.844e8  # m (average)
        return np.array([
            moon_distance * math.cos(angle),
            moon_distance * math.sin(angle),
            0.0
        ])
    
    def sun_accel(self, r_sat: np.ndarray, timestamp: float) -> np.ndarray:
        """Solar third-body perturbation."""
        # Reuse SRP sun position function
        srp = SolarRadiationPressure()
        r_sun = srp.sun_position_ecliptic(timestamp)
        return self.third_body_accel(r_sat, r_sun, self.mu_sun)
    
    def moon_accel(self, r_sat: np.ndarray, timestamp: float) -> np.ndarray:
        """Lunar third-body perturbation."""
        r_moon = self.moon_position(timestamp)
        return self.third_body_accel(r_sat, r_moon, self.mu_moon)