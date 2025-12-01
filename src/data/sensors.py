"""
Sensor simulation for optical, radar, and imaging systems.

This module provides:
- Optical angle measurements (RA/Dec) with realistic noise
- Radar range/Doppler measurements with beam patterns
- Image generation with PSF and motion blur
- Observation scheduling and visibility constraints
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Constants
R_EARTH = 6.378137e6  # Earth radius (m)
EARTH_ROT_RATE = 7.2921159e-5  # Earth rotation rate (rad/s)
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

@dataclass
class OpticalObservation:
    """Single optical observation (angles-only)."""
    timestamp: float
    ra: float  # Right ascension (radians)
    dec: float  # Declination (radians) 
    magnitude: Optional[float] = None
    noise_sigma: float = 1e-5  # radians

@dataclass
class RadarObservation:
    """Single radar observation (range/Doppler)."""
    timestamp: float
    range_km: float
    range_rate_kms: float
    azimuth: float
    elevation: float
    snr_db: float

def eci_to_radec(r_eci: np.ndarray) -> Tuple[float, float]:
    """
    Convert ECI position to Right Ascension and Declination.
    
    Args:
        r_eci: Position vector in ECI frame (m)
    
    Returns:
        Tuple of (right_ascension, declination) in radians
    """
    x, y, z = r_eci
    
    # Range
    range_m = np.linalg.norm(r_eci)
    
    # Declination (angle from equatorial plane)
    declination = np.arcsin(z / range_m)
    
    # Right ascension (angle in equatorial plane from vernal equinox)
    right_ascension = np.arctan2(y, x)
    
    # Ensure RA is in [0, 2π]
    if right_ascension < 0:
        right_ascension += 2 * np.pi
    
    return right_ascension, declination

def eci_to_ecef(r_eci: np.ndarray, time: float) -> np.ndarray:
    """
    Convert Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF).
    
    Args:
        r_eci: Position vector in ECI frame (m)
        time: Time since epoch (seconds)
    
    Returns:
        Position vector in ECEF frame (m)
    """
    theta = EARTH_ROT_RATE * time  # Earth rotation angle
    
    # Rotation matrix from ECI to ECEF
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    R = np.array([
        [cos_theta,  sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0,          0,         1]
    ])
    
    return R @ r_eci

def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert geodetic coordinates to ECEF.
    
    Args:
        lat: Latitude (radians)
        lon: Longitude (radians)
        alt: Altitude above sea level (meters)
    
    Returns:
        Position vector in ECEF frame (m)
    """
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis (m)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2  # First eccentricity squared
    
    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    
    return np.array([x, y, z])

def ecef_to_topocentric(r_ecef: np.ndarray, 
                       station_ecef: np.ndarray,
                       station_lat: float, 
                       station_lon: float) -> np.ndarray:
    """
    Convert ECEF position to topocentric (East-North-Up) coordinates.
    
    Args:
        r_ecef: Target position in ECEF (m)
        station_ecef: Station position in ECEF (m)
        station_lat: Station latitude (radians)
        station_lon: Station longitude (radians)
    
    Returns:
        Position in topocentric frame (East, North, Up) (m)
    """
    # Vector from station to target
    delta_r = r_ecef - station_ecef
    
    # Rotation matrix from ECEF to topocentric
    sin_lat = np.sin(station_lat)
    cos_lat = np.cos(station_lat)
    sin_lon = np.sin(station_lon)
    cos_lon = np.cos(station_lon)
    
    R = np.array([
        [-sin_lon,           cos_lon,          0],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat],
        [cos_lat*cos_lon,   cos_lat*sin_lon,   sin_lat]
    ])
    
    return R @ delta_r

def topocentric_to_azel(r_enu: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert topocentric coordinates to azimuth, elevation, and range.
    
    Args:
        r_enu: Position in East-North-Up frame (m)
    
    Returns:
        Tuple of (azimuth, elevation, range) in (radians, radians, meters)
    """
    east, north, up = r_enu
    
    # Range (distance)
    range_m = np.linalg.norm(r_enu)
    
    # Elevation angle
    elevation = np.arcsin(up / range_m)
    
    # Azimuth angle (measured from North towards East)
    azimuth = np.arctan2(east, north)
    
    # Ensure azimuth is in [0, 2π]
    if azimuth < 0:
        azimuth += 2 * np.pi
    
    return azimuth, elevation, range_m

class OpticalSensor:
    """Ground-based optical telescope simulator."""
    
    def __init__(self, lat: float, lon: float, alt: float = 0.0,
                 noise_sigma: float = 1e-5, min_elevation: float = 10.0):
        """
        Initialize optical sensor.
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)  
            alt: Altitude above sea level (meters)
            noise_sigma: Measurement noise standard deviation (radians)
            min_elevation: Minimum elevation angle for observations (degrees)
        """
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.alt = alt
        self.noise_sigma = noise_sigma
        self.min_elevation = np.radians(min_elevation)
        
        # Precompute station ECEF position
        self.station_ecef = geodetic_to_ecef(self.lat, self.lon, self.alt)
        
        # Random number generator for consistent noise
        self.rng = np.random.default_rng()
    
    def set_random_state(self, seed: int) -> None:
        """Set random seed for reproducible measurements."""
        self.rng = np.random.default_rng(seed)
        
    def simulate_observation(self, r_eci: np.ndarray, t: float, 
                           add_noise: bool = True) -> OpticalObservation:
        """
        Convert ECI position to topocentric RA/Dec observation.
        
        Args:
            r_eci: Position in ECI frame (m)
            t: Time since epoch (seconds)
            add_noise: Whether to add measurement noise
        
        Returns:
            OpticalObservation with RA/Dec measurements
        """
        # Convert to RA/Dec directly from ECI
        ra, dec = eci_to_radec(r_eci)
        
        # Add noise if requested
        if add_noise:
            ra_noise = self.rng.normal(0, self.noise_sigma)
            dec_noise = self.rng.normal(0, self.noise_sigma)
            ra += ra_noise
            dec += dec_noise
        
        # Wrap RA to [0, 2π]
        ra = ra % (2 * np.pi)
        
        # Estimate magnitude based on range (very simplified)
        range_km = np.linalg.norm(r_eci) / 1000.0
        magnitude = 5.0 + 5.0 * np.log10(range_km / 1000.0)  # Rough approximation
        
        return OpticalObservation(
            timestamp=t,
            ra=ra,
            dec=dec,
            magnitude=magnitude,
            noise_sigma=self.noise_sigma
        )
        
    def check_visibility(self, r_eci: np.ndarray, t: float) -> bool:
        """
        Check if object is visible (above horizon, nighttime, etc.).
        
        Args:
            r_eci: Position in ECI frame (m)
            t: Time since epoch (seconds)
            
        Returns:
            True if object is visible, False otherwise
        """
        # Convert ECI to ECEF
        r_ecef = eci_to_ecef(r_eci, t)
        
        # Convert to topocentric coordinates
        r_enu = ecef_to_topocentric(r_ecef, self.station_ecef, self.lat, self.lon)
        
        # Get elevation angle
        _, elevation, _ = topocentric_to_azel(r_enu)
        
        # Check if above minimum elevation
        return elevation >= self.min_elevation

class RadarSensor:
    """Ground-based radar system simulator."""
    
    def __init__(self, lat: float, lon: float, alt: float = 0.0,
                 range_noise_km: float = 0.01, angle_noise_deg: float = 0.1,
                 min_elevation: float = 5.0, max_range_km: float = 3000.0):
        """
        Initialize radar sensor.
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude above sea level (meters)
            range_noise_km: Range measurement noise (km)
            angle_noise_deg: Angular measurement noise (degrees)
            min_elevation: Minimum elevation angle (degrees)
            max_range_km: Maximum detection range (km)
        """
        self.lat = np.radians(lat)
        self.lon = np.radians(lon)
        self.alt = alt
        self.range_noise_km = range_noise_km
        self.angle_noise_rad = np.radians(angle_noise_deg)
        self.min_elevation = np.radians(min_elevation)
        self.max_range_km = max_range_km
        
        # Precompute station ECEF position
        self.station_ecef = geodetic_to_ecef(self.lat, self.lon, self.alt)
        
        # Random number generator
        self.rng = np.random.default_rng()
    
    def set_random_state(self, seed: int) -> None:
        """Set random seed for reproducible measurements."""
        self.rng = np.random.default_rng(seed)
        
    def simulate_observation(self, r_eci: np.ndarray, v_eci: np.ndarray, 
                           t: float, add_noise: bool = True) -> RadarObservation:
        """
        Generate radar range/Doppler observation.
        
        Args:
            r_eci: Position in ECI frame (m)
            v_eci: Velocity in ECI frame (m/s)
            t: Time since epoch (seconds)
            add_noise: Whether to add measurement noise
            
        Returns:
            RadarObservation with range/Doppler measurements
        """
        # Convert ECI to ECEF (position and velocity)
        r_ecef = eci_to_ecef(r_eci, t)
        
        # Convert velocity (need to account for Earth rotation)
        theta = EARTH_ROT_RATE * t
        omega_earth = np.array([0, 0, EARTH_ROT_RATE])
        v_ecef = eci_to_ecef(v_eci, t) - np.cross(omega_earth, r_ecef)
        
        # Convert to topocentric coordinates
        r_enu = ecef_to_topocentric(r_ecef, self.station_ecef, self.lat, self.lon)
        
        # Convert velocity to topocentric frame
        sin_lat = np.sin(self.lat)
        cos_lat = np.cos(self.lat)
        sin_lon = np.sin(self.lon)
        cos_lon = np.cos(self.lon)
        
        R_ecef_to_enu = np.array([
            [-sin_lon,           cos_lon,          0],
            [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat],
            [cos_lat*cos_lon,   cos_lat*sin_lon,   sin_lat]
        ])
        
        v_enu = R_ecef_to_enu @ (v_ecef - np.cross(omega_earth, self.station_ecef))
        
        # Get range, azimuth, elevation
        azimuth, elevation, range_m = topocentric_to_azel(r_enu)
        
        # Calculate range rate (radial velocity)
        range_unit_vector = r_enu / np.linalg.norm(r_enu)
        range_rate_ms = np.dot(v_enu, range_unit_vector)
        
        # Convert to km and km/s
        range_km = range_m / 1000.0
        range_rate_kms = range_rate_ms / 1000.0
        
        # Add noise if requested
        if add_noise:
            range_km += self.rng.normal(0, self.range_noise_km)
            range_rate_kms += self.rng.normal(0, self.range_noise_km * 0.1)  # Doppler noise
            azimuth += self.rng.normal(0, self.angle_noise_rad)
            elevation += self.rng.normal(0, self.angle_noise_rad)
        
        # Estimate SNR based on range (simplified)
        snr_db = 30.0 - 20.0 * np.log10(range_km / 100.0)  # Decreases with range
        
        return RadarObservation(
            timestamp=t,
            range_km=range_km,
            range_rate_kms=range_rate_kms,
            azimuth=azimuth,
            elevation=elevation,
            snr_db=snr_db
        )
    
    def check_visibility(self, r_eci: np.ndarray, t: float) -> bool:
        """
        Check if object is detectable by radar.
        
        Args:
            r_eci: Position in ECI frame (m)
            t: Time since epoch (seconds)
            
        Returns:
            True if object is detectable, False otherwise
        """
        # Convert ECI to ECEF
        r_ecef = eci_to_ecef(r_eci, t)
        
        # Convert to topocentric coordinates
        r_enu = ecef_to_topocentric(r_ecef, self.station_ecef, self.lat, self.lon)
        
        # Get elevation and range
        _, elevation, range_m = topocentric_to_azel(r_enu)
        range_km = range_m / 1000.0
        
        # Check constraints
        if elevation < self.min_elevation:
            return False
        if range_km > self.max_range_km:
            return False
            
        return True

def render_optical_image(observations: List[OpticalObservation], 
                        exposure_time: float = 30.0,
                        image_size: Tuple[int, int] = (1024, 1024),
                        fov_deg: float = 1.0) -> np.ndarray:
    """
    Render synthetic optical image with star streaks.
    
    Args:
        observations: List of optical observations during exposure
        exposure_time: Exposure duration (seconds)
        image_size: Output image dimensions (pixels)
        fov_deg: Field of view (degrees)
        
    Returns:
        Synthetic image array with streaked objects
    """
    if not observations:
        return np.zeros(image_size)
    
    # Create image array
    image = np.zeros(image_size)
    height, width = image_size
    
    # Convert field of view to radians
    fov_rad = np.radians(fov_deg)
    
    # Image center in RA/Dec (use first observation as reference)
    center_ra = observations[0].ra
    center_dec = observations[0].dec
    
    # Pixel scale (radians per pixel)
    pixel_scale = fov_rad / min(width, height)
    
    for obs in observations:
        # Convert RA/Dec to pixel coordinates relative to center
        delta_ra = obs.ra - center_ra
        delta_dec = obs.dec - center_dec
        
        # Handle RA wrap-around
        if delta_ra > np.pi:
            delta_ra -= 2 * np.pi
        elif delta_ra < -np.pi:
            delta_ra += 2 * np.pi
        
        # Convert to pixel coordinates
        # Note: RA increases to the East (left in images)
        x_pixel = width // 2 - delta_ra / pixel_scale
        y_pixel = height // 2 + delta_dec / pixel_scale
        
        # Check if within image bounds
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            # Add point source (could be made more realistic with PSF)
            x_int, y_int = int(x_pixel), int(y_pixel)
            
            # Simple magnitude-to-intensity conversion
            if obs.magnitude is not None:
                intensity = 10**(-0.4 * (obs.magnitude - 10.0))  # Relative to mag 10
            else:
                intensity = 1.0
            
            # Add Gaussian PSF (simplified)
            psf_sigma = 1.5  # pixels
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    px, py = x_int + dx, y_int + dy
                    if 0 <= px < width and 0 <= py < height:
                        # Gaussian PSF
                        psf_val = np.exp(-(dx**2 + dy**2) / (2 * psf_sigma**2))
                        image[py, px] += intensity * psf_val
    
    # Normalize image
    if image.max() > 0:
        image = image / image.max()
    
    return image