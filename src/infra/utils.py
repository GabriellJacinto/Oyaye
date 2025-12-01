"""
Utility functions for NP-SNN project.

This module provides:
- Configuration loading and validation  
- Random seed management
- Common coordinate transformations
- File I/O helpers
"""

import os
import json
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import hashlib
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config

def save_config(config: Dict[str, Any], 
                output_path: Union[str, Path],
                format: str = "yaml") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
        format: File format ('yaml' or 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if format.lower() == 'yaml':
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

def set_random_seeds(seed: int = 42, 
                    deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def create_experiment_id(config: Dict[str, Any],
                        timestamp: bool = True) -> str:
    """
    Create unique experiment identifier based on configuration.
    
    Args:
        config: Experiment configuration
        timestamp: Whether to include timestamp
        
    Returns:
        Unique experiment ID string
    """
    # Create hash from config
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Create ID components
    components = []
    
    # Add model type
    model_type = config.get('model', {}).get('type', 'npsnn')
    components.append(model_type)
    
    # Add key parameters
    if 'snn' in config.get('model', {}):
        snn_config = config['model']['snn']
        hidden_sizes = snn_config.get('hidden_sizes', [128])
        components.append(f"h{'-'.join(map(str, hidden_sizes))}")
    
    # Add loss configuration
    loss_config = config.get('loss', {})
    if 'w_dynamics' in loss_config:
        components.append(f"wd{loss_config['w_dynamics']}")
    
    # Add timestamp if requested
    if timestamp:
        timestamp_str = datetime.now().strftime("%m%d_%H%M")
        components.append(timestamp_str)
    
    # Add config hash
    components.append(config_hash)
    
    return "_".join(components)

def eci_to_ecef(r_eci: np.ndarray, 
                t_utc: float,
                omega_earth: float = 7.292115e-5) -> np.ndarray:
    """
    Transform from Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF).
    
    Args:
        r_eci: Position vector in ECI frame (m)
        t_utc: Time in seconds since epoch
        omega_earth: Earth rotation rate (rad/s)
        
    Returns:
        Position vector in ECEF frame (m)
    """
    # Earth rotation angle
    theta = omega_earth * t_utc
    
    # Rotation matrix from ECI to ECEF
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    R = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    return R @ r_eci

def ecef_to_geodetic(r_ecef: np.ndarray,
                    a: float = 6.378137e6,  # WGS84 semi-major axis
                    f: float = 1.0/298.257223563) -> tuple:
    """
    Convert ECEF coordinates to geodetic (lat, lon, alt).
    
    Args:
        r_ecef: ECEF position vector (m)
        a: Semi-major axis (m)
        f: Flattening factor
        
    Returns:
        (latitude_rad, longitude_rad, altitude_m)
    """
    x, y, z = r_ecef
    
    # Longitude
    lon = np.arctan2(y, x)
    
    # Iterative solution for latitude and altitude
    e2 = f * (2 - f)  # First eccentricity squared
    p = np.sqrt(x**2 + y**2)
    
    # Initial guess
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(5):  # Usually converges quickly
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
    
    return lat, lon, alt

def compute_elevation_azimuth(r_sat_ecef: np.ndarray,
                             observer_lat: float,
                             observer_lon: float,
                             observer_alt: float = 0.0) -> tuple:
    """
    Compute elevation and azimuth angles from observer to satellite.
    
    Args:
        r_sat_ecef: Satellite position in ECEF (m)
        observer_lat: Observer latitude (radians)
        observer_lon: Observer longitude (radians)
        observer_alt: Observer altitude (m)
        
    Returns:
        (elevation_rad, azimuth_rad)
    """
    # Observer position in ECEF
    a = 6.378137e6  # WGS84 semi-major axis
    f = 1.0/298.257223563  # Flattening
    e2 = f * (2 - f)
    
    N = a / np.sqrt(1 - e2 * np.sin(observer_lat)**2)
    
    r_obs_ecef = np.array([
        (N + observer_alt) * np.cos(observer_lat) * np.cos(observer_lon),
        (N + observer_alt) * np.cos(observer_lat) * np.sin(observer_lon),
        (N * (1 - e2) + observer_alt) * np.sin(observer_lat)
    ])
    
    # Relative position vector
    r_rel_ecef = r_sat_ecef - r_obs_ecef
    
    # Transform to topocentric (East-North-Up) frame
    sin_lat = np.sin(observer_lat)
    cos_lat = np.cos(observer_lat)
    sin_lon = np.sin(observer_lon)
    cos_lon = np.cos(observer_lon)
    
    R_ecef_to_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    r_enu = R_ecef_to_enu @ r_rel_ecef
    
    # Compute elevation and azimuth
    east, north, up = r_enu
    
    slant_range = np.linalg.norm(r_enu)
    elevation = np.arcsin(up / slant_range)
    azimuth = np.arctan2(east, north)
    
    # Ensure azimuth is in [0, 2π]
    if azimuth < 0:
        azimuth += 2 * np.pi
    
    return elevation, azimuth

def validate_config(config: Dict[str, Any], 
                   required_keys: List[str]) -> None:
    """
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (dot-separated)
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
        except (KeyError, TypeError):
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(filepath: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Hex string of file hash
    """
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def merge_configs(base_config: Dict[str, Any],
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"{self.name} completed in {format_duration(duration)}")
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()