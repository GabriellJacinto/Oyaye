"""
Data generation and scenario sampling for space debris simulation.

This module provides:
- Scenario generator with realistic orbital parameter sampling
- Catalog-based priors for object characteristics
- Metadata and provenance tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import hashlib
import warnings

# Import our propagators
try:
    from ..physics.propagators import NumericalPropagator, TwoBodyPropagator, OrbitState
except ImportError:
    from physics.propagators import NumericalPropagator, TwoBodyPropagator, OrbitState

# Physical constants
GM_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6.378137e6  # m

@dataclass
class OrbitalElements:
    """Classical orbital elements."""
    a: float          # Semi-major axis (m)
    e: float          # Eccentricity
    i: float          # Inclination (rad) 
    raan: float       # Right ascension of ascending node (rad)
    w: float          # Argument of periapsis (rad)
    nu: float         # True anomaly (rad)
    epoch: float      # Epoch time (s)

@dataclass
class SpaceObject:
    """Space object with orbital and physical properties."""
    name: str
    oe: OrbitalElements
    mass: float           # kg
    area: float           # m^2 
    drag_coeff: float     # Cd
    reflectivity: float   # SRP coefficient
    object_type: str      # "debris", "satellite", "rocket_body"

@dataclass
class Scenario:
    """Complete space scenario with multiple objects."""
    objects: List[SpaceObject]
    time_span: Tuple[float, float]  # Start and end time
    metadata: Dict
    
class OrbitalElementSampler:
    """Sample realistic orbital elements from distributions."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)
        
        # Define realistic distributions for different orbital regimes
        self.distributions = {
            "leo": {
                "altitude_range": (200e3, 2000e3),  # 200-2000 km
                "inclination_dist": "uniform",       # Uniform 0-180 deg
                "eccentricity_mean": 0.05,
                "eccentricity_std": 0.02
            },
            "meo": {
                "altitude_range": (2000e3, 35786e3),
                "inclination_dist": "clustered",     # Clustered around 55 deg
                "eccentricity_mean": 0.1,
                "eccentricity_std": 0.05
            },
            "geo": {
                "altitude_range": (35786e3, 35786e3),  # Fixed GEO altitude
                "inclination_dist": "equatorial",      # Near 0 deg
                "eccentricity_mean": 0.001,
                "eccentricity_std": 0.001
            }
        }
    
    def sample_leo_debris(self) -> OrbitalElements:
        """Sample LEO debris orbital elements."""
        dist = self.distributions["leo"]
        
        # Semi-major axis from altitude range
        alt_min, alt_max = dist["altitude_range"]
        altitude = self.rng.uniform(alt_min, alt_max)
        a = R_EARTH + altitude
        
        # Eccentricity (low for LEO debris)
        e = max(0, self.rng.normal(dist["eccentricity_mean"], dist["eccentricity_std"]))
        e = min(e, 0.99)  # Cap at 0.99
        
        # Inclination (uniform distribution for LEO debris)
        i = self.rng.uniform(0, np.pi)
        
        # RAAN and argument of periapsis (uniform)
        raan = self.rng.uniform(0, 2*np.pi)
        w = self.rng.uniform(0, 2*np.pi)
        
        # True anomaly (uniform)
        nu = self.rng.uniform(0, 2*np.pi)
        
        return OrbitalElements(a=a, e=e, i=i, raan=raan, w=w, nu=nu, epoch=0.0)
    
    def sample_operational_satellite(self) -> OrbitalElements:
        """Sample operational satellite orbital elements."""
        # Mix of LEO, MEO, and GEO
        regime = self.rng.choice(["leo", "meo", "geo"], p=[0.6, 0.3, 0.1])
        
        if regime == "leo":
            return self._sample_leo_operational()
        elif regime == "meo":
            return self._sample_meo_operational()
        else:
            return self._sample_geo_operational()
    
    def _sample_leo_operational(self) -> OrbitalElements:
        """Sample LEO operational satellite (ISS, Starlink, etc.)."""
        # Common LEO operational altitudes
        common_alts = [400e3, 550e3, 1200e3]  # ISS, Starlink, etc.
        altitude = self.rng.choice(common_alts) + self.rng.normal(0, 50e3)
        altitude = max(300e3, min(2000e3, altitude))
        
        a = R_EARTH + altitude
        e = max(0, self.rng.normal(0.01, 0.005))  # Very circular
        
        # Common inclinations for operational sats
        common_incs = [0, 28.5, 51.6, 97.8]  # Equatorial, Cape, ISS, SSO
        i_deg = self.rng.choice(common_incs) + self.rng.normal(0, 2)
        i = np.radians(i_deg)
        
        raan = self.rng.uniform(0, 2*np.pi)
        w = self.rng.uniform(0, 2*np.pi) 
        nu = self.rng.uniform(0, 2*np.pi)
        
        return OrbitalElements(a=a, e=e, i=i, raan=raan, w=w, nu=nu, epoch=0.0)
    
    def _sample_meo_operational(self) -> OrbitalElements:
        """Sample MEO operational (GPS, Galileo, etc.)."""
        # GPS: ~20,200 km, 55 deg
        # Galileo: ~23,222 km, 56 deg
        altitudes = [20200e3, 23222e3]
        altitude = self.rng.choice(altitudes) + self.rng.normal(0, 100e3)
        
        a = R_EARTH + altitude
        e = max(0, self.rng.normal(0.01, 0.005))
        
        # MEO constellation inclinations
        i = np.radians(self.rng.normal(55, 2))
        
        raan = self.rng.uniform(0, 2*np.pi)
        w = self.rng.uniform(0, 2*np.pi)
        nu = self.rng.uniform(0, 2*np.pi)
        
        return OrbitalElements(a=a, e=e, i=i, raan=raan, w=w, nu=nu, epoch=0.0)
    
    def _sample_geo_operational(self) -> OrbitalElements:
        """Sample GEO operational satellite."""
        a = R_EARTH + 35786e3  # GEO altitude
        e = max(0, self.rng.normal(0.001, 0.0005))  # Very circular
        i = np.radians(self.rng.normal(0, 1))  # Near equatorial
        
        raan = self.rng.uniform(0, 2*np.pi)
        w = self.rng.uniform(0, 2*np.pi)
        nu = self.rng.uniform(0, 2*np.pi)
        
        return OrbitalElements(a=a, e=e, i=i, raan=raan, w=w, nu=nu, epoch=0.0)

def oe_to_cartesian(oe: OrbitalElements, mu: float = GM_EARTH) -> Tuple[np.ndarray, np.ndarray]:
    """Convert orbital elements to Cartesian state vectors."""
    a, e, i, raan, w, nu = oe.a, oe.e, oe.i, oe.raan, oe.w, oe.nu
    
    # Orbital parameter
    p = a * (1 - e**2)
    
    # Position and velocity in orbital plane
    r_mag = p / (1 + e * np.cos(nu))
    
    r_pqw = np.array([
        r_mag * np.cos(nu),
        r_mag * np.sin(nu),
        0.0
    ])
    
    v_pqw = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0.0
    ])
    
    # Rotation matrices
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(w), np.sin(w)
    
    # Rotation from perifocal to inertial frame
    R11 = cos_raan*cos_w - sin_raan*sin_w*cos_i
    R12 = -cos_raan*sin_w - sin_raan*cos_w*cos_i
    R13 = sin_raan*sin_i
    
    R21 = sin_raan*cos_w + cos_raan*sin_w*cos_i
    R22 = -sin_raan*sin_w + cos_raan*cos_w*cos_i
    R23 = -cos_raan*sin_i
    
    R31 = sin_w*sin_i
    R32 = cos_w*sin_i
    R33 = cos_i
    
    R = np.array([[R11, R12, R13],
                  [R21, R22, R23],
                  [R31, R32, R33]])
    
    # Transform to inertial frame
    r_eci = R @ r_pqw
    v_eci = R @ v_pqw
    
    return r_eci, v_eci

class ObjectParameterSampler:
    """Sample realistic physical properties for space objects."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)
        
        # Object type distributions based on ESA space debris model
        self.object_types = {
            "debris": {
                "mass_range": (0.01, 1000),  # 10g to 1 ton
                "area_to_mass": (0.1, 10.0),  # m^2/kg
                "drag_coeff": (2.0, 2.5),
                "reflectivity": (1.0, 1.5)
            },
            "satellite": {
                "mass_range": (1, 10000),  # 1 kg to 10 tons
                "area_to_mass": (0.01, 0.1),  # More compact
                "drag_coeff": (2.0, 2.5),
                "reflectivity": (1.2, 2.0)  # Solar panels
            },
            "rocket_body": {
                "mass_range": (100, 50000),  # 100 kg to 50 tons
                "area_to_mass": (0.05, 0.5),
                "drag_coeff": (2.0, 2.5),
                "reflectivity": (1.0, 1.8)
            }
        }
    
    def sample_object_properties(self, object_type: str) -> Dict:
        """Sample physical properties for given object type."""
        if object_type not in self.object_types:
            raise ValueError(f"Unknown object type: {object_type}")
        
        props = self.object_types[object_type]
        
        # Sample mass
        mass_min, mass_max = props["mass_range"]
        mass = self.rng.lognormal(
            np.log(np.sqrt(mass_min * mass_max)), 
            0.5
        )
        mass = np.clip(mass, mass_min, mass_max)
        
        # Sample area-to-mass ratio, then compute area
        am_min, am_max = props["area_to_mass"]
        area_to_mass = self.rng.uniform(am_min, am_max)
        area = mass * area_to_mass
        
        # Sample drag coefficient
        cd_min, cd_max = props["drag_coeff"]
        drag_coeff = self.rng.uniform(cd_min, cd_max)
        
        # Sample reflectivity
        refl_min, refl_max = props["reflectivity"]
        reflectivity = self.rng.uniform(refl_min, refl_max)
        
        return {
            "mass": mass,
            "area": area,
            "drag_coeff": drag_coeff,
            "reflectivity": reflectivity
        }

class ScenarioGenerator:
    """Generate realistic space debris scenarios for training."""
    
    def __init__(self, config: Dict, random_state: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(random_state)
        
        # Initialize samplers
        self.oe_sampler = OrbitalElementSampler(random_state)
        self.obj_sampler = ObjectParameterSampler(random_state)
        
        # Initialize propagator
        propagator_config = config.get("propagator", {})
        self.propagator = NumericalPropagator(
            include_j2=propagator_config.get("include_j2", True),
            include_drag=propagator_config.get("include_drag", False),
            include_srp=propagator_config.get("include_srp", False)
        )
    
    def sample_from_catalog(self, regime: str = "leo", object_type: str = "debris") -> SpaceObject:
        """Sample orbital parameters from realistic distributions."""
        
        # Sample orbital elements
        if object_type == "debris":
            oe = self.oe_sampler.sample_leo_debris()
        else:
            oe = self.oe_sampler.sample_operational_satellite()
        
        # Sample physical properties
        props = self.obj_sampler.sample_object_properties(object_type)
        
        # Generate unique name
        obj_id = self.rng.randint(10000, 99999)
        name = f"{object_type}_{obj_id}"
        
        return SpaceObject(
            name=name,
            oe=oe,
            mass=props["mass"],
            area=props["area"],
            drag_coeff=props["drag_coeff"],
            reflectivity=props["reflectivity"],
            object_type=object_type
        )
    
    def generate_scenario(self, n_objects: int = 1, duration_hours: float = 2.0) -> Scenario:
        """Generate complete scenario with multiple objects."""
        
        objects = []
        
        # Sample object mix based on config
        object_mix = self.config.get("object_mix", {
            "debris": 0.7,
            "satellite": 0.2,
            "rocket_body": 0.1
        })
        
        for i in range(n_objects):
            # Choose object type based on mix
            obj_type = self.rng.choice(
                list(object_mix.keys()),
                p=list(object_mix.values())
            )
            
            # Sample object
            obj = self.sample_from_catalog(object_type=obj_type)
            objects.append(obj)
        
        # Time span
        duration_seconds = duration_hours * 3600
        time_span = (0.0, duration_seconds)
        
        # Metadata
        metadata = {
            "generation_time": time.time(),
            "n_objects": n_objects,
            "duration_hours": duration_hours,
            "config": self.config.copy(),
            "random_seed": self.rng.get_state()[1][0] if hasattr(self.rng, 'get_state') else None
        }
        
        return Scenario(
            objects=objects,
            time_span=time_span,
            metadata=metadata
        )
    
    def propagate_scenario(self, scenario: Scenario, time_step: float = 60.0) -> Dict:
        """Propagate all objects in scenario and return trajectories."""
        
        start_time, end_time = scenario.time_span
        t_span = np.arange(start_time, end_time + time_step, time_step)
        
        trajectories = {}
        
        for obj in scenario.objects:
            # Convert orbital elements to Cartesian
            r0, v0 = oe_to_cartesian(obj.oe)
            initial_state = OrbitState(r=r0, v=v0, t=start_time)
            
            # Propagation parameters
            kwargs = {
                "ballistic_coeff": obj.drag_coeff * obj.area / obj.mass,
                "area_mass_ratio": obj.area / obj.mass,
                "reflectivity": obj.reflectivity
            }
            
            try:
                # Propagate trajectory
                positions, velocities = self.propagator.propagate(
                    initial_state, t_span, **kwargs
                )
                
                trajectories[obj.name] = {
                    "times": t_span,
                    "positions": positions,
                    "velocities": velocities,
                    "object_properties": asdict(obj)
                }
                
            except Exception as e:
                warnings.warn(f"Propagation failed for {obj.name}: {e}")
                continue
        
        return {
            "scenario": asdict(scenario),
            "trajectories": trajectories,
            "propagation_metadata": {
                "time_step": time_step,
                "n_points": len(t_span),
                "propagator_config": {
                    "include_j2": self.propagator.include_j2,
                    "include_drag": self.propagator.include_drag,
                    "include_srp": self.propagator.include_srp
                }
            }
        }
    
    def save_scenario(self, scenario_data: Dict, filepath: Path) -> None:
        """Save scenario with full metadata and provenance."""
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file metadata
        file_metadata = {
            "file_format_version": "1.0",
            "save_time": time.time(),
            "file_hash": None  # Will be computed after saving
        }
        
        scenario_data["file_metadata"] = file_metadata
        
        # Save as JSON (for now - could use HDF5 for larger datasets)
        with open(filepath, 'w') as f:
            json.dump(scenario_data, f, indent=2, default=_json_serializer)
        
        # Compute and update file hash
        file_hash = _compute_file_hash(filepath)
        scenario_data["file_metadata"]["file_hash"] = file_hash
        
        # Re-save with hash
        with open(filepath, 'w') as f:
            json.dump(scenario_data, f, indent=2, default=_json_serializer)

def sample_orbital_elements(distribution: str = "leo", random_state: Optional[int] = None) -> OrbitalElements:
    """Sample realistic orbital elements for given distribution."""
    sampler = OrbitalElementSampler(random_state)
    
    if distribution == "leo":
        return sampler.sample_leo_debris()
    elif distribution == "operational":
        return sampler.sample_operational_satellite()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def _json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return str(obj)

def _compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file for integrity checking."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()