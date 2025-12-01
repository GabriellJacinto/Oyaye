"""
Data generation and management for the NP-SNN space debris tracking project.

This module provides:
- Orbital scenario generation with realistic parameters
- Sensor simulation (optical telescopes, radar systems)
- Data persistence and management for neural network training
- Coordinate transformations and utilities
"""

from .generators import (
    ScenarioGenerator,
    OrbitalElementSampler, 
    ObjectParameterSampler,
    OrbitalElements,
    SpaceObject,
    Scenario,
    oe_to_cartesian
)

from .sensors import (
    OpticalSensor, 
    RadarSensor, 
    OpticalObservation, 
    RadarObservation,
    eci_to_radec, 
    render_optical_image
)

from .sensor_data import (
    SensorDataManager, 
    TrackingPass, 
    create_sensor_dataset
)

from .orbital_generator import OrbitalDataGenerator
from .orbital_dataset import OrbitalDataset, create_orbital_dataloader

__all__ = [
    # Orbital generation
    'ScenarioGenerator',
    'OrbitalElementSampler',
    'ObjectParameterSampler', 
    'OrbitalElements',
    'SpaceObject',
    'Scenario',
    'oe_to_cartesian',
    
    # Sensor simulation
    'OpticalSensor',
    'RadarSensor',
    'OpticalObservation',
    'RadarObservation', 
    'eci_to_radec',
    'render_optical_image',
    
    # Data management
    'SensorDataManager',
    'TrackingPass',
    'create_sensor_dataset',
    
    # Training data generation
    'OrbitalDataGenerator',
    'OrbitalDataset', 
    'create_orbital_dataloader'
]