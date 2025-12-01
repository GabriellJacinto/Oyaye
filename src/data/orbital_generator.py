"""
Orbital data generator for NP-SNN training.

This module provides:
- OrbitalDataGenerator class compatible with training pipeline
- Integration with existing scenario generation and sensor simulation
- Batch generation for efficient dataset creation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path

from .generators import ScenarioGenerator, oe_to_cartesian
from .sensors import OpticalSensor, RadarSensor
from .orbital_dataset import OrbitalDataset, create_orbital_dataloader

class OrbitalDataGenerator:
    """
    Generate orbital trajectory data for NP-SNN training.
    
    Integrates scenario generation and sensor simulation to create
    realistic training datasets with observation gaps and noise.
    """
    
    def __init__(self, 
                 n_objects: int = 1000,
                 time_horizon_hours: float = 24.0,
                 dt_minutes: float = 5.0,
                 observation_noise_std: float = 0.001,
                 device: torch.device = None):
        """
        Initialize orbital data generator.
        
        Args:
            n_objects: Number of objects to track
            time_horizon_hours: Trajectory length in hours
            dt_minutes: Time step in minutes
            observation_noise_std: Standard deviation for observation noise
            device: PyTorch device for tensor operations
        """
        self.n_objects = n_objects
        self.time_horizon_hours = time_horizon_hours
        self.dt_minutes = dt_minutes
        self.observation_noise_std = observation_noise_std
        self.device = device or torch.device('cpu')
        
        # Time parameters
        self.dt_seconds = dt_minutes * 60.0
        self.duration_seconds = time_horizon_hours * 3600.0
        self.n_timesteps = int(self.duration_seconds / self.dt_seconds) + 1
        
        # Initialize scenario generator
        config = {
            "propagator": {
                "include_j2": True,
                "include_drag": False,  # Simplified for now
                "include_srp": False
            },
            "object_mix": {
                "debris": 0.7,
                "satellite": 0.2, 
                "rocket_body": 0.1
            }
        }
        
        self.scenario_generator = ScenarioGenerator(config, random_state=42)
        
        # Initialize sensors for observation simulation
        self.optical_sensors = [
            OpticalSensor(lat=40.0, lon=-74.0, noise_sigma=observation_noise_std),  # East Coast
            OpticalSensor(lat=34.0, lon=-118.0, noise_sigma=observation_noise_std), # West Coast
            OpticalSensor(lat=52.0, lon=5.0, noise_sigma=observation_noise_std),    # Europe
        ]
        
        self.radar_sensors = [
            RadarSensor(lat=35.0, lon=-106.0, range_noise_km=0.01),  # Southwest US
            RadarSensor(lat=64.0, lon=-21.0, range_noise_km=0.01),   # Iceland
        ]
    
    def generate_single_trajectory(self, 
                                  add_observation_gaps: bool = True,
                                  gap_probability: float = 0.2) -> Dict[str, torch.Tensor]:
        """
        Generate a single object trajectory with observations.
        
        Args:
            add_observation_gaps: Whether to simulate observation gaps
            gap_probability: Probability of missing an observation
            
        Returns:
            Dictionary containing trajectory data
        """
        # Generate scenario with single object
        scenario = self.scenario_generator.generate_scenario(
            n_objects=1, 
            duration_hours=self.time_horizon_hours
        )
        
        # Propagate trajectory
        scenario_data = self.scenario_generator.propagate_scenario(
            scenario, 
            time_step=self.dt_seconds
        )
        
        if not scenario_data['trajectories']:
            # Fallback to simple circular orbit if propagation fails
            return self._generate_fallback_trajectory(add_observation_gaps, gap_probability)
        
        # Extract trajectory data
        obj_name = list(scenario_data['trajectories'].keys())[0]
        traj_data = scenario_data['trajectories'][obj_name]
        
        times = torch.tensor(traj_data['times'], dtype=torch.float32, device=self.device)
        positions = torch.tensor(traj_data['positions'], dtype=torch.float32, device=self.device)
        velocities = torch.tensor(traj_data['velocities'], dtype=torch.float32, device=self.device)
        
        # Create state vector (position + velocity)
        states = torch.cat([positions, velocities], dim=-1)  # (T, 6)
        
        # Simulate observations with sensors
        observations, observation_mask = self._simulate_observations(
            positions, velocities, times, add_observation_gaps, gap_probability
        )
        
        return {
            'times': times,           # (T,)
            'states': states,         # (T, 6) - [x, y, z, vx, vy, vz]
            'observations': observations,  # (T, 6) - observed states (may have gaps)
            'observation_mask': observation_mask,  # (T,) - boolean mask
            'object_properties': traj_data['object_properties']
        }
    
    def _generate_fallback_trajectory(self, 
                                    add_observation_gaps: bool,
                                    gap_probability: float) -> Dict[str, torch.Tensor]:
        """Generate simple circular orbit as fallback."""
        # Simple circular LEO orbit
        a = 6.378137e6 + 500e3  # 500 km altitude
        mu = 3.986004418e14     # Earth's GM
        n = np.sqrt(mu / a**3)  # Mean motion
        
        times = np.linspace(0, self.duration_seconds, self.n_timesteps)
        
        # Circular orbit in x-y plane
        positions = np.zeros((len(times), 3))
        velocities = np.zeros((len(times), 3))
        
        for i, t in enumerate(times):
            theta = n * t
            positions[i] = [a * np.cos(theta), a * np.sin(theta), 0]
            velocities[i] = [-a * n * np.sin(theta), a * n * np.cos(theta), 0]
        
        # Convert to tensors
        times_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
        velocities_tensor = torch.tensor(velocities, dtype=torch.float32, device=self.device)
        states = torch.cat([positions_tensor, velocities_tensor], dim=-1)
        
        # Simulate observations
        observations, observation_mask = self._simulate_observations(
            positions_tensor, velocities_tensor, times_tensor, 
            add_observation_gaps, gap_probability
        )
        
        return {
            'times': times_tensor,
            'states': states,
            'observations': observations,
            'observation_mask': observation_mask,
            'object_properties': {'mass': 100.0, 'area': 1.0, 'object_type': 'fallback'}
        }
    
    def _simulate_observations(self, 
                             positions: torch.Tensor,
                             velocities: torch.Tensor, 
                             times: torch.Tensor,
                             add_gaps: bool,
                             gap_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate sensor observations with gaps and noise.
        
        Args:
            positions: Position vectors (T, 3)
            velocities: Velocity vectors (T, 3)
            times: Time stamps (T,)
            add_gaps: Whether to add observation gaps
            gap_probability: Probability of missing observation
            
        Returns:
            Tuple of (observations, observation_mask)
        """
        n_times = len(times)
        observations = torch.zeros((n_times, 6), dtype=torch.float32, device=self.device)
        observation_mask = torch.ones(n_times, dtype=torch.bool, device=self.device)
        
        # Copy states as baseline observations
        observations = torch.cat([positions, velocities], dim=-1)
        
        # Add observation noise
        if self.observation_noise_std > 0:
            noise = torch.normal(0, self.observation_noise_std, 
                               observations.shape, device=self.device)
            observations = observations + noise
        
        # Add observation gaps
        if add_gaps and gap_probability > 0:
            # Random gaps
            gap_mask = torch.rand(n_times, device=self.device) < gap_probability
            
            # Ensure we have some observations (at least 20%)
            n_gaps = gap_mask.sum().item()
            max_gaps = int(0.8 * n_times)
            if n_gaps > max_gaps:
                # Randomly select which gaps to keep
                gap_indices = torch.where(gap_mask)[0]
                keep_indices = gap_indices[torch.randperm(len(gap_indices))[:max_gaps]]
                gap_mask.fill_(False)
                gap_mask[keep_indices] = True
            
            observation_mask = ~gap_mask
            
            # Zero out observations where we have gaps
            observations[gap_mask] = 0.0
        
        return observations, observation_mask
    
    def generate_batch(self, 
                      batch_size: int,
                      add_observation_gaps: bool = True,
                      gap_probability: float = 0.2) -> List[Dict[str, torch.Tensor]]:
        """
        Generate a batch of trajectories.
        
        Args:
            batch_size: Number of trajectories to generate
            add_observation_gaps: Whether to simulate observation gaps
            gap_probability: Probability of missing observations
            
        Returns:
            List of trajectory dictionaries
        """
        trajectories = []
        
        for i in range(batch_size):
            try:
                traj = self.generate_single_trajectory(add_observation_gaps, gap_probability)
                trajectories.append(traj)
            except Exception as e:
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                # Generate fallback trajectory
                traj = self._generate_fallback_trajectory(add_observation_gaps, gap_probability)
                trajectories.append(traj)
        
        return trajectories
    
    def set_time_horizon(self, hours: float):
        """Update time horizon and recompute time parameters."""
        self.time_horizon_hours = hours
        self.duration_seconds = hours * 3600.0
        self.n_timesteps = int(self.duration_seconds / self.dt_seconds) + 1
    
    def set_observation_noise(self, noise_std: float):
        """Update observation noise level."""
        self.observation_noise_std = noise_std
        # Update sensor noise levels
        for sensor in self.optical_sensors:
            sensor.noise_sigma = noise_std
        for sensor in self.radar_sensors:
            sensor.range_noise_km = noise_std * 100  # Scale for radar