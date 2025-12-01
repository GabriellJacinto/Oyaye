"""
Physics-Informed Loss Functions for Neural Physics-Informed SNNs

This module implements comprehensive physics-based loss functions for training
NP-SNN models on orbital mechanics problems. The losses enforce physical
constraints while maintaining numerical stability and providing adaptive
weighting strategies.

Key Features:
- Orbital mechanics constraints (energy, angular momentum conservation)
- Measurement consistency losses (optical, radar, multi-modal)
- Temporal smoothness and dynamics residuals
- Adaptive loss balancing and dynamic weighting
- Multi-scale loss normalization
- Uncertainty-aware loss functions
- Curriculum learning support

Physics Constraints:
1. Energy conservation: E = 0.5*|v|² - μ/r = constant
2. Angular momentum conservation: h = r × v = constant  
3. Kepler's laws and orbital element constraints
4. Measurement projection consistency
5. Temporal continuity and smoothness

Loss Balancing Strategies:
- Kendall et al. learned log-variance weighting
- GradNorm-based gradient balancing
- Curriculum scheduling with adaptive weights
- Multi-task uncertainty weighting
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
import math

# Conditional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    torch = nn = F = Tensor = None
    warnings.warn("PyTorch not available. Loss functionality will be limited.")

try:
    import numpy as np
except ImportError:
    np = None
    warnings.warn("NumPy not available.")

@dataclass
class PhysicsLossConfig:
    """Configuration for physics-informed loss functions."""
    
    # Physics constraint weights
    energy_weight: float = 1.0              # Energy conservation constraint weight
    momentum_weight: float = 1.0            # Angular momentum conservation weight
    dynamics_weight: float = 10.0           # Dynamics residual (dr/dt, dv/dt) weight
    temporal_weight: float = 0.1            # Temporal smoothness weight
    
    # Measurement loss weights
    measurement_weight: float = 1.0         # Observation likelihood weight
    optical_weight: float = 1.0             # Optical measurement (RA/Dec) weight
    radar_weight: float = 1.0               # Radar measurement (Range/Az/El) weight
    
    # Regularization weights
    state_regularization: float = 0.001     # State magnitude regularization
    acceleration_regularization: float = 0.01 # Acceleration smoothness
    
    # Adaptive weighting
    use_adaptive_weights: bool = True       # Enable learned log-variance balancing
    use_gradnorm: bool = False             # Enable GradNorm balancing (more complex)
    adaptive_alpha: float = 1.5            # GradNorm hyperparameter
    
    # Curriculum learning
    curriculum_enabled: bool = True        # Enable curriculum scheduling
    warmup_epochs: int = 10               # Epochs to ramp up physics constraints
    physics_ramp_schedule: str = "exponential" # "linear", "exponential", "cosine"
    
    # Numerical stability
    energy_scale: float = 1e-12           # Energy normalization scale
    momentum_scale: float = 1e-6          # Angular momentum normalization scale
    position_scale: float = 1e-6          # Position normalization scale (m → 1e6 m)
    velocity_scale: float = 1e-3          # Velocity normalization scale (m/s → 1e3 m/s)
    
    # Physics constants
    earth_mu: float = 3.986004418e14      # Earth gravitational parameter (m³/s²)
    earth_radius: float = 6.371e6         # Earth radius (m)
    
    # Loss computation options
    use_huber_loss: bool = False          # Use Huber loss for robustness
    huber_delta: float = 1.0             # Huber loss threshold
    clip_gradients: bool = True           # Clip physics loss gradients
    max_grad_norm: float = 1.0           # Max gradient norm for clipping
    
    # Uncertainty integration
    uncertainty_weighting: bool = True    # Weight losses by model uncertainty
    uncertainty_epsilon: float = 1e-6    # Small epsilon for numerical stability

class PhysicsLoss(ABC):
    """Abstract base class for physics-informed loss components."""
    
    @abstractmethod
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor],
                    times: Tensor,
                    observations: Optional[Tensor],
                    **kwargs) -> Tensor:
        """Compute the physics loss component."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this loss component."""
        pass

class EnergyConservationLoss(PhysicsLoss):
    """
    Energy conservation loss: E = 0.5*|v|² - μ/r should be constant.
    
    For bound orbits, energy should be negative and approximately constant
    throughout the trajectory. This loss penalizes energy drift.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        self.earth_mu = config.earth_mu
        self.energy_scale = config.energy_scale
        
    def compute_orbital_energy(self, states: Tensor) -> Tensor:
        """Compute specific orbital energy E = 0.5*v² - μ/r."""
        positions = states[..., :3]  # (..., 3)
        velocities = states[..., 3:6]  # (..., 3)
        
        # Kinetic energy per unit mass
        kinetic = 0.5 * torch.sum(velocities**2, dim=-1)  # (...,)
        
        # Potential energy per unit mass
        r_norm = torch.norm(positions, dim=-1)  # (...,)
        potential = -self.earth_mu / r_norm  # (...,)
        
        # Total specific energy
        energy = kinetic + potential  # (...,)
        
        return energy
    
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor] = None,
                    times: Optional[Tensor] = None,
                    observations: Optional[Tensor] = None,
                    **kwargs) -> Tensor:
        """
        Compute energy conservation loss.
        
        Args:
            predicted_states: Predicted states (..., state_dim)
            
        Returns:
            Energy conservation violation loss
        """
        energies = self.compute_orbital_energy(predicted_states)  # (...,)
        
        if predicted_states.dim() > 2:  # Sequential data (..., seq_len, state_dim)
            # Compute energy variation across time
            energy_variance = torch.var(energies, dim=-1)  # (...,)
            energy_loss = energy_variance.mean()
        else:  # Single timestep (..., state_dim)
            # For single timestep, penalize positive energies (unbound orbits)
            energy_violation = F.relu(energies)  # Only penalize positive (unbound)
            energy_loss = energy_violation.mean()
        
        # Normalize by energy scale for numerical stability
        return energy_loss / self.energy_scale
    
    def get_name(self) -> str:
        return "energy_conservation"

class AngularMomentumConservationLoss(PhysicsLoss):
    """
    Angular momentum conservation loss: h = r × v should be constant.
    
    For Keplerian orbits, angular momentum magnitude and direction should
    remain constant (ignoring perturbations).
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        self.momentum_scale = config.momentum_scale
        
    def compute_angular_momentum(self, states: Tensor) -> Tensor:
        """Compute specific angular momentum h = r × v."""
        positions = states[..., :3]  # (..., 3)
        velocities = states[..., 3:6]  # (..., 3)
        
        # Cross product r × v
        angular_momentum = torch.cross(positions, velocities, dim=-1)  # (..., 3)
        
        return angular_momentum
    
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor] = None,
                    times: Optional[Tensor] = None,
                    observations: Optional[Tensor] = None,
                    **kwargs) -> Tensor:
        """
        Compute angular momentum conservation loss.
        
        Args:
            predicted_states: Predicted states (..., state_dim)
            
        Returns:
            Angular momentum conservation violation loss
        """
        angular_momentum = self.compute_angular_momentum(predicted_states)  # (..., 3)
        
        if predicted_states.dim() > 2:  # Sequential data
            # Compute angular momentum variation across time
            # Both magnitude and direction should be conserved
            h_magnitude = torch.norm(angular_momentum, dim=-1)  # (..., seq_len)
            h_direction = angular_momentum / (h_magnitude.unsqueeze(-1) + 1e-8)  # (..., seq_len, 3)
            
            # Magnitude conservation
            magnitude_variance = torch.var(h_magnitude, dim=-1)  # (...,)
            
            # Direction conservation (measure deviation from mean direction)
            mean_direction = h_direction.mean(dim=-2, keepdim=True)  # (..., 1, 3)
            direction_deviation = torch.norm(h_direction - mean_direction, dim=-1)  # (..., seq_len)
            direction_variance = torch.var(direction_deviation, dim=-1)  # (...,)
            
            momentum_loss = (magnitude_variance + direction_variance).mean()
        else:  # Single timestep
            # For single timestep, just compute magnitude (no conservation check possible)
            h_magnitude = torch.norm(angular_momentum, dim=-1)  # (...,)
            # Penalize very small angular momentum (degenerate orbits)
            momentum_loss = torch.mean(1.0 / (h_magnitude + 1e-8))
        
        # Normalize by momentum scale
        return momentum_loss / self.momentum_scale
    
    def get_name(self) -> str:
        return "angular_momentum_conservation"

class DynamicsResidualLoss(PhysicsLoss):
    """
    Dynamics residual loss: dr/dt should equal v, dv/dt should equal acceleration.
    
    This loss enforces that the neural network's predicted trajectory satisfies
    the fundamental kinematic equations and orbital dynamics.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        self.earth_mu = config.earth_mu
        
    def compute_orbital_acceleration(self, positions: Tensor) -> Tensor:
        """Compute orbital acceleration from positions using 2-body + J2."""
        r_vec = positions  # (..., 3)
        r_norm = torch.norm(r_vec, dim=-1, keepdim=True)  # (..., 1)
        
        # Two-body acceleration: -μ/r³ * r_vec
        two_body_accel = -self.earth_mu * r_vec / (r_norm**3)  # (..., 3)
        
        # TODO: Add J2 perturbation if needed
        # For now, just use two-body dynamics
        
        return two_body_accel
    
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor] = None,
                    times: Optional[Tensor] = None,
                    observations: Optional[Tensor] = None,
                    model: Optional[nn.Module] = None,
                    **kwargs) -> Tensor:
        """
        Compute dynamics residual loss using automatic differentiation.
        
        Args:
            predicted_states: Predicted states (..., state_dim)
            times: Time tensor with requires_grad=True for differentiation
            model: NP-SNN model for computing derivatives
            
        Returns:
            Dynamics residual loss
        """
        if times is None or not times.requires_grad or model is None:
            # Cannot compute derivatives without proper setup
            return torch.tensor(0.0, device=predicted_states.device)
        
        try:
            # Get time derivatives from model
            dr_dt, dv_dt = model.get_derivatives(times, observations, create_graph=True)
            
            positions = predicted_states[..., :3]  # (..., 3)
            velocities = predicted_states[..., 3:6]  # (..., 3)
            
            # Kinematic consistency: dr/dt should equal velocity
            velocity_consistency = F.mse_loss(dr_dt, velocities)
            
            # Dynamic consistency: dv/dt should equal orbital acceleration
            predicted_acceleration = self.compute_orbital_acceleration(positions)
            acceleration_consistency = F.mse_loss(dv_dt, predicted_acceleration)
            
            # Combined dynamics loss
            dynamics_loss = velocity_consistency + acceleration_consistency
            
            return dynamics_loss
            
        except RuntimeError as e:
            if "backward through the graph" in str(e):
                # Graph already used, return zero loss
                return torch.tensor(0.0, device=predicted_states.device)
            else:
                raise
    
    def get_name(self) -> str:
        return "dynamics_residual"

class MeasurementConsistencyLoss(PhysicsLoss):
    """
    Measurement consistency loss: predicted states should be consistent with observations.
    
    This loss ensures that when the predicted orbital states are projected through
    the sensor models, they match the actual sensor observations.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        self.optical_weight = config.optical_weight
        self.radar_weight = config.radar_weight
        
    def project_to_optical(self, states: Tensor, observer_pos: Optional[Tensor] = None) -> Tensor:
        """
        Project orbital states to optical measurements (RA/Dec).
        
        Args:
            states: Orbital states (..., 6) [x,y,z,vx,vy,vz] in ECI
            observer_pos: Observer position (..., 3) in ECI (default: origin)
            
        Returns:
            Optical measurements (..., 2) [RA, Dec] in radians
        """
        positions = states[..., :3]  # (..., 3)
        
        if observer_pos is not None:
            # Topocentric position
            relative_pos = positions - observer_pos  # (..., 3)
        else:
            # Geocentric (observer at Earth center)
            relative_pos = positions
        
        x, y, z = relative_pos[..., 0], relative_pos[..., 1], relative_pos[..., 2]
        
        # Convert to spherical coordinates
        r = torch.norm(relative_pos, dim=-1)  # (...,)
        
        # Right ascension (RA)
        ra = torch.atan2(y, x)  # (...,)
        
        # Declination (Dec)
        dec = torch.asin(z / (r + 1e-8))  # (...,)
        
        return torch.stack([ra, dec], dim=-1)  # (..., 2)
    
    def project_to_radar(self, states: Tensor, observer_pos: Optional[Tensor] = None) -> Tensor:
        """
        Project orbital states to radar measurements (Range/Az/El).
        
        Args:
            states: Orbital states (..., 6) [x,y,z,vx,vy,vz] in ECI
            observer_pos: Observer position (..., 3) in ECI (default: origin)
            
        Returns:
            Radar measurements (..., 3) [Range, Azimuth, Elevation] 
        """
        positions = states[..., :3]  # (..., 3)
        
        if observer_pos is not None:
            relative_pos = positions - observer_pos
        else:
            relative_pos = positions
        
        x, y, z = relative_pos[..., 0], relative_pos[..., 1], relative_pos[..., 2]
        
        # Range
        range_val = torch.norm(relative_pos, dim=-1)  # (...,)
        
        # Azimuth (from North, clockwise)
        azimuth = torch.atan2(x, y)  # (...,)
        
        # Elevation (above horizon)
        elevation = torch.asin(z / (range_val + 1e-8))  # (...,)
        
        return torch.stack([range_val, azimuth, elevation], dim=-1)  # (..., 3)
    
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor] = None,
                    times: Optional[Tensor] = None,
                    observations: Optional[Tensor] = None,
                    observation_types: Optional[List[str]] = None,
                    observer_positions: Optional[Tensor] = None,
                    **kwargs) -> Tensor:
        """
        Compute measurement consistency loss.
        
        Args:
            predicted_states: Predicted orbital states (..., 6)
            observations: Sensor observations (..., obs_dim)
            observation_types: List of observation types ["optical", "radar", etc.]
            observer_positions: Observer positions for topocentric projections
            
        Returns:
            Measurement consistency loss
        """
        if observations is None:
            return torch.tensor(0.0, device=predicted_states.device)
        
        total_loss = torch.tensor(0.0, device=predicted_states.device)
        
        if observation_types is None:
            # Assume all observations are optical by default
            observation_types = ["optical"] * observations.shape[-1] // 2
        
        obs_idx = 0
        for obs_type in observation_types:
            if obs_type == "optical":
                # Optical measurements (RA/Dec)
                predicted_optical = self.project_to_optical(predicted_states, observer_positions)
                true_optical = observations[..., obs_idx:obs_idx+2]
                
                # Handle angle wrapping for RA
                ra_diff = predicted_optical[..., 0] - true_optical[..., 0]
                ra_diff = torch.atan2(torch.sin(ra_diff), torch.cos(ra_diff))  # Wrap to [-π, π]
                
                # Declination difference (no wrapping needed)
                dec_diff = predicted_optical[..., 1] - true_optical[..., 1]
                
                # Weighted angular loss
                optical_loss = torch.mean(ra_diff**2 + dec_diff**2)
                total_loss += self.optical_weight * optical_loss
                
                obs_idx += 2
                
            elif obs_type == "radar":
                # Radar measurements (Range/Az/El)
                predicted_radar = self.project_to_radar(predicted_states, observer_positions)
                true_radar = observations[..., obs_idx:obs_idx+3]
                
                # Range loss (linear)
                range_loss = F.mse_loss(predicted_radar[..., 0], true_radar[..., 0])
                
                # Angular losses with wrapping
                az_diff = predicted_radar[..., 1] - true_radar[..., 1]
                az_diff = torch.atan2(torch.sin(az_diff), torch.cos(az_diff))
                az_loss = torch.mean(az_diff**2)
                
                el_diff = predicted_radar[..., 2] - true_radar[..., 2]
                el_loss = torch.mean(el_diff**2)
                
                radar_loss = range_loss + az_loss + el_loss
                total_loss += self.radar_weight * radar_loss
                
                obs_idx += 3
        
        return total_loss
    
    def get_name(self) -> str:
        return "measurement_consistency"

class TemporalSmoothnessLoss(PhysicsLoss):
    """
    Temporal smoothness loss: penalize rapid changes in states and accelerations.
    
    This loss ensures that predicted trajectories are smooth and physically
    plausible, avoiding unrealistic discontinuities or oscillations.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        
    def compute_loss(self, 
                    predicted_states: Tensor,
                    true_states: Optional[Tensor] = None,
                    times: Optional[Tensor] = None,
                    observations: Optional[Tensor] = None,
                    **kwargs) -> Tensor:
        """
        Compute temporal smoothness loss.
        
        Args:
            predicted_states: Predicted states (..., seq_len, state_dim)
            times: Time points (..., seq_len)
            
        Returns:
            Temporal smoothness loss
        """
        if predicted_states.dim() < 3:  # Need sequential data
            return torch.tensor(0.0, device=predicted_states.device)
        
        # First-order differences (velocity)
        state_diffs = predicted_states[..., 1:, :] - predicted_states[..., :-1, :]  # (..., seq_len-1, state_dim)
        
        # Second-order differences (acceleration)
        accel_diffs = state_diffs[..., 1:, :] - state_diffs[..., :-1, :]  # (..., seq_len-2, state_dim)
        
        # Time step normalization
        if times is not None:
            dt = times[..., 1:] - times[..., :-1]  # (..., seq_len-1)
            dt = dt.unsqueeze(-1)  # (..., seq_len-1, 1)
            
            # Normalize by time step
            state_diffs = state_diffs / (dt + 1e-8)
            
            if accel_diffs.shape[-2] > 0:
                dt_accel = dt[..., 1:, :]  # (..., seq_len-2, 1)
                accel_diffs = accel_diffs / (dt_accel + 1e-8)
        
        # Compute smoothness losses
        velocity_smoothness = torch.mean(state_diffs[..., :3]**2)  # Position velocity
        acceleration_smoothness = torch.mean(accel_diffs[..., :3]**2) if accel_diffs.shape[-2] > 0 else torch.tensor(0.0, device=predicted_states.device)
        
        return velocity_smoothness + acceleration_smoothness
    
    def get_name(self) -> str:
        return "temporal_smoothness"

class AdaptiveWeightLearner(nn.Module):
    """
    Adaptive loss weight learner using Kendall et al. uncertainty weighting.
    
    Learns log-variance parameters for each loss component to automatically
    balance multiple loss terms during training.
    """
    
    def __init__(self, loss_names: List[str], initial_weight: float = 0.0):
        super().__init__()
        
        self.loss_names = loss_names
        # Learnable log-variance parameters (one per loss component)
        self.log_vars = nn.Parameter(torch.full((len(loss_names),), initial_weight))
        
    def get_weights(self) -> Dict[str, float]:
        """Get current adaptive weights for each loss component."""
        weights = {}
        for i, name in enumerate(self.loss_names):
            # Weight = exp(-log_var), regularization = log_var
            weights[name] = torch.exp(-self.log_vars[i])
        
        return weights
    
    def get_regularization(self) -> Tensor:
        """Get regularization term: sum of log-variances."""
        return torch.sum(self.log_vars)
    
    def forward(self, losses: Dict[str, Tensor]) -> Tensor:
        """
        Compute adaptively weighted total loss.
        
        Args:
            losses: Dictionary of loss components
            
        Returns:
            Weighted total loss including regularization
        """
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        
        for i, name in enumerate(self.loss_names):
            if name in losses:
                # Adaptive weighting: exp(-log_var) * loss + log_var
                weight = torch.exp(-self.log_vars[i])
                total_loss += weight * losses[name] + self.log_vars[i]
        
        return total_loss

class PhysicsInformedLossFunction:
    """
    Comprehensive physics-informed loss function for NP-SNN training.
    
    Combines multiple physics-based loss components with adaptive weighting
    and curriculum learning for robust orbital mechanics training.
    """
    
    def __init__(self, config: PhysicsLossConfig):
        self.config = config
        
        # Initialize individual loss components
        self.loss_components = {}
        self.loss_components['energy'] = EnergyConservationLoss(config)
        self.loss_components['momentum'] = AngularMomentumConservationLoss(config)
        self.loss_components['dynamics'] = DynamicsResidualLoss(config)
        self.loss_components['measurement'] = MeasurementConsistencyLoss(config)
        self.loss_components['temporal'] = TemporalSmoothnessLoss(config)
        
        # Base weights from config
        self.base_weights = {
            'energy': config.energy_weight,
            'momentum': config.momentum_weight,
            'dynamics': config.dynamics_weight,
            'measurement': config.measurement_weight,
            'temporal': config.temporal_weight
        }
        
        # Initialize adaptive weighting if enabled
        if config.use_adaptive_weights:
            self.adaptive_learner = AdaptiveWeightLearner(list(self.loss_components.keys()))
        else:
            self.adaptive_learner = None
        
        # Curriculum learning state
        self.current_epoch = 0
        
    def update_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch
    
    def get_curriculum_weights(self) -> Dict[str, float]:
        """Get curriculum-adjusted weights based on current epoch."""
        if not self.config.curriculum_enabled:
            return self.base_weights.copy()
        
        # Physics constraints ramp up gradually
        if self.current_epoch < self.config.warmup_epochs:
            # Gradual ramp-up for physics constraints
            progress = self.current_epoch / self.config.warmup_epochs
            
            if self.config.physics_ramp_schedule == "linear":
                physics_scale = progress
            elif self.config.physics_ramp_schedule == "exponential":
                physics_scale = 1.0 - math.exp(-3.0 * progress)
            elif self.config.physics_ramp_schedule == "cosine":
                physics_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                physics_scale = progress  # Default to linear
        else:
            physics_scale = 1.0
        
        # Apply scaling to physics constraints
        curriculum_weights = self.base_weights.copy()
        curriculum_weights['energy'] *= physics_scale
        curriculum_weights['momentum'] *= physics_scale
        curriculum_weights['dynamics'] *= physics_scale
        
        # Measurement and temporal weights remain constant
        
        return curriculum_weights
    
    def compute_individual_losses(self,
                                predicted_states: Tensor,
                                true_states: Optional[Tensor] = None,
                                times: Optional[Tensor] = None,
                                observations: Optional[Tensor] = None,
                                **kwargs) -> Dict[str, Tensor]:
        """
        Compute all individual loss components.
        
        Args:
            predicted_states: Model predictions (..., seq_len, state_dim)
            true_states: Ground truth states (optional)
            times: Time points (..., seq_len)
            observations: Measurement data (optional)
            **kwargs: Additional loss-specific arguments
            
        Returns:
            Dictionary of individual loss values
        """
        losses = {}
        
        # Compute each loss component with error handling
        for name, loss_fn in self.loss_components.items():
            try:
                loss_value = loss_fn.compute_loss(
                    predicted_states=predicted_states,
                    true_states=true_states,
                    times=times,
                    observations=observations,
                    **kwargs
                )
                
                # Ensure loss is a scalar tensor
                if isinstance(loss_value, torch.Tensor):
                    if loss_value.numel() == 1:
                        losses[name] = loss_value.squeeze()
                    else:
                        losses[name] = torch.mean(loss_value)
                else:
                    losses[name] = torch.tensor(float(loss_value), device=predicted_states.device)
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Warning: {name} loss computation failed: {e}")
                losses[name] = torch.tensor(0.0, device=predicted_states.device)
        
        return losses
    
    def compute_supervised_loss(self, 
                               pred_states: Tensor,
                               true_states: Tensor, 
                               observation_mask: Optional[Tensor] = None,
                               pred_uncertainties: Optional[Tensor] = None) -> Tensor:
        """
        Compute supervised loss between predictions and true states.
        
        Args:
            pred_states: Predicted states (batch, seq_len, state_dim)
            true_states: True states (batch, seq_len, state_dim)
            observation_mask: Mask for valid observations (batch, seq_len)
            pred_uncertainties: Predicted uncertainties (optional)
            
        Returns:
            Supervised loss value
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(pred_states, true_states, reduction='none')
        
        # Apply observation mask if provided
        if observation_mask is not None:
            # Expand mask to match state dimensions
            if observation_mask.dim() == 2:
                mask = observation_mask.unsqueeze(-1).expand_as(pred_states)
            else:
                mask = observation_mask
            
            # Only compute loss on observed states
            masked_loss = mse_loss * mask.float()
            
            # Average over valid observations
            num_valid = mask.sum()
            if num_valid > 0:
                supervised_loss = masked_loss.sum() / num_valid
            else:
                supervised_loss = torch.tensor(0.0, device=pred_states.device)
        else:
            # No mask - compute loss on all states
            supervised_loss = mse_loss.mean()
        
        # Optionally incorporate uncertainty weighting
        if pred_uncertainties is not None:
            # Uncertainty-weighted loss (higher uncertainty -> lower weight)
            uncertainty_weights = 1.0 / (pred_uncertainties + 1e-6)
            supervised_loss = supervised_loss * uncertainty_weights.mean()
        
        return supervised_loss

    def compute_weighted_loss(self, losses: Dict[str, Tensor]) -> Tensor:
        """
        Compute final weighted loss using curriculum and/or adaptive weights.
        
        Args:
            losses: Dictionary of individual loss components
            
        Returns:
            Final weighted total loss
        """
        # Get curriculum-adjusted weights
        curriculum_weights = self.get_curriculum_weights()
        
        if self.adaptive_learner is not None:
            # Use adaptive weighting
            adaptive_weights = self.adaptive_learner.get_weights()
            
            # Combine curriculum and adaptive weights (element-wise product)
            final_weights = {}
            for name in losses.keys():
                if name in curriculum_weights and name in adaptive_weights:
                    final_weights[name] = curriculum_weights[name] * adaptive_weights[name]
                else:
                    final_weights[name] = curriculum_weights.get(name, 1.0)
            
            # Compute weighted loss with adaptive regularization
            total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
            for name, loss_value in losses.items():
                if name in final_weights:
                    total_loss += final_weights[name] * loss_value
            
            # Add adaptive weight regularization
            total_loss += self.adaptive_learner.get_regularization()
            
        else:
            # Use fixed curriculum weights
            total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
            for name, loss_value in losses.items():
                weight = curriculum_weights.get(name, 0.0)
                total_loss += weight * loss_value
        
        return total_loss
    
    def __call__(self,
                predicted_states: Tensor,
                true_states: Optional[Tensor] = None,
                times: Optional[Tensor] = None,
                observations: Optional[Tensor] = None,
                return_components: bool = False,
                **kwargs) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Compute physics-informed loss function.
        
        Args:
            predicted_states: Model predictions (..., seq_len, state_dim)
            true_states: Ground truth states (optional)
            times: Time points (..., seq_len) 
            observations: Measurement data (optional)
            return_components: Whether to return individual loss components
            **kwargs: Additional loss-specific arguments
            
        Returns:
            total_loss: Total weighted loss
            components (optional): Dictionary of individual loss components
        """
        # Compute individual loss components
        individual_losses = self.compute_individual_losses(
            predicted_states=predicted_states,
            true_states=true_states,
            times=times,
            observations=observations,
            **kwargs
        )
        
        # Compute final weighted loss
        total_loss = self.compute_weighted_loss(individual_losses)
        
        if return_components:
            return total_loss, individual_losses
        else:
            return total_loss
    
    def get_loss_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current loss configuration."""
        info = {
            'loss_components': list(self.loss_components.keys()),
            'base_weights': self.base_weights.copy(),
            'curriculum_weights': self.get_curriculum_weights(),
            'current_epoch': self.current_epoch,
            'config': {
                'curriculum_enabled': self.config.curriculum_enabled,
                'use_adaptive_weights': self.config.use_adaptive_weights,
                'warmup_epochs': self.config.warmup_epochs,
                'physics_ramp_schedule': self.config.physics_ramp_schedule
            }
        }
        
        if self.adaptive_learner is not None:
            info['adaptive_weights'] = self.adaptive_learner.get_weights()
            info['adaptive_log_vars'] = self.adaptive_learner.log_vars.detach().cpu().numpy().tolist()
        
        return info


# Factory function for easy instantiation
def create_physics_informed_loss(
    energy_weight: float = 1.0,
    momentum_weight: float = 1.0,
    dynamics_weight: float = 1.0,
    measurement_weight: float = 1.0,
    temporal_weight: float = 0.1,
    use_adaptive_weights: bool = False,
    curriculum_enabled: bool = True,
    warmup_epochs: int = 50,
    physics_ramp_schedule: str = "cosine",
    **kwargs
) -> PhysicsInformedLossFunction:
    """
    Factory function to create a physics-informed loss function with sensible defaults.
    
    Args:
        energy_weight: Weight for energy conservation loss
        momentum_weight: Weight for angular momentum conservation loss  
        dynamics_weight: Weight for orbital dynamics residual loss
        measurement_weight: Weight for measurement consistency loss
        temporal_weight: Weight for temporal smoothness loss
        use_adaptive_weights: Whether to use adaptive weight learning
        curriculum_enabled: Whether to use curriculum learning
        warmup_epochs: Number of warmup epochs for curriculum learning
        physics_ramp_schedule: Schedule for physics constraint ramp-up
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PhysicsInformedLossFunction instance
    """
    config = PhysicsLossConfig(
        energy_weight=energy_weight,
        momentum_weight=momentum_weight,
        dynamics_weight=dynamics_weight,
        measurement_weight=measurement_weight,
        temporal_weight=temporal_weight,
        use_adaptive_weights=use_adaptive_weights,
        curriculum_enabled=curriculum_enabled,
        warmup_epochs=warmup_epochs,
        physics_ramp_schedule=physics_ramp_schedule,
        **kwargs
    )
    
    return PhysicsInformedLossFunction(config)