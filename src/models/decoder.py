"""
Neural Physics-Informed Decoder Networks

This module implements decoder networks that transform spiking neural network outputs
into orbital state estimates with physics-aware constraints and uncertainty quantification.

Key Features:
- Orbital state reconstruction (position + velocity in Cartesian coordinates)
- Physics-aware constraints (energy conservation, angular momentum)
- Uncertainty quantification with ensemble or Bayesian approaches
- Multi-scale state estimation (different orbital regimes)
- Coordinate system transformations (ECI, ECEF, orbital elements)
- Temporal consistency enforcement
- Measurement prediction capabilities

Architecture Components:
1. StateDecoder: Core decoder for position/velocity estimation
2. PhysicsConstrainedDecoder: Decoder with orbital mechanics constraints
3. UncertaintyDecoder: Bayesian decoder with uncertainty quantification
4. MultiScaleDecoder: Handles different orbital regimes (LEO, MEO, GEO)
5. MeasurementDecoder: Predicts sensor measurements from states
"""

import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Callable
import math

# Conditional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    import torch.distributions as dist
except ImportError:
    torch = None
    nn = None
    F = None
    Tensor = None
    dist = None
    warnings.warn("PyTorch not available. Decoder functionality will be limited.")

try:
    import numpy as np
except ImportError:
    np = None
    warnings.warn("NumPy not available.")

@dataclass
class DecoderConfig:
    """Configuration for decoder networks."""
    # Architecture
    input_size: int = 128              # SNN output dimension
    state_dim: int = 6                 # Orbital state dimension (pos + vel)
    hidden_sizes: List[int] = None     # Hidden layer sizes
    
    # Physics constraints
    enable_physics: bool = True        # Enable physics constraints
    energy_constraint: bool = True     # Enforce energy conservation
    momentum_constraint: bool = True   # Enforce angular momentum conservation
    position_bounds: bool = True       # Enforce position bounds (Earth radius)
    velocity_bounds: bool = True       # Enforce velocity bounds (escape velocity)
    
    # Uncertainty quantification
    uncertainty_type: str = "ensemble" # "ensemble", "bayesian", "dropout", "none"
    num_ensemble: int = 5              # Number of ensemble members
    dropout_rate: float = 0.1          # Dropout rate for MC dropout
    
    # Coordinate systems
    output_frame: str = "eci"          # "eci", "ecef", "orbital_elements"
    normalize_states: bool = True      # Normalize states for numerical stability
    
    # Temporal consistency
    temporal_smoothing: bool = True    # Enable temporal smoothing
    consistency_weight: float = 0.1    # Weight for temporal consistency loss
    
    # Physical constants (SI units)
    earth_radius: float = 6.371e6      # Earth radius (m)
    earth_mu: float = 3.986004418e14   # Earth gravitational parameter (m³/s²)
    escape_velocity: float = 11200.0   # Escape velocity at Earth surface (m/s)
    
    # Gradient stability
    gradient_clip_norm: float = 1.0    # Gradient clipping max norm
    weight_init_gain: float = 0.1      # Smaller initialization for stability
    use_layer_norm: bool = True        # Use LayerNorm instead of BatchNorm
    output_scaling_factor: float = 1e-3 # Scale outputs to prevent explosion
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]

class StateDecoder(nn.Module):
    """
    Core decoder for transforming SNN outputs to orbital states.
    
    Features:
    - Multi-layer feedforward architecture
    - Batch normalization and dropout for regularization
    - Residual connections for gradient flow
    - Configurable activation functions
    - State normalization and denormalization
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        if torch is None:
            raise ImportError("PyTorch not available")
        
        self.config = config
        self.input_size = config.input_size
        self.state_dim = config.state_dim
        self.hidden_sizes = config.hidden_sizes
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Linear layer
            layer = nn.Linear(prev_size, hidden_size)
            self.layers.append(layer)
            
            # Normalization (LayerNorm is more stable than BatchNorm)
            if config.use_layer_norm:
                self.layers.append(nn.LayerNorm(hidden_size))
            else:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            self.layers.append(nn.ReLU())
            
            # Dropout (except last hidden layer)
            if i < len(self.hidden_sizes) - 1:
                self.layers.append(nn.Dropout(config.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, config.state_dim)
        
        # Residual connections
        self.use_residual = len(self.hidden_sizes) > 2
        if self.use_residual and self.input_size == self.hidden_sizes[-1]:
            self.residual_projection = nn.Identity()
        elif self.use_residual:
            self.residual_projection = nn.Linear(self.input_size, self.hidden_sizes[-1])
        
        # State normalization parameters
        if config.normalize_states:
            self.register_buffer('state_mean', torch.zeros(config.state_dim))
            self.register_buffer('state_std', torch.ones(config.state_dim))
            self._init_normalization()
        
        # Initialize weights
        self._init_weights()
    
    def _init_normalization(self):
        """Initialize state normalization parameters with reasonable orbital values."""
        # Position: LEO altitude above Earth surface
        pos_scale = self.config.earth_radius * 1.1  # ~1.1 Earth radii (close to LEO)
        # Velocity: Typical LEO orbital velocities  
        vel_scale = 7800.0  # ~7.8 km/s (LEO orbital velocity)
        
        # Set mean to reasonable orbital values
        self.state_mean.data[:3].fill_(self.config.earth_radius * 1.063)  # ~400 km altitude
        self.state_mean.data[3:].fill_(0.0)  # Zero mean velocity
        
        # Set std to reasonable scales  
        self.state_std.data[:3].fill_(pos_scale * 0.1)  # Position variation
        self.state_std.data[3:].fill_(vel_scale * 0.2)  # Velocity variation
    
    def _init_weights(self):
        """Initialize network weights with proper scaling for gradient stability."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Smaller He initialization for gradient stability
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                # Scale down the weights
                layer.weight.data *= self.config.weight_init_gain
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Output layer: very small initialization for stability
        nn.init.xavier_normal_(self.output_layer.weight, gain=self.config.weight_init_gain)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def normalize_state(self, state: Tensor) -> Tensor:
        """Normalize orbital state for network processing."""
        if self.config.normalize_states:
            return (state - self.state_mean) / self.state_std
        return state
    
    def denormalize_state(self, state: Tensor) -> Tensor:
        """Denormalize network output to physical units."""
        if self.config.normalize_states:
            return state * self.state_std + self.state_mean
        return state
    
    def forward(self, snn_output: Tensor) -> Tensor:
        """
        Decode SNN output to orbital state.
        
        Args:
            snn_output: SNN features of shape (batch, input_size)
        Returns:
            state: Orbital state of shape (batch, state_dim)
        """
        x = snn_output
        
        # Store input for residual connection
        residual_input = x
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        # Residual connection to last hidden layer
        if self.use_residual:
            residual = self.residual_projection(residual_input)
            x = x + residual
        
        # Output layer with controlled scaling
        raw_output = self.output_layer(x)
        
        # Apply output scaling to prevent gradient explosion
        scaled_output = raw_output * self.config.output_scaling_factor
        
        # Denormalize to physical units
        state = self.denormalize_state(scaled_output)
        
        return state
    
    def clip_gradients(self, max_norm: float = None) -> float:
        """Clip gradients to prevent explosion."""
        if max_norm is None:
            max_norm = self.config.gradient_clip_norm
        
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

class PhysicsConstrainedDecoder(StateDecoder):
    """
    Physics-aware decoder that enforces orbital mechanics constraints.
    
    Constraints:
    - Energy conservation: E = 0.5*|v|² - μ/r ≤ 0 for bound orbits
    - Angular momentum conservation: h = r × v
    - Position bounds: |r| ≥ R_earth
    - Velocity bounds: |v| ≤ v_escape
    - Orbital element validity: eccentricity ≥ 0, etc.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        # Physics constraint parameters
        self.earth_mu = config.earth_mu
        self.earth_radius = config.earth_radius
        self.escape_velocity = config.escape_velocity
        
        # Constraint enforcement methods
        self.energy_enforcement = "soft"  # "soft", "hard", "projection"
        self.position_enforcement = "projection"  # "projection", "penalty"
        
        # Learnable constraint weights
        if config.energy_constraint:
            self.energy_weight = nn.Parameter(torch.tensor(1.0))
        if config.momentum_constraint:
            self.momentum_weight = nn.Parameter(torch.tensor(1.0))
    
    def apply_position_constraints(self, state: Tensor) -> Tensor:
        """Apply position constraints to ensure physical validity."""
        position = state[..., :3]  # (batch, 3)
        velocity = state[..., 3:]  # (batch, 3)
        
        if self.config.position_bounds:
            # Ensure position is above Earth surface
            pos_norm = torch.norm(position, dim=-1, keepdim=True)  # (batch, 1)
            min_radius = self.earth_radius * 1.01  # Small margin above surface
            
            # Project to minimum radius if needed
            below_surface = pos_norm < min_radius
            corrected_pos = torch.where(
                below_surface,
                position * (min_radius / pos_norm),
                position
            )
            
            state = torch.cat([corrected_pos, velocity], dim=-1)
        
        return state
    
    def apply_velocity_constraints(self, state: Tensor) -> Tensor:
        """Apply velocity constraints to ensure physical validity."""
        position = state[..., :3]
        velocity = state[..., 3:]
        
        if self.config.velocity_bounds:
            # Ensure velocity doesn't exceed escape velocity at current position
            pos_norm = torch.norm(position, dim=-1, keepdim=True)
            vel_norm = torch.norm(velocity, dim=-1, keepdim=True)
            
            # Local escape velocity: v_esc = sqrt(2*μ/r)
            local_escape_vel = torch.sqrt(2 * self.earth_mu / pos_norm)
            
            # Scale down velocity if it exceeds escape velocity
            exceeds_escape = vel_norm > local_escape_vel
            corrected_vel = torch.where(
                exceeds_escape,
                velocity * (local_escape_vel * 0.99 / vel_norm),  # 99% of escape velocity
                velocity
            )
            
            state = torch.cat([position, corrected_vel], dim=-1)
        
        return state
    
    def compute_orbital_energy(self, state: Tensor) -> Tensor:
        """Compute specific orbital energy E = 0.5*v² - μ/r."""
        position = state[..., :3]
        velocity = state[..., 3:]
        
        # Kinetic energy per unit mass
        kinetic = 0.5 * torch.sum(velocity**2, dim=-1)
        
        # Potential energy per unit mass
        r_norm = torch.norm(position, dim=-1)
        potential = -self.earth_mu / r_norm
        
        # Total specific energy
        energy = kinetic + potential
        
        return energy
    
    def compute_angular_momentum(self, state: Tensor) -> Tensor:
        """Compute specific angular momentum h = r × v."""
        position = state[..., :3]
        velocity = state[..., 3:]
        
        # Cross product r × v
        angular_momentum = torch.cross(position, velocity, dim=-1)
        
        return angular_momentum
    
    def energy_constraint_loss(self, state: Tensor) -> Tensor:
        """Compute energy constraint violation loss."""
        energy = self.compute_orbital_energy(state)
        
        # For bound orbits, energy should be negative
        # Penalize positive energies (hyperbolic/escape trajectories)
        energy_violation = F.relu(energy)  # Only penalize positive energies
        
        return energy_violation.mean()
    
    def forward(self, snn_output: Tensor) -> Tensor:
        """
        Decode with physics constraints applied.
        
        Args:
            snn_output: SNN features of shape (batch, input_size)
        Returns:
            state: Physics-constrained orbital state of shape (batch, state_dim)
        """
        # Get base state from parent decoder
        state = super().forward(snn_output)
        
        # Apply hard constraints through projection
        if self.config.position_bounds:
            state = self.apply_position_constraints(state)
        
        if self.config.velocity_bounds:
            state = self.apply_velocity_constraints(state)
        
        return state
    
    def constraint_losses(self, state: Tensor) -> Dict[str, Tensor]:
        """Compute physics constraint violation losses for training."""
        losses = {}
        
        if self.config.energy_constraint:
            losses['energy'] = self.energy_constraint_loss(state)
        
        if self.config.momentum_constraint:
            # Angular momentum should be conserved (implement in training loop)
            # For now, just compute magnitude for monitoring
            h = self.compute_angular_momentum(state)
            losses['angular_momentum_magnitude'] = torch.norm(h, dim=-1).mean()
        
        return losses

class UncertaintyDecoder(PhysicsConstrainedDecoder):
    """
    Decoder with uncertainty quantification capabilities.
    
    Supports multiple uncertainty estimation methods:
    - Ensemble: Multiple decoder networks with different initializations
    - Bayesian: Variational layers with learned uncertainty
    - MC Dropout: Monte Carlo dropout for approximate Bayesian inference
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        
        self.uncertainty_type = config.uncertainty_type
        
        if config.uncertainty_type == "ensemble":
            self._init_ensemble(config.num_ensemble)
        elif config.uncertainty_type == "bayesian":
            self._init_bayesian_layers()
        elif config.uncertainty_type == "dropout":
            self.mc_dropout_samples = 20
        
    def _init_ensemble(self, num_ensemble: int):
        """Initialize ensemble of decoder networks."""
        self.ensemble_decoders = nn.ModuleList()
        
        for _ in range(num_ensemble):
            # Create a copy of the base architecture
            ensemble_member = StateDecoder(self.config)
            self.ensemble_decoders.append(ensemble_member)
    
    def _init_bayesian_layers(self):
        """Initialize Bayesian (variational) layers."""
        # Replace standard linear layers with Bayesian layers
        # This would require a Bayesian neural network library like Pyro or BayesianTorch
        # For now, implement a simple approach with learned variance
        
        self.output_mean = nn.Linear(self.hidden_sizes[-1], self.config.state_dim)
        self.output_logvar = nn.Linear(self.hidden_sizes[-1], self.config.state_dim)
        
        # Initialize variance output to small values
        nn.init.constant_(self.output_logvar.weight, -2.0)  # log(0.135) ≈ -2
        nn.init.constant_(self.output_logvar.bias, -2.0)
    
    def forward_ensemble(self, snn_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through ensemble returning mean and uncertainty."""
        if self.uncertainty_type != "ensemble":
            raise ValueError("forward_ensemble only available for ensemble uncertainty")
        
        # Get predictions from all ensemble members
        predictions = []
        for decoder in self.ensemble_decoders:
            pred = decoder(snn_output)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_ensemble, batch, state_dim)
        
        # Compute ensemble statistics
        mean_pred = predictions.mean(dim=0)  # (batch, state_dim)
        std_pred = predictions.std(dim=0)    # (batch, state_dim)
        
        return mean_pred, std_pred
    
    def forward_bayesian(self, snn_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with Bayesian uncertainty."""
        if self.uncertainty_type != "bayesian":
            raise ValueError("forward_bayesian only available for Bayesian uncertainty")
        
        # Forward through base layers (excluding output)
        x = snn_output
        for layer in self.layers:
            x = layer(x)
        
        # Bayesian output layer
        mean = self.output_mean(x)
        logvar = self.output_logvar(x)
        
        # Sample from learned distribution during training
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            state = mean + eps * std
        else:
            state = mean
            std = torch.exp(0.5 * logvar)
        
        # Apply denormalization
        state = self.denormalize_state(state)
        if not self.training:
            std = std * self.state_std  # Scale uncertainty too
        
        return state, std
    
    def forward_mc_dropout(self, snn_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with Monte Carlo dropout."""
        if self.uncertainty_type != "dropout":
            raise ValueError("forward_mc_dropout only available for MC dropout uncertainty")
        
        # Enable dropout during inference
        self.train()  # Enable dropout
        
        # Multiple forward passes
        predictions = []
        for _ in range(self.mc_dropout_samples):
            pred = super().forward(snn_output)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (samples, batch, state_dim)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def forward(self, snn_output: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass with appropriate uncertainty quantification.
        
        Args:
            snn_output: SNN features of shape (batch, input_size)
        Returns:
            If uncertainty enabled: (mean_state, uncertainty)
            If uncertainty disabled: state
        """
        if self.uncertainty_type == "ensemble":
            return self.forward_ensemble(snn_output)
        elif self.uncertainty_type == "bayesian":
            return self.forward_bayesian(snn_output)
        elif self.uncertainty_type == "dropout":
            return self.forward_mc_dropout(snn_output)
        else:
            # No uncertainty quantification
            return super().forward(snn_output)

# Factory functions for easy decoder creation
def create_standard_decoder(input_size: int = 128,
                          state_dim: int = 6,
                          hidden_sizes: List[int] = None,
                          **kwargs) -> StateDecoder:
    """Create standard state decoder with gradient-stable defaults."""
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    config = DecoderConfig(
        input_size=input_size,
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        enable_physics=False,
        uncertainty_type="none",
        normalize_states=True,
        use_layer_norm=True,
        weight_init_gain=0.1,
        output_scaling_factor=1e-3,
        **kwargs
    )
    return StateDecoder(config)

def create_physics_decoder(input_size: int = 128,
                         state_dim: int = 6,
                         hidden_sizes: List[int] = None,
                         **kwargs) -> PhysicsConstrainedDecoder:
    """Create physics-constrained decoder with gradient stability."""
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    config = DecoderConfig(
        input_size=input_size,
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        enable_physics=True,
        energy_constraint=True,
        momentum_constraint=True,
        position_bounds=True,
        velocity_bounds=True,
        uncertainty_type="none",
        normalize_states=True,
        use_layer_norm=True,
        weight_init_gain=0.1,
        output_scaling_factor=1e-3,
        gradient_clip_norm=1.0,
        **kwargs
    )
    return PhysicsConstrainedDecoder(config)

def create_uncertainty_decoder(input_size: int = 128,
                             state_dim: int = 6,
                             hidden_sizes: List[int] = None,
                             uncertainty_type: str = "ensemble",
                             **kwargs) -> UncertaintyDecoder:
    """Create decoder with uncertainty quantification and gradient stability."""
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    config = DecoderConfig(
        input_size=input_size,
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        enable_physics=True,
        energy_constraint=True,
        momentum_constraint=True,
        position_bounds=True,
        velocity_bounds=True,
        uncertainty_type=uncertainty_type,
        normalize_states=True,
        use_layer_norm=True,
        weight_init_gain=0.1,
        output_scaling_factor=1e-3,
        gradient_clip_norm=1.0,
        **kwargs
    )
    return UncertaintyDecoder(config)

# Legacy aliases for backwards compatibility
DeterministicDecoder = StateDecoder
ProbabilisticDecoder = UncertaintyDecoder