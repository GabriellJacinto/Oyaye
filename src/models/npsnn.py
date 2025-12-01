"""
Neural Physics-Informed Spiking Neural Network (NP-SNN) - Complete Model

This module provides the unified NP-SNN architecture integrating:
- Time encoding for continuous temporal dynamics
- Spiking neural network core with multi-timescale processing
- Physics-constrained decoders with uncertainty quantification
- End-to-end training capability with gradient stability
- Physics-informed loss integration

Key Features:
- Orbital mechanics-aware temporal encoding
- Multi-scale SNN dynamics for orbital periods (minutes to days)
- Physics constraints (energy conservation, angular momentum)
- Uncertainty quantification for state estimation
- Automatic differentiation for physics losses
- Robust gradient flow with proper normalization
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

# Conditional imports with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:
    torch = nn = F = Tensor = None
    warnings.warn("PyTorch not available. Please install dependencies.")

try:
    import numpy as np
except ImportError:
    np = None
    warnings.warn("NumPy not available.")

# Import our components
from .time_encoding import (
    FourierTimeEncoding, 
    LearnedTimeEncoding, 
    TimeEncodingConfig,
    TimeEncodingFactory
)
from .snn_core import (
    SNNCore, 
    AdaptiveLIFLayer,
    MultiTimescaleLIFLayer,
    SNNConfig
)
from .decoder import (
    StateDecoder,
    PhysicsConstrainedDecoder,
    UncertaintyDecoder,
    DecoderConfig,
    create_standard_decoder,
    create_physics_decoder,
    create_uncertainty_decoder
)

@dataclass
class NPSNNConfig:
    """Configuration for Neural Physics-Informed Spiking Neural Network."""
    # Time encoding configuration
    time_encoding: TimeEncodingConfig = None
    
    # Input observation encoding
    obs_input_size: int = 6           # Input observation dimension (e.g., sensor measurements)
    obs_encoding_dim: int = 32        # Encoded observation dimension
    
    # SNN core configuration
    snn: SNNConfig = None
    
    # Decoder configuration
    decoder: DecoderConfig = None
    
    # Model architecture
    enable_physics_constraints: bool = True    # Enable physics-aware constraints
    uncertainty_quantification: bool = True    # Enable uncertainty estimation
    temporal_consistency: bool = True          # Enable temporal consistency
    
    # Training configuration
    gradient_clip_norm: float = 1.0           # Global gradient clipping
    enable_auto_diff: bool = True             # Enable automatic differentiation for physics
    physics_loss_weight: float = 0.1          # Weight for physics-based losses
    
    # Numerical stability
    state_normalization: bool = True          # Normalize states for numerical stability
    adaptive_time_step: bool = False          # Use adaptive time stepping (future feature)
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.time_encoding is None:
            self.time_encoding = TimeEncodingConfig()
        if self.snn is None:
            self.snn = SNNConfig()
        if self.decoder is None:
            self.decoder = DecoderConfig()

class NPSNN(nn.Module):
    """
    Neural Physics-Informed Spiking Neural Network for orbital mechanics.
    
    Architecture Pipeline:
    1. Time encoding: Maps continuous time to neural features
    2. Observation encoding: Maps sensor measurements to neural features  
    3. SNN core: Processes spatio-temporal patterns with multi-timescale dynamics
    4. Decoder: Transforms SNN outputs to physics-constrained state estimates
    
    Key Capabilities:
    - End-to-end differentiable orbital state estimation
    - Physics-aware constraints (energy, angular momentum conservation)
    - Uncertainty quantification for robust predictions
    - Multi-timescale dynamics for orbital mechanics (minutes to days)
    - Automatic differentiation for physics-informed losses
    """
    
    def __init__(self, config: NPSNNConfig):
        super().__init__()
        
        if torch is None:
            raise ImportError("PyTorch not available")
            
        self.config = config
        
        # Time encoding component
        self.time_encoder = TimeEncodingFactory.create_encoder(config.time_encoding)
        time_dim = config.time_encoding.d_model
        
        # Observation encoding
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.obs_input_size, config.obs_encoding_dim),
            nn.LayerNorm(config.obs_encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Update SNN config with proper input size
        snn_input_size = time_dim + config.obs_encoding_dim
        snn_config = config.snn
        snn_config.input_size = snn_input_size
        
        # SNN core with multi-timescale dynamics
        self.snn_core = SNNCore(snn_config)
        snn_output_size = snn_config.output_size
        
        # Update decoder config with proper input size
        decoder_config = config.decoder
        decoder_config.input_size = snn_output_size
        
        # Update decoder config with proper input size
        decoder_config.input_size = snn_output_size
        
        # Decoder selection based on configuration
        if config.uncertainty_quantification:
            # Use uncertainty decoder with physics constraints
            decoder_config.enable_physics = config.enable_physics_constraints
            self.decoder = create_uncertainty_decoder(
                input_size=decoder_config.input_size,
                state_dim=decoder_config.state_dim,
                hidden_sizes=decoder_config.hidden_sizes,
                uncertainty_type=decoder_config.uncertainty_type
            )
            self.decoder_type = "uncertainty"
        elif config.enable_physics_constraints:
            # Use physics-constrained decoder  
            self.decoder = create_physics_decoder(
                input_size=decoder_config.input_size,
                state_dim=decoder_config.state_dim,
                hidden_sizes=decoder_config.hidden_sizes
            )
            self.decoder_type = "physics"
        else:
            # Use standard decoder
            self.decoder = create_standard_decoder(
                input_size=decoder_config.input_size,
                state_dim=decoder_config.state_dim,
                hidden_sizes=decoder_config.hidden_sizes
            )
            self.decoder_type = "standard"
        
        # Physics constants for constraint computation
        self.register_buffer('earth_mu', torch.tensor(3.986004418e14))  # Earth gravitational parameter
        self.register_buffer('earth_radius', torch.tensor(6.371e6))     # Earth radius
        
        # State normalization parameters (will be updated during training)
        if config.state_normalization:
            self.register_buffer('state_mean', torch.zeros(decoder_config.state_dim))
            self.register_buffer('state_std', torch.ones(decoder_config.state_dim))
            self._initialize_state_normalization()
        
        # Initialize weights for gradient stability
        self._initialize_weights()
    
    def _initialize_state_normalization(self):
        """Initialize state normalization with typical orbital values."""
        # Position: LEO altitude (400 km above Earth)
        pos_mean = self.config.decoder.earth_radius * 1.063  # ~6.77e6 m
        pos_std = self.config.decoder.earth_radius * 0.1     # ~6.37e5 m variation
        
        # Velocity: Typical LEO orbital velocity  
        vel_mean = 0.0      # Zero mean velocity
        vel_std = 7800.0    # ~7.8 km/s (LEO orbital velocity)
        
        # Set normalization parameters
        self.state_mean[:3] = pos_mean
        self.state_mean[3:] = vel_mean
        self.state_std[:3] = pos_std
        self.state_std[3:] = vel_std
    
    def _initialize_weights(self):
        """Initialize weights for gradient stability."""
        def init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.1  # Scale down for stability
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Apply to observation encoder
        self.obs_encoder.apply(init_module)
        
    def forward(self, 
                t: Tensor,
                obs: Optional[Tensor] = None,
                mem_states: Optional[List[Tensor]] = None,
                return_uncertainty: bool = None) -> Dict[str, Tensor]:
        """
        Forward pass of NP-SNN.
        
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, seq_len)
            obs: Observation tensor of shape (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
            mem_states: Previous SNN membrane states from previous time step
            return_uncertainty: Whether to return uncertainty estimates (if supported)
            
        Returns:
            outputs: Dictionary containing:
                - 'states': Predicted orbital states (batch, state_dim) or (batch, seq_len, state_dim)
                - 'uncertainty': Uncertainty estimates if enabled (same shape as states)
                - 'mem_states': Updated SNN membrane states for next time step
                - 'physics_violations': Physics constraint violations (for loss computation)
                - 'time_features': Encoded time features (for analysis)
                - 'snn_features': SNN output features (for analysis)
        """
        batch_size = t.shape[0]
        
        # Handle different input shapes
        if t.dim() == 1:
            # Single time point per batch element
            seq_len = 1
            t = t.unsqueeze(1)  # (batch_size, 1)
            single_timestep = True
        else:
            # Multiple time points per batch element
            seq_len = t.shape[1]
            single_timestep = False
        
        # Ensure observations match time dimensionality
        if obs is None:
            # Use zero observations if not provided
            obs = torch.zeros(batch_size, seq_len, self.config.obs_input_size, 
                            device=t.device, dtype=t.dtype)
        elif obs.dim() == 2 and seq_len > 1:
            # Broadcast observations across sequence
            obs = obs.unsqueeze(1).expand(-1, seq_len, -1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        # Initialize outputs
        all_states = []
        all_uncertainties = []
        all_physics_violations = []
        time_features_list = []
        snn_features_list = []
        
        # Process each time step
        current_mem_states = mem_states
        
        for step in range(seq_len):
            # Extract current time and observations
            t_current = t[:, step]  # (batch_size,)
            obs_current = obs[:, step, :]  # (batch_size, obs_dim)
            
            # 1. Time encoding
            time_features = self.time_encoder(t_current)  # (batch_size, time_dim)
            time_features_list.append(time_features)
            
            # 2. Observation encoding
            obs_features = self.obs_encoder(obs_current)  # (batch_size, obs_encoding_dim)
            
            # 3. Combine time and observation features
            snn_input = torch.cat([time_features, obs_features], dim=-1)  # (batch_size, snn_input_size)
            
            # 4. SNN core processing
            snn_output, current_mem_states = self.snn_core(snn_input, current_mem_states)
            snn_features_list.append(snn_output)
            
            # 5. Decode to orbital states
            if self.decoder_type == "uncertainty":
                decoder_output = self.decoder(snn_output)
                if isinstance(decoder_output, tuple):
                    states, uncertainty = decoder_output
                    all_uncertainties.append(uncertainty)
                else:
                    states = decoder_output
                    all_uncertainties.append(torch.zeros_like(states))
            else:
                states = self.decoder(snn_output)
                all_uncertainties.append(torch.zeros_like(states))
            
            all_states.append(states)
            
            # 6. Compute physics constraint violations (if physics decoder)
            if hasattr(self.decoder, 'constraint_losses'):
                physics_violations = self.decoder.constraint_losses(states)
            else:
                physics_violations = {}
            all_physics_violations.append(physics_violations)
        
        # Stack outputs across time dimension
        states = torch.stack(all_states, dim=1)  # (batch_size, seq_len, state_dim)
        uncertainties = torch.stack(all_uncertainties, dim=1)  # (batch_size, seq_len, state_dim)
        time_features_stacked = torch.stack(time_features_list, dim=1)  # (batch_size, seq_len, time_dim)
        snn_features_stacked = torch.stack(snn_features_list, dim=1)  # (batch_size, seq_len, snn_output_dim)
        
        # Aggregate physics violations across time
        aggregated_violations = {}
        if all_physics_violations and all_physics_violations[0]:
            for key in all_physics_violations[0].keys():
                violation_values = [v[key] for v in all_physics_violations if key in v]
                if violation_values:
                    aggregated_violations[key] = torch.stack(violation_values, dim=0).mean()
        
        # Remove sequence dimension if single timestep
        if single_timestep:
            states = states.squeeze(1)
            uncertainties = uncertainties.squeeze(1)
            time_features_stacked = time_features_stacked.squeeze(1)
            snn_features_stacked = snn_features_stacked.squeeze(1)
        
        # Normalize states if enabled
        if self.config.state_normalization:
            states = self.normalize_states(states)
        
        # Build output dictionary
        outputs = {
            'states': states,
            'mem_states': current_mem_states,
            'physics_violations': aggregated_violations,
            'time_features': time_features_stacked,
            'snn_features': snn_features_stacked
        }
        
        # Add uncertainty if requested and available
        if return_uncertainty is None:
            return_uncertainty = self.config.uncertainty_quantification
        
        if return_uncertainty:
            outputs['uncertainty'] = uncertainties
        
        return outputs
    
    def normalize_states(self, states: Tensor) -> Tensor:
        """Normalize states using running statistics."""
        if self.config.state_normalization:
            return (states - self.state_mean) / self.state_std
        return states
    
    def denormalize_states(self, states: Tensor) -> Tensor:
        """Denormalize states back to physical units."""
        if self.config.state_normalization:
            return states * self.state_std + self.state_mean
        return states
        
    def predict_trajectory(self,
                          t_span: Tensor,
                          initial_obs: Optional[Tensor] = None,
                          return_uncertainty: bool = True) -> Dict[str, Tensor]:
        """
        Predict orbital trajectory over time span with physics consistency.
        
        Args:
            t_span: Time points to evaluate at (seq_len,) or (batch_size, seq_len)
            initial_obs: Initial observations to condition on (batch_size, obs_dim)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            trajectory: Dictionary containing:
                - 'states': Predicted states (batch_size, seq_len, state_dim)
                - 'uncertainty': Uncertainty bounds if requested
                - 'physics_consistency': Physics constraint satisfaction metrics
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Handle input shapes
            if t_span.dim() == 1:
                # Single trajectory
                batch_size = 1 if initial_obs is None else initial_obs.shape[0]
                t_span = t_span.unsqueeze(0).expand(batch_size, -1)
            else:
                batch_size = t_span.shape[0]
            
            seq_len = t_span.shape[1]
            
            # Initialize observations if not provided
            if initial_obs is None:
                initial_obs = torch.zeros(batch_size, self.config.obs_input_size,
                                        device=t_span.device, dtype=t_span.dtype)
            
            # For trajectory prediction, we'll use the initial observation for all timesteps
            # In practice, this could be more sophisticated (e.g., integrating predicted states)
            obs_trajectory = initial_obs.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Forward pass through the model
            outputs = self.forward(
                t=t_span,
                obs=obs_trajectory,
                mem_states=None,
                return_uncertainty=return_uncertainty
            )
            
            states = outputs['states']
            physics_violations = outputs['physics_violations']
            
            # Compute physics consistency metrics
            consistency_metrics = self._compute_physics_consistency(states, t_span)
            
            result = {
                'states': states,
                'physics_consistency': consistency_metrics,
                'physics_violations': physics_violations,
                'time_features': outputs['time_features'],
                'snn_features': outputs['snn_features']
            }
            
            if return_uncertainty and 'uncertainty' in outputs:
                result['uncertainty'] = outputs['uncertainty']
            
            return result
    
    def _compute_physics_consistency(self, states: Tensor, t_span: Tensor) -> Dict[str, Tensor]:
        """Compute physics consistency metrics for trajectory."""
        # Extract positions and velocities
        positions = states[..., :3]  # (..., 3)
        velocities = states[..., 3:6]  # (..., 3)
        
        # Compute orbital energy: E = 0.5*v² - μ/r
        kinetic_energy = 0.5 * torch.sum(velocities**2, dim=-1)  # (...,)
        potential_energy = -self.earth_mu / torch.norm(positions, dim=-1)  # (...,)
        total_energy = kinetic_energy + potential_energy  # (...,)
        
        # Compute angular momentum: h = r × v
        angular_momentum = torch.cross(positions, velocities, dim=-1)  # (..., 3)
        angular_momentum_magnitude = torch.norm(angular_momentum, dim=-1)  # (...,)
        
        # Check position bounds (above Earth surface)
        distance_from_earth = torch.norm(positions, dim=-1)  # (...,)
        altitude = distance_from_earth - self.earth_radius
        
        # Compute consistency metrics
        consistency = {
            'energy_mean': total_energy.mean(),
            'energy_std': total_energy.std(),
            'angular_momentum_mean': angular_momentum_magnitude.mean(),
            'angular_momentum_std': angular_momentum_magnitude.std(),
            'min_altitude': altitude.min(),
            'mean_altitude': altitude.mean(),
            'energy_conservation_violation': total_energy.std() / (torch.abs(total_energy.mean()) + 1e-8),
            'angular_momentum_conservation_violation': angular_momentum_magnitude.std() / (angular_momentum_magnitude.mean() + 1e-8)
        }
        
        return consistency
    
    def get_derivatives(self, 
                       t: Tensor,
                       obs: Optional[Tensor] = None,
                       create_graph: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Compute time derivatives using automatic differentiation for physics-informed losses.
        
        Args:
            t: Time tensor with requires_grad=True
            obs: Optional observations
            create_graph: Whether to create computation graph for higher-order derivatives
            
        Returns:
            dr_dt: Position time derivative (velocity) 
            dv_dt: Velocity time derivative (acceleration)
        """
        if not self.config.enable_auto_diff:
            raise ValueError("Automatic differentiation not enabled in config")
        
        # Ensure t requires gradients
        if not t.requires_grad:
            t = t.detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.forward(t=t, obs=obs, return_uncertainty=False)
        states = outputs['states']
        
        # Extract position and velocity components
        positions = states[..., :3]  # (..., 3)
        velocities = states[..., 3:6]  # (..., 3)
        
        # Compute time derivatives using automatic differentiation
        # dr/dt should equal velocity (consistency check)
        dr_dt = []
        dv_dt = []
        
        for i in range(3):  # For each spatial dimension
            # Position derivative
            dr_dt_i = torch.autograd.grad(
                positions[..., i].sum(), t,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )[0]
            dr_dt.append(dr_dt_i)
            
            # Velocity derivative (acceleration)
            dv_dt_i = torch.autograd.grad(
                velocities[..., i].sum(), t,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )[0]
            dv_dt.append(dv_dt_i)
        
        # Stack derivatives
        dr_dt = torch.stack(dr_dt, dim=-1)  # (..., 3)
        dv_dt = torch.stack(dv_dt, dim=-1)  # (..., 3)
        
        return dr_dt, dv_dt
    
    def physics_informed_loss(self, 
                            t: Tensor, 
                            obs: Optional[Tensor] = None,
                            true_states: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Compute physics-informed loss components for training.
        
        Args:
            t: Time tensor with requires_grad=True
            obs: Optional observations
            true_states: Ground truth states for supervised loss
            
        Returns:
            losses: Dictionary of loss components
        """
        losses = {}
        
        # Forward pass
        outputs = self.forward(t=t, obs=obs, return_uncertainty=False)
        predicted_states = outputs['states']
        
        # Data reconstruction loss (if ground truth available)
        if true_states is not None:
            losses['reconstruction'] = F.mse_loss(predicted_states, true_states)
        
        # Physics constraint violations from decoder
        if outputs['physics_violations']:
            for key, violation in outputs['physics_violations'].items():
                losses[f'physics_{key}'] = violation
        
        # Physics-informed derivative consistency
        if self.config.enable_auto_diff and t.requires_grad:
            try:
                dr_dt, dv_dt = self.get_derivatives(t, obs, create_graph=True)
                predicted_velocities = predicted_states[..., 3:6]
                
                # Derivative consistency: dr/dt should equal predicted velocity
                losses['derivative_consistency'] = F.mse_loss(dr_dt, predicted_velocities)
            except RuntimeError as e:
                if "backward through the graph a second time" in str(e):
                    # Skip derivative consistency if graph already used
                    losses['derivative_consistency'] = torch.tensor(0.0, device=predicted_states.device)
                else:
                    raise
        
        # Temporal smoothness (if sequential data)
        if predicted_states.dim() > 2:  # (batch, seq_len, state_dim)
            state_diffs = predicted_states[:, 1:] - predicted_states[:, :-1]
            losses['temporal_smoothness'] = torch.mean(state_diffs**2)
        
        return losses
    
    def clip_gradients(self, max_norm: float = None) -> float:
        """Clip gradients across all model components."""
        if max_norm is None:
            max_norm = self.config.gradient_clip_norm
        
        # Clip gradients for the entire model
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        
        # Also clip decoder gradients specifically (they tend to explode)
        if hasattr(self.decoder, 'clip_gradients'):
            decoder_norm = self.decoder.clip_gradients(max_norm)
        
        return total_norm
        
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       epoch: int = 0, metrics: Optional[Dict] = None) -> None:
        """
        Save comprehensive model checkpoint with configuration and training state.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optimizer state dict (optional)
            epoch: Current training epoch
            metrics: Training metrics dictionary
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__,
            'epoch': epoch,
            'decoder_type': self.decoder_type,
        }
        
        # Add optional components
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Save normalization parameters if enabled
        if self.config.state_normalization:
            checkpoint['state_mean'] = self.state_mean
            checkpoint['state_std'] = self.state_std
        
        torch.save(checkpoint, filepath)
        
    @classmethod  
    def load_checkpoint(cls, filepath: str, device: Optional[str] = None) -> Tuple['NPSNN', Dict]:
        """
        Load model from checkpoint with full configuration restoration.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on ('cpu', 'cuda', etc.)
            
        Returns:
            model: Loaded NP-SNN model
            checkpoint: Full checkpoint dictionary with training state
        """
        if torch is None:
            raise ImportError("PyTorch not available")
        
        # Load checkpoint (disable weights_only for our trusted config objects)
        if device is None:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        else:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        # Restore configuration
        config = checkpoint['config']
        
        # Create model
        model = cls(config)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore normalization parameters if available
        if 'state_mean' in checkpoint and model.config.state_normalization:
            model.state_mean = checkpoint['state_mean']
            model.state_std = checkpoint['state_std']
        
        # Move to specified device
        if device is not None:
            model = model.to(device)
        
        return model, checkpoint
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Neural Physics-Informed Spiking Neural Network',
            'decoder_type': self.decoder_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config,
            'components': {
                'time_encoder': type(self.time_encoder).__name__,
                'snn_core': type(self.snn_core).__name__, 
                'decoder': type(self.decoder).__name__
            }
        }
        
        return info

# Factory Functions for Easy Model Creation

def create_npsnn_for_orbital_tracking(obs_input_size: int = 6,
                                      state_dim: int = 6,
                                      uncertainty: bool = True,
                                      physics_constraints: bool = True,
                                      **kwargs) -> NPSNN:
    """
    Create NP-SNN optimized for orbital tracking with gradient stability.
    
    Args:
        obs_input_size: Input observation dimension (sensor measurements)
        state_dim: Output state dimension (position + velocity)  
        uncertainty: Enable uncertainty quantification
        physics_constraints: Enable physics-aware constraints
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured NP-SNN model ready for training
    """
    # Time encoding: Multi-scale for orbital mechanics
    time_config = TimeEncodingConfig(
        encoding_type="fourier",
        d_model=64,
        max_frequency=1e-3,  # ~16 minute periods (fast orbital dynamics)
        num_frequencies=16,
        time_scale=3600.0,   # Hours normalization
        learnable_freqs=True
    )
    
    # SNN configuration: Multi-timescale dynamics
    snn_config = SNNConfig(
        input_size=64 + 32,  # Will be updated by NPSNN init
        hidden_sizes=[128, 64, 32],
        output_size=64,
        beta=0.9,
        threshold=1.0,
        spike_grad="fast_sigmoid",
        recurrent=True,
        residual=True,
        multi_timescale=True,
        adaptive_beta=True,
        dropout=0.1,
        membrane_clip=5.0
    )
    
    # Decoder configuration: Physics-aware with gradient stability
    decoder_config = DecoderConfig(
        input_size=64,  # Will be updated by NPSNN init
        state_dim=state_dim,
        hidden_sizes=[128, 64, 32],
        enable_physics=physics_constraints,
        energy_constraint=physics_constraints,
        momentum_constraint=physics_constraints,
        position_bounds=physics_constraints,
        velocity_bounds=physics_constraints,
        uncertainty_type="ensemble" if uncertainty else "none",
        num_ensemble=5 if uncertainty else 1,
        normalize_states=True,
        use_layer_norm=True,
        weight_init_gain=0.1,
        output_scaling_factor=1e-3,
        gradient_clip_norm=1.0
    )
    
    # Overall NP-SNN configuration
    config = NPSNNConfig(
        time_encoding=time_config,
        obs_input_size=obs_input_size,
        obs_encoding_dim=32,
        snn=snn_config,
        decoder=decoder_config,
        enable_physics_constraints=physics_constraints,
        uncertainty_quantification=uncertainty,
        temporal_consistency=True,
        gradient_clip_norm=1.0,
        enable_auto_diff=True,
        physics_loss_weight=0.1,
        state_normalization=True,
        **kwargs
    )
    
    return NPSNN(config)

def create_npsnn_for_debris_tracking(sensor_types: List[str] = ["optical", "radar"],
                                    multi_object: bool = False,  # Disable multi-object for now
                                    **kwargs) -> NPSNN:
    """
    Create NP-SNN specialized for space debris tracking scenarios.
    
    Args:
        sensor_types: List of sensor types ("optical", "radar", "tle")
        multi_object: Enable multi-object tracking capabilities (currently limited to single object)
        **kwargs: Additional configuration overrides
        
    Returns:
        Specialized NP-SNN for debris tracking
    """
    # Determine input size based on sensor types
    sensor_dims = {"optical": 2, "radar": 3, "tle": 6}  # RA/Dec, Range/Az/El, TLE elements
    obs_input_size = sum(sensor_dims.get(sensor, 0) for sensor in sensor_types)
    
    if multi_object:
        obs_input_size *= 2  # Support multiple objects
        # For multi-object, we need to modify physics constraints to handle multiple states
        # For now, keep single object (state_dim=6)
        print("Warning: Multi-object tracking not fully implemented. Using single object.")
    
    # Use base orbital tracking configuration
    model = create_npsnn_for_orbital_tracking(
        obs_input_size=obs_input_size,
        state_dim=6,  # Keep single object for now
        uncertainty=True,  # Critical for debris tracking
        physics_constraints=True,  # Essential for orbital mechanics
        **kwargs
    )
    
    return model

def create_minimal_npsnn(obs_input_size: int = 6, **kwargs) -> NPSNN:
    """Create minimal NP-SNN for testing and development."""
    time_config = TimeEncodingConfig(d_model=32, num_frequencies=8)
    snn_config = SNNConfig(hidden_sizes=[64, 32], output_size=32)
    decoder_config = DecoderConfig(
        state_dim=6, 
        hidden_sizes=[64, 32],
        enable_physics=False,
        uncertainty_type="none",
        weight_init_gain=0.1,
        output_scaling_factor=1e-3
    )
    
    config = NPSNNConfig(
        time_encoding=time_config,
        obs_input_size=obs_input_size,
        obs_encoding_dim=16,
        snn=snn_config,
        decoder=decoder_config,
        enable_physics_constraints=False,
        uncertainty_quantification=False,
        **kwargs
    )
    
    return NPSNN(config)