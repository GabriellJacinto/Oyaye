"""
Spiking Neural Network core implementation for Neural Physics-Informed SNNs.

This module provides:
- LIF and RLIF neuron layers with proper surrogate gradients
- Feedforward and recurrent SNN architectures optimized for continuous dynamics
- Membrane potential normalization and stability for orbital mechanics
- Time-adaptive integration for varying temporal scales
- Physics-compatible gradient flow

Key innovations for orbital mechanics:
- Multi-timescale LIF dynamics (fast spikes + slow adaptation)
- Continuous membrane potential evolution
- Gradient-friendly surrogate functions
- Memory-efficient recurrent processing
"""

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
    import torch.nn.functional as F
except ImportError:
    # Handle missing dependencies gracefully
    print("Warning: PyTorch and snnTorch not available. Please install dependencies.")
    torch = nn = snn = surrogate = F = None

import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SNNConfig:
    """Configuration for SNN architectures."""
    # Architecture
    input_size: int = 64              # Input feature dimension
    hidden_sizes: List[int] = None    # Hidden layer sizes
    output_size: int = 128            # Output feature dimension
    
    # Neuron dynamics
    beta: float = 0.9                 # Membrane decay constant
    threshold: float = 1.0            # Spike threshold
    spike_grad: str = "fast_sigmoid"  # Surrogate gradient function
    reset_mechanism: str = "subtract" # "subtract" or "zero"
    
    # Architecture features
    recurrent: bool = True            # Enable recurrent connections
    residual: bool = True             # Enable residual connections
    batch_norm: bool = False          # Batch normalization (careful with SNN dynamics)
    dropout: float = 0.1              # Dropout rate
    
    # Dynamics control
    adaptive_beta: bool = False       # Learnable decay constants
    multi_timescale: bool = True      # Multiple timescale dynamics
    membrane_clip: float = 5.0        # Clip membrane potentials for stability
    
    # Initialization
    init_method: str = "xavier"       # "xavier", "kaiming", "normal"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 128]

class AdaptiveLIFLayer(nn.Module):
    """
    Enhanced Leaky Integrate-and-Fire neuron layer for continuous dynamics.
    
    Features:
    - Adaptive decay constants (learnable beta)
    - Multi-timescale dynamics for orbital mechanics
    - Gradient clipping for stability
    - Recurrent connections for temporal memory
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 beta: float = 0.9,
                 threshold: float = 1.0,
                 spike_grad: str = "fast_sigmoid",
                 reset_mechanism: str = "subtract",
                 adaptive_beta: bool = False,
                 recurrent: bool = False,
                 dropout: float = 0.0,
                 membrane_clip: float = 5.0):
        super().__init__()
        
        if torch is None:
            raise ImportError("PyTorch not available")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.membrane_clip = membrane_clip
        
        # Decay constant (beta)
        if adaptive_beta:
            # Learnable decay with proper initialization
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))
        
        # Linear transformations
        self.fc_input = nn.Linear(input_size, hidden_size)
        
        # Recurrent connections
        self.recurrent = recurrent
        if recurrent:
            self.fc_recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Surrogate gradient function
        spike_grad_fn = self._get_spike_grad_fn(spike_grad)
        
        # Handle adaptive beta
        if adaptive_beta:
            # Create learnable parameter for beta
            self.beta_param = nn.Parameter(torch.tensor(beta))
            self.lif = snn.Leaky(
                beta=beta,  # Initial value, will be updated by parameter
                threshold=threshold,
                spike_grad=spike_grad_fn,
                init_hidden=False,  # We'll manage hidden state manually
                reset_mechanism=reset_mechanism,
                learn_beta=True
            )
        else:
            self.beta_param = None
            self.lif = snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad_fn,
                init_hidden=False,  # We'll manage hidden state manually
                reset_mechanism=reset_mechanism
            )
        
        # Initialize weights
        self._init_weights()
    
    def _get_spike_grad_fn(self, spike_grad: str):
        """Get surrogate gradient function by name."""
        if spike_grad == "fast_sigmoid":
            return surrogate.fast_sigmoid(slope=25)
        elif spike_grad == "straight_through":
            return surrogate.straight_through_estimator()
        elif spike_grad == "atan":
            return surrogate.atan()
        elif spike_grad == "sigmoid":
            return surrogate.sigmoid()
        else:
            return surrogate.fast_sigmoid(slope=25)  # Default
    
    def _init_weights(self):
        """Initialize weights with proper scaling for SNN dynamics."""
        # Input weights: He initialization for ReLU-like activation
        nn.init.kaiming_uniform_(self.fc_input.weight, mode='fan_in', nonlinearity='relu')
        if self.fc_input.bias is not None:
            nn.init.zeros_(self.fc_input.bias)
        
        # Recurrent weights: smaller initialization for stability
        if self.recurrent:
            nn.init.orthogonal_(self.fc_recurrent.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, 
                spk_rec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF layer.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            mem: Previous membrane potential (batch, hidden_size)
            spk_rec: Previous spike output for recurrent connections
        Returns:
            spikes: Output spikes (batch, hidden_size)
            mem: Updated membrane potential (batch, hidden_size)
        """
        # Apply input transformation
        current_input = self.fc_input(x)
        
        # Add recurrent input
        if self.recurrent and spk_rec is not None:
            current_input = current_input + self.fc_recurrent(spk_rec)
        
        # Apply dropout to input current
        if self.dropout is not None and self.training:
            current_input = self.dropout(current_input)
        
        # LIF dynamics with adaptive beta
        if hasattr(self, 'beta') and isinstance(self.beta, nn.Parameter):
            # Use learnable beta (need to update LIF manually)
            if mem is None:
                mem = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
            
            # Manual LIF update with adaptive beta
            mem = self.beta * mem + current_input
            
            # Clip membrane potential for stability
            mem = torch.clamp(mem, -self.membrane_clip, self.membrane_clip)
            
            # Generate spikes
            spikes = (mem >= self.threshold).float()
            
            # Apply surrogate gradient
            spikes = self.lif.spike_grad(mem - self.threshold) * spikes.detach() + spikes - spikes.detach()
            
            # Reset mechanism
            if self.reset_mechanism == "subtract":
                mem = mem - self.threshold * spikes
            elif self.reset_mechanism == "zero":
                mem = mem * (1 - spikes)
        else:
            # Use snnTorch LIF with fixed beta
            if mem is None:
                mem = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
            
            spikes, mem = self.lif(current_input, mem)
            
            # Clip for stability
            mem = torch.clamp(mem, -self.membrane_clip, self.membrane_clip)
        
        return spikes, mem

class MultiTimescaleLIFLayer(nn.Module):
    """
    Multi-timescale LIF layer for capturing different temporal dynamics.
    
    Combines fast (spike-generating) and slow (adaptation) timescales
    relevant to orbital mechanics where we have:
    - Fast dynamics: sensor measurements, short-term predictions
    - Slow dynamics: orbital evolution, long-term trends
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 beta_fast: float = 0.9,   # Fast timescale (spikes)
                 beta_slow: float = 0.99,  # Slow timescale (adaptation)
                 **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Fast LIF for spike generation
        self.lif_fast = AdaptiveLIFLayer(
            input_size, hidden_size, 
            beta=beta_fast, **kwargs
        )
        
        # Slow adaptation term
        self.adaptation = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_slow = beta_slow
        
        # Gating mechanism
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with multi-timescale dynamics.
        
        Args:
            x: Input tensor
            state: Dictionary with 'mem_fast', 'mem_slow', 'spk_rec'
        Returns:
            spikes: Output spikes
            new_state: Updated state dictionary
        """
        if state is None:
            state = {
                'mem_fast': None,
                'mem_slow': torch.zeros(x.size(0), self.hidden_size, device=x.device),
                'spk_rec': None
            }
        
        # Fast LIF dynamics
        spikes, mem_fast = self.lif_fast(x, state['mem_fast'], state['spk_rec'])
        
        # Slow adaptation
        adaptation_input = self.adaptation(spikes)
        mem_slow = self.beta_slow * state['mem_slow'] + (1 - self.beta_slow) * adaptation_input
        
        # Gating based on input and slow state
        gate_input = torch.cat([x, mem_slow], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        
        # Modulate spikes with slow adaptation
        modulated_spikes = spikes * gate_values
        
        new_state = {
            'mem_fast': mem_fast,
            'mem_slow': mem_slow,
            'spk_rec': modulated_spikes
        }
        
        return modulated_spikes, new_state

class SNNCore(nn.Module):
    """
    Multi-layer Spiking Neural Network optimized for continuous dynamics.
    
    Features:
    - Configurable architecture with residual connections
    - Multi-timescale dynamics for orbital mechanics
    - Gradient clipping and membrane potential normalization
    - Memory-efficient recurrent processing
    - Batch normalization compatible with SNN dynamics
    """
    
    def __init__(self, config: SNNConfig):
        super().__init__()
        
        if torch is None:
            raise ImportError("PyTorch not available")
            
        self.config = config
        self.input_size = config.input_size
        self.hidden_sizes = config.hidden_sizes
        self.output_size = config.output_size
        self.num_layers = len(config.hidden_sizes)
        
        # Build SNN layers
        self.layers = nn.ModuleList()
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(config.hidden_sizes):
            if config.multi_timescale and i == 0:
                # Use multi-timescale for first layer (input processing)
                layer = MultiTimescaleLIFLayer(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    beta_fast=config.beta,
                    beta_slow=0.99,
                    threshold=config.threshold,
                    spike_grad=config.spike_grad,
                    reset_mechanism=config.reset_mechanism,
                    adaptive_beta=config.adaptive_beta,
                    recurrent=config.recurrent,
                    dropout=config.dropout,
                    membrane_clip=config.membrane_clip
                )
            else:
                # Standard adaptive LIF layers
                layer = AdaptiveLIFLayer(
                    input_size=prev_size,
                    hidden_size=hidden_size,
                    beta=config.beta,
                    threshold=config.threshold,
                    spike_grad=config.spike_grad,
                    reset_mechanism=config.reset_mechanism,
                    adaptive_beta=config.adaptive_beta,
                    recurrent=config.recurrent,
                    dropout=config.dropout,
                    membrane_clip=config.membrane_clip
                )
            
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output projection
        self.output_projection = nn.Linear(prev_size, config.output_size)
        
        # Residual connections
        self.residual = config.residual
        if config.residual:
            self.residual_projections = nn.ModuleList()
            for hidden_size in config.hidden_sizes:
                if self.input_size != hidden_size:
                    self.residual_projections.append(nn.Linear(self.input_size, hidden_size))
                else:
                    self.residual_projections.append(nn.Identity())
        
        # Batch normalization (optional, careful with SNN dynamics)
        self.batch_norm = config.batch_norm
        if config.batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_size, momentum=0.1, track_running_stats=False)
                for hidden_size in config.hidden_sizes
            ])
        
        # Gradient clipping
        self.gradient_clip_val = 1.0
        
        # Initialize output projection
        self._init_output_projection()
    
    def _init_output_projection(self):
        """Initialize output projection layer."""
        nn.init.kaiming_uniform_(self.output_projection.weight, mode='fan_in', nonlinearity='relu')
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """
        Forward pass through multi-layer SNN.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            hidden_states: Previous hidden states for each layer
        Returns:
            output: Network output (batch, output_size)
            new_hidden_states: Updated hidden states
        """
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        new_hidden_states = []
        layer_input = x
        
        # Forward through SNN layers
        for i, layer in enumerate(self.layers):
            # Get layer output
            if isinstance(layer, MultiTimescaleLIFLayer):
                # MultiTimescaleLIFLayer expects dict state, not tuple
                layer_state = hidden_states[i] if (hidden_states[i] is not None and isinstance(hidden_states[i], dict)) else None
                spikes, new_state = layer(layer_input, layer_state)
            else:
                # Extract membrane potential and spike history from hidden state
                if hidden_states[i] is not None and isinstance(hidden_states[i], (tuple, list)):
                    mem, spk_rec = hidden_states[i]
                else:
                    mem, spk_rec = None, None
                
                spikes, mem = layer(layer_input, mem, spk_rec)
                new_state = (mem, spikes)
            
            new_hidden_states.append(new_state)
            
            # Apply batch normalization (optional)
            if self.batch_norm and hasattr(self, 'bn_layers'):
                spikes = self.bn_layers[i](spikes)
            
            # Residual connection
            if self.residual and i == 0:
                # Only for first layer (input to first hidden)
                residual = self.residual_projections[i](x)
                spikes = spikes + residual
            
            # Prepare input for next layer
            layer_input = spikes
        
        # Output projection
        output = self.output_projection(layer_input)
        
        return output, new_hidden_states
    
    def init_hidden_states(self, batch_size: int, device: torch.device) -> List:
        """Initialize hidden states for all layers."""
        hidden_states = []
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiTimescaleLIFLayer):
                # Multi-timescale layer state
                state = {
                    'mem_fast': torch.zeros(batch_size, layer.hidden_size, device=device),
                    'mem_slow': torch.zeros(batch_size, layer.hidden_size, device=device),
                    'spk_rec': torch.zeros(batch_size, layer.hidden_size, device=device)
                }
            else:
                # Standard LIF layer state (membrane potential, spike record)
                mem = torch.zeros(batch_size, layer.hidden_size, device=device)
                spk_rec = torch.zeros(batch_size, layer.hidden_size, device=device)
                state = (mem, spk_rec)
            
            hidden_states.append(state)
        
        return hidden_states
    
    def get_spike_rates(self, spikes: torch.Tensor, window_size: int = 100) -> torch.Tensor:
        """Calculate spike rates over a time window."""
        # spikes: (time, batch, features)
        if len(spikes.shape) == 2:
            spikes = spikes.unsqueeze(0)  # Add time dimension
        
        # Moving average over time window
        if spikes.size(0) < window_size:
            return spikes.mean(dim=0)
        else:
            # Use unfold to create sliding windows
            windowed = spikes.unfold(0, window_size, 1)  # (time-window+1, batch, features, window)
            return windowed.mean(dim=-1).mean(dim=0)  # Average over window and time
    
    def reset_hidden_states(self, hidden_states: List) -> List:
        """Reset hidden states to zero."""
        reset_states = []
        
        for i, state in enumerate(hidden_states):
            if isinstance(state, dict):
                # Multi-timescale layer
                reset_state = {k: torch.zeros_like(v) for k, v in state.items()}
            else:
                # Standard layer (tuple of tensors)
                reset_state = tuple(torch.zeros_like(s) for s in state)
            
            reset_states.append(reset_state)
        
        return reset_states
    
    def get_membrane_stats(self, hidden_states: List) -> Dict:
        """Get statistics about membrane potentials for monitoring."""
        stats = {}
        
        for i, state in enumerate(hidden_states):
            if isinstance(state, dict):
                # Multi-timescale layer
                mem_fast = state['mem_fast']
                mem_slow = state['mem_slow']
                
                stats[f'layer_{i}_mem_fast_mean'] = mem_fast.mean().item()
                stats[f'layer_{i}_mem_fast_std'] = mem_fast.std().item()
                stats[f'layer_{i}_mem_slow_mean'] = mem_slow.mean().item()
                stats[f'layer_{i}_mem_slow_std'] = mem_slow.std().item()
            else:
                # Standard layer
                mem, spikes = state
                
                stats[f'layer_{i}_mem_mean'] = mem.mean().item()
                stats[f'layer_{i}_mem_std'] = mem.std().item()
                stats[f'layer_{i}_spike_rate'] = spikes.mean().item()
        
        return stats


# Factory functions for easy creation
def create_standard_snn(input_size: int = 64, 
                       hidden_sizes: List[int] = None,
                       output_size: int = 128,
                       **kwargs) -> SNNCore:
    """Create standard SNN with sensible defaults."""
    if hidden_sizes is None:
        hidden_sizes = [128, 128]
    
    config = SNNConfig(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        **kwargs
    )
    return SNNCore(config)

def create_physics_snn(input_size: int = 64,
                      hidden_sizes: List[int] = None,
                      output_size: int = 128) -> SNNCore:
    """Create SNN optimized for physics-informed learning."""
    if hidden_sizes is None:
        hidden_sizes = [128, 256, 128]  # Wider middle layer
    
    config = SNNConfig(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        beta=0.95,                    # Longer memory for physics
        threshold=1.0,
        spike_grad="fast_sigmoid",
        recurrent=True,               # Temporal memory
        residual=True,                # Gradient flow
        multi_timescale=True,         # Multi-scale dynamics
        adaptive_beta=True,           # Learnable dynamics
        dropout=0.05,                 # Light regularization
        membrane_clip=10.0            # Higher clip for physics
    )
    return SNNCore(config)