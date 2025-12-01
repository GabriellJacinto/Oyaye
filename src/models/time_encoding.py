"""
Time encoding strategies for continuous-time neural networks.

This module provides:
- Fourier feature encodings for periodic phenomena
- Learned temporal embeddings  
- Multi-scale time representations
- Configuration-based encoding strategy selection

Key components for Neural Physics-Informed SNNs:
- FourierTimeEncoding: Multi-scale Fourier features for orbital periods
- LearnedTimeEncoding: Adaptive MLP-based temporal representations
- PositionalEncoding: Transformer-style continuous positional encoding
- TimeEncodingFactory: Configuration-based encoder creation
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass  
class TimeEncodingConfig:
    """Configuration for time encoding strategies."""
    encoding_type: str = "fourier"  # "fourier", "learned", "positional", "mixed"
    d_model: int = 64              # Output feature dimension
    max_frequency: float = 1e-2     # Maximum frequency for Fourier features (Hz)
    num_frequencies: int = 16       # Number of Fourier frequency components
    hidden_dim: int = 128          # Hidden dimension for learned encodings
    num_layers: int = 2            # Number of layers for learned encodings
    time_scale: float = 3600.0      # Time normalization scale (seconds -> hours)
    learnable_freqs: bool = False   # Whether Fourier frequencies are learnable
    dropout: float = 0.0           # Dropout rate for learned encodings

class FourierTimeEncoding(nn.Module):
    """
    Fourier feature-based time encoding for multi-scale temporal patterns.
    
    Maps time t to features: [sin(w1*t), cos(w1*t), sin(w2*t), cos(w2*t), ...]
    where frequencies w_i capture different temporal scales relevant to orbital mechanics:
    - Fast scales: ~minutes (satellite passes, sensor measurements)
    - Medium scales: ~hours (orbital periods)
    - Slow scales: ~days (precession, long-term dynamics)
    """
    
    def __init__(self, 
                 d_model: int = 64, 
                 max_frequency: float = 1e-2,  # ~100 second periods
                 time_scale: float = 3600.0,   # Normalize to hours
                 learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_frequency = max_frequency
        self.time_scale = time_scale
        
        # Generate frequency components in log space for multi-scale coverage
        num_freqs = d_model // 2
        log_freqs = torch.linspace(
            np.log(1e-4),  # Very slow: ~2.8 hour periods (better gradient flow)
            np.log(max_frequency),  # Fast: defined by max_frequency
            num_freqs
        )
        freqs = torch.exp(log_freqs)
        
        if learnable:
            self.register_parameter('freqs', nn.Parameter(freqs))
        else:
            self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier time encoding.
        
        Args:
            t: Time tensor of shape (...,) in seconds
        Returns:
            Encoded time features of shape (..., d_model)
        """
        # Normalize time to reasonable scale (seconds -> hours)
        # Add small epsilon to prevent zero gradients at t=0
        t_norm = (t + 1e-8) / self.time_scale
        
        # Expand for broadcasting: (..., 1) * (num_freq,) -> (..., num_freq)
        t_expanded = t_norm.unsqueeze(-1)  # (..., 1)
        
        # Compute phases: 2π * frequency * time
        phases = 2 * math.pi * t_expanded * self.freqs  # (..., num_freq)
        
        # Generate sin/cos features
        sin_features = torch.sin(phases)  # (..., num_freq)
        cos_features = torch.cos(phases)  # (..., num_freq)
        
        # Concatenate sin and cos features
        features = torch.cat([sin_features, cos_features], dim=-1)  # (..., d_model)
        
        return features
    
    def get_frequency_info(self) -> Dict:
        """Return information about the frequency components."""
        return {
            "num_frequencies": len(self.freqs),
            "min_frequency_hz": float(self.freqs.min()),
            "max_frequency_hz": float(self.freqs.max()),  
            "min_period_hours": float(1 / (self.freqs.max() * 3600)),
            "max_period_hours": float(1 / (self.freqs.min() * 3600)),
            "time_scale": self.time_scale
        }

class LearnedTimeEncoding(nn.Module):
    """
    Learned MLP-based time encoding with adaptive temporal representations.
    
    Uses multilayer perceptrons to learn optimal time representations from data.
    This allows the network to discover temporal patterns specific to the orbital
    dynamics problem that may not be captured by fixed Fourier features.
    """
    
    def __init__(self, 
                 d_model: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 time_scale: float = 3600.0,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.time_scale = time_scale
        
        # Build MLP encoder
        layers = []
        
        # Input layer: normalized time -> hidden
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
        # Output layer: hidden -> d_model
        layers.append(nn.Linear(hidden_dim, d_model))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with He/Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply learned time encoding.
        
        Args:
            t: Time tensor of shape (...,) in seconds
        Returns:
            Encoded time features of shape (..., d_model)
        """
        # Normalize time and reshape for MLP
        # Add small epsilon for gradient stability at t=0
        t_norm = (t + 1e-8) / self.time_scale  # Normalize to hours
        t_input = t_norm.unsqueeze(-1)  # (..., 1)
        
        # Pass through MLP encoder
        features = self.encoder(t_input)  # (..., d_model)
        
        return features

class PositionalEncoding(nn.Module):
    """
    Transformer-style positional encoding adapted for continuous time.
    
    Similar to standard transformer positional encoding but handles continuous
    time values rather than discrete positions. Uses alternating sin/cos
    patterns with different frequencies for each dimension.
    """
    
    def __init__(self, d_model: int = 64, time_scale: float = 3600.0, max_len: float = 1e6):
        super().__init__()
        self.d_model = d_model
        self.time_scale = time_scale
        
        # Create positional encoding matrix
        # Each dimension gets a different frequency
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(max_len) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            t: Time tensor of shape (...,) in seconds
        Returns:
            Encoded time features of shape (..., d_model)
        """
        # Normalize time
        t_norm = t / self.time_scale
        
        # Expand for broadcasting
        t_expanded = t_norm.unsqueeze(-1)  # (..., 1)
        
        # Create encoding
        encoding = torch.zeros(*t.shape, self.d_model, device=t.device, dtype=t.dtype)
        
        # Apply sin to even indices
        encoding[..., 0::2] = torch.sin(t_expanded * self.div_term)
        
        # Apply cos to odd indices  
        if self.d_model > 1:
            encoding[..., 1::2] = torch.cos(t_expanded * self.div_term)
        
        return encoding


class MixedTimeEncoding(nn.Module):
    """
    Mixed encoding combining Fourier features with learned representations.
    
    Combines the benefits of both approaches:
    - Fourier features for known periodicities (orbital mechanics)
    - Learned features for adaptive pattern discovery
    """
    
    def __init__(self,
                 d_model: int = 64,
                 fourier_dim: int = 32,
                 learned_dim: int = 32,
                 max_frequency: float = 1e-2,
                 time_scale: float = 3600.0,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        
        assert fourier_dim + learned_dim == d_model, \
            f"fourier_dim ({fourier_dim}) + learned_dim ({learned_dim}) must equal d_model ({d_model})"
        
        self.d_model = d_model
        self.fourier_dim = fourier_dim
        self.learned_dim = learned_dim
        
        # Fourier component
        self.fourier_encoder = FourierTimeEncoding(
            d_model=fourier_dim,
            max_frequency=max_frequency,
            time_scale=time_scale,
            learnable=False
        )
        
        # Learned component
        self.learned_encoder = LearnedTimeEncoding(
            d_model=learned_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_scale=time_scale,
            dropout=dropout
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed encoding.
        
        Args:
            t: Time tensor of shape (...,) in seconds
        Returns:
            Combined encoded features of shape (..., d_model)
        """
        fourier_features = self.fourier_encoder(t)  # (..., fourier_dim)
        learned_features = self.learned_encoder(t)  # (..., learned_dim)
        
        # Concatenate features
        mixed_features = torch.cat([fourier_features, learned_features], dim=-1)
        
        return mixed_features


class TimeEncodingFactory:
    """Factory for creating time encoders based on configuration."""
    
    @staticmethod
    def create_encoder(config: TimeEncodingConfig) -> nn.Module:
        """Create time encoder from configuration."""
        if config.encoding_type == "fourier":
            return FourierTimeEncoding(
                d_model=config.d_model,
                max_frequency=config.max_frequency,
                time_scale=config.time_scale,
                learnable=config.learnable_freqs
            )
        elif config.encoding_type == "learned":
            return LearnedTimeEncoding(
                d_model=config.d_model,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                time_scale=config.time_scale,
                dropout=config.dropout
            )
        elif config.encoding_type == "positional":
            return PositionalEncoding(
                d_model=config.d_model,
                time_scale=config.time_scale
            )
        elif config.encoding_type == "mixed":
            fourier_dim = config.d_model // 2
            learned_dim = config.d_model - fourier_dim
            return MixedTimeEncoding(
                d_model=config.d_model,
                fourier_dim=fourier_dim,
                learned_dim=learned_dim,
                max_frequency=config.max_frequency,
                time_scale=config.time_scale,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown encoding type: {config.encoding_type}")


# Convenience functions for quick usage
def create_fourier_encoder(d_model: int = 64, 
                          max_frequency: float = 1e-2,
                          time_scale: float = 3600.0) -> FourierTimeEncoding:
    """Create Fourier time encoder with orbital mechanics defaults."""
    return FourierTimeEncoding(
        d_model=d_model,
        max_frequency=max_frequency,
        time_scale=time_scale,
        learnable=False
    )

def create_learned_encoder(d_model: int = 64,
                          hidden_dim: int = 128,
                          time_scale: float = 3600.0) -> LearnedTimeEncoding:
    """Create learned time encoder with sensible defaults."""
    return LearnedTimeEncoding(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=2,
        time_scale=time_scale,
        dropout=0.1
    )

def create_mixed_encoder(d_model: int = 64,
                        max_frequency: float = 1e-2,
                        time_scale: float = 3600.0) -> MixedTimeEncoding:
    """Create mixed time encoder balancing Fourier and learned features."""
    fourier_dim = d_model // 2
    learned_dim = d_model - fourier_dim
    return MixedTimeEncoding(
        d_model=d_model,
        fourier_dim=fourier_dim,
        learned_dim=learned_dim,
        max_frequency=max_frequency,
        time_scale=time_scale,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    )