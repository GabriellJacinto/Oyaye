"""
Neural Physics-Informed Spiking Neural Network Models.

This package contains the core neural network components for the NP-SNN:
- Time encoding strategies for continuous temporal dynamics
- Spiking neural network core with LIF neurons  
- Decoder networks for state estimation
- Complete NP-SNN model integration
"""

# Time encoding
from .time_encoding import (
    FourierTimeEncoding,
    LearnedTimeEncoding, 
    PositionalEncoding,
    MixedTimeEncoding,
    TimeEncodingConfig,
    TimeEncodingFactory,
    create_fourier_encoder,
    create_learned_encoder,
    create_mixed_encoder
)

# Core model components (to be implemented)
# from .snn_core import SNNCore, LIFLayer, SpikingBlock  
# from .decoder import StateDecoder, UncertaintyDecoder
# from .npsnn import NeuralPhysicsInformedSNN

__all__ = [
    # Time encoding
    'FourierTimeEncoding',
    'LearnedTimeEncoding',
    'PositionalEncoding', 
    'MixedTimeEncoding',
    'TimeEncodingConfig',
    'TimeEncodingFactory',
    'create_fourier_encoder',
    'create_learned_encoder',
    'create_mixed_encoder',
    
    # Core components (coming next)
    # 'SNNCore',
    # 'LIFLayer', 
    # 'SpikingBlock',
    # 'StateDecoder',
    # 'UncertaintyDecoder',
    # 'NeuralPhysicsInformedSNN'
]