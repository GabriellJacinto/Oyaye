"""
Baseline models for NP-SNN comparison.

This module implements classical orbital propagation and filtering methods
for scientific validation of the NP-SNN approach, as per implementation 
plan section 4.6.

Baselines included:
- SGP4: Standard orbital propagation from TLE elements
- EKF: Extended Kalman Filter with 2-body + J2 dynamics
- UKF: Unscented Kalman Filter for nonlinearity handling  
- MLP: Pure neural network without spiking dynamics
- Particle Filter: Non-Gaussian baseline (optional)
"""

from .sgp4_baseline import SGP4Baseline
from .ekf_baseline import EKFBaseline
from .ukf_baseline import UKFBaseline
from .mlp_baseline import MLPBaseline
from .particle_filter_baseline import ParticleFilterBaseline
from .npsnn_baseline import NPSNNBaselineWrapper
from .baseline_evaluator import BaselineEvaluator

__all__ = [
    'SGP4Baseline',
    'EKFBaseline', 
    'UKFBaseline',
    'MLPBaseline',
    'ParticleFilterBaseline',
    'NPSNNBaselineWrapper',
    'BaselineEvaluator'
]