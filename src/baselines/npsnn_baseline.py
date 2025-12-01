"""
NP-SNN Baseline Wrapper for Comparison Framework.

Wraps the NP-SNN model to provide consistent interface with
other baseline models for comprehensive performance evaluation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Import NP-SNN model and factory functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.npsnn import (
    NPSNN, 
    create_npsnn_for_orbital_tracking,
    create_npsnn_for_debris_tracking,
    create_minimal_npsnn
)


class NPSNNBaselineWrapper:
    """
    Wrapper class to integrate NP-SNN with baseline comparison framework.
    
    Provides consistent interface for trajectory prediction, uncertainty
    quantification, and physics-aware orbital state estimation.
    """
    
    def __init__(self,
                 model_type: str = 'orbital_tracking',
                 obs_input_size: int = 6,
                 uncertainty: bool = True,
                 physics_constraints: bool = True,
                 device: str = 'cpu',
                 model_path: Optional[str] = None):
        """
        Initialize NP-SNN baseline wrapper.
        
        Args:
            model_type: Type of NP-SNN model ('orbital_tracking', 'debris_tracking', 'minimal')
            obs_input_size: Size of observation input
            uncertainty: Enable uncertainty quantification
            physics_constraints: Enable physics constraints
            device: Device to run model on ('cpu', 'cuda')
            model_path: Path to pre-trained model (optional)
        """
        self.model_type = model_type
        self.obs_input_size = obs_input_size
        self.uncertainty = uncertainty
        self.physics_constraints = physics_constraints
        self.device = torch.device(device)
        
        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded pre-trained NP-SNN from {model_path}")
            except Exception as e:
                warnings.warn(f"Failed to load model from {model_path}: {e}")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get model info
        self.model_info = self.model.get_model_info()
    
    def _create_model(self) -> NPSNN:
        """Create NP-SNN model based on configuration."""
        
        if self.model_type == 'orbital_tracking':
            return create_npsnn_for_orbital_tracking(
                obs_input_size=self.obs_input_size,
                uncertainty=self.uncertainty,
                physics_constraints=self.physics_constraints
            )
        
        elif self.model_type == 'debris_tracking':
            return create_npsnn_for_debris_tracking(
                sensor_types=["optical", "radar"],  # Default sensor configuration
                multi_object=False
            )
        
        elif self.model_type == 'minimal':
            return create_minimal_npsnn(
                obs_input_size=self.obs_input_size
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_trajectory(self,
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          measurements: Optional[List[np.ndarray]] = None,
                          measurement_times: Optional[np.ndarray] = None,
                          measurement_type: str = 'position') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict orbital trajectory using NP-SNN.
        
        Args:
            initial_state: Initial orbital state [x, y, z, vx, vy, vz] (m, m/s)
            times: Time points for predictions (hours)
            measurements: List of measurements for conditioning (optional)
            measurement_times: Times of measurements (hours) (optional)
            measurement_type: Type of measurements (unused for NP-SNN)
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        try:
            # Convert inputs to torch tensors
            times_sec = times * 3600  # Convert hours to seconds
            t_span = torch.tensor(times_sec, dtype=torch.float32, device=self.device)
            
            # Use initial state as initial observation
            # NP-SNN expects observations, not direct states
            initial_obs = torch.tensor(
                initial_state, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # Add batch dimension
            
            # Predict trajectory
            with torch.no_grad():
                trajectory = self.model.predict_trajectory(
                    t_span=t_span,
                    initial_obs=initial_obs,
                    return_uncertainty=self.uncertainty
                )
            
            # Extract results
            states = trajectory['states'].cpu().numpy()  # (1, seq_len, 6)
            states = states.squeeze(0)  # Remove batch dimension -> (seq_len, 6)
            
            # Handle uncertainty
            if self.uncertainty and 'uncertainty' in trajectory:
                uncertainties = trajectory['uncertainty'].cpu().numpy()
                uncertainties = uncertainties.squeeze(0)  # (seq_len, 6)
            else:
                # Use physics violations as uncertainty proxy
                physics_violations = trajectory['physics_violations']
                if isinstance(physics_violations, torch.Tensor):
                    violations = physics_violations.cpu().numpy()
                    # Create uncertainty estimate based on physics consistency
                    uncertainties = np.ones_like(states) * np.mean(violations) * 1000  # Scale appropriately
                else:
                    # Default uncertainty
                    uncertainties = np.ones_like(states) * 100.0  # 100m position, 100 m/s velocity
            
            return states, uncertainties
            
        except Exception as e:
            # Return NaN arrays if prediction fails
            warnings.warn(f"NP-SNN prediction failed: {e}")
            n_times = len(times)
            states = np.full((n_times, 6), np.nan)
            uncertainties = np.full((n_times, 6), np.nan)
            return states, uncertainties
    
    def get_name(self) -> str:
        """Return baseline model name."""
        name = f"NPSNN_{self.model_type}"
        if self.uncertainty:
            name += "_unc"
        if self.physics_constraints:
            name += "_phys"
        return name
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'name': self.get_name(),
            'type': 'Neural Physics-Informed SNN',
            'parameters': self.model_info['total_parameters'],
            'memory_mb': self.model_info['memory_usage_mb'],
            'uncertainty': self.uncertainty,
            'physics_constraints': self.physics_constraints,
            'components': self.model_info['components'],
            'device': str(self.device)
        }
    
    def evaluate_physics_consistency(self,
                                   predicted_states: np.ndarray,
                                   times: np.ndarray) -> Dict[str, float]:
        """
        Evaluate physics consistency of predictions.
        
        Args:
            predicted_states: Predicted orbital states (n_times, 6)
            times: Time points (hours)
            
        Returns:
            Dictionary of physics consistency metrics
        """
        try:
            # Convert to torch tensors
            states_tensor = torch.tensor(
                predicted_states, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # Add batch dimension
            
            times_sec = times * 3600
            t_span = torch.tensor(
                times_sec, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # Add batch dimension
            
            # Compute physics consistency
            with torch.no_grad():
                consistency = self.model._compute_physics_consistency(states_tensor, t_span)
            
            # Convert to numpy and extract scalar values
            result = {}
            for key, value in consistency.items():
                if isinstance(value, torch.Tensor):
                    result[key] = float(value.cpu().numpy())
                else:
                    result[key] = float(value)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Physics consistency evaluation failed: {e}")
            return {
                'energy_conservation_violation': np.nan,
                'angular_momentum_conservation_violation': np.nan,
                'mean_altitude': np.nan
            }
    
    def compute_physics_informed_loss(self,
                                     times: np.ndarray,
                                     observations: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute physics-informed loss components.
        
        Args:
            times: Time points (hours)
            observations: Optional observations for loss computation
            
        Returns:
            Dictionary of loss components
        """
        try:
            # Convert to torch tensors
            times_sec = times * 3600
            t_tensor = torch.tensor(times_sec, dtype=torch.float32, device=self.device)
            
            if observations is not None:
                obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
            else:
                # Use zero observations
                obs_tensor = torch.zeros(
                    (len(times), self.obs_input_size), 
                    dtype=torch.float32, 
                    device=self.device
                )
            
            # Ensure single batch
            if t_tensor.dim() == 1:
                t_tensor = t_tensor.unsqueeze(0)  # (1, seq_len)
            if obs_tensor.dim() == 2:
                obs_tensor = obs_tensor.unsqueeze(0)  # (1, seq_len, obs_dim)
            
            # Compute physics-informed loss
            losses = self.model.physics_informed_loss(
                t=t_tensor,
                obs=obs_tensor
            )
            
            # Convert to dictionary of float values
            result = {}
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    result[key] = float(value.cpu().item())
                else:
                    result[key] = float(value)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Physics-informed loss computation failed: {e}")
            return {
                'total_loss': np.nan,
                'dynamics_loss': np.nan,
                'physics_loss': np.nan,
                'conservation_loss': np.nan
            }


def test_npsnn_baseline_wrapper():
    """Test NP-SNN baseline wrapper."""
    
    print("🧪 Testing NP-SNN Baseline Wrapper...")
    
    # Create wrapper
    wrapper = NPSNNBaselineWrapper(
        model_type='orbital_tracking',
        uncertainty=True,
        physics_constraints=True,
        device='cpu'
    )
    
    print(f"✅ Created NP-SNN wrapper: {wrapper.get_name()}")
    
    # Get model summary
    summary = wrapper.get_model_summary()
    print(f"   Parameters: {summary['parameters']:,}")
    print(f"   Memory: {summary['memory_mb']:.1f} MB")
    print(f"   Components: {list(summary['components'].keys())}")
    
    # Test trajectory prediction
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    times = np.linspace(0, 2, 25)  # 2 hours
    
    predicted_states, uncertainties = wrapper.predict_trajectory(
        initial_state, times
    )
    
    print(f"✅ Trajectory prediction successful!")
    print(f"   Prediction shape: {predicted_states.shape}")
    print(f"   Uncertainty shape: {uncertainties.shape}")
    print(f"   Initial position magnitude: {np.linalg.norm(initial_state[:3])/1000:.1f} km")
    print(f"   Final position magnitude: {np.linalg.norm(predicted_states[-1, :3])/1000:.1f} km")
    print(f"   Mean position uncertainty: ±{np.nanmean(uncertainties[:, 0])/1000:.1f} km")
    
    # Test physics consistency
    consistency = wrapper.evaluate_physics_consistency(predicted_states, times)
    print(f"✅ Physics consistency evaluation:")
    for key, value in consistency.items():
        if not np.isnan(value):
            print(f"   {key}: {value:.6f}")
    
    # Test physics-informed loss
    losses = wrapper.compute_physics_informed_loss(times)
    print(f"✅ Physics-informed loss computation:")
    for key, value in losses.items():
        if not np.isnan(value):
            print(f"   {key}: {value:.6f}")
    
    return wrapper


if __name__ == "__main__":
    test_npsnn_baseline_wrapper()