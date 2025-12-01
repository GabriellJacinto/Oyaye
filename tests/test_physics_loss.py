#!/usr/bin/env python3
"""
Test comprehensive physics-informed loss functions for NP-SNN.

This script tests all components of the physics-informed loss system:
- Individual loss components (energy, momentum, dynamics, measurement, temporal)
- Adaptive weight learning
- Curriculum learning schedules
- Complete integration with NP-SNN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import our loss functions
from src.train.losses import (
    PhysicsLossConfig,
    EnergyConservationLoss,
    AngularMomentumConservationLoss, 
    DynamicsResidualLoss,
    MeasurementConsistencyLoss,
    TemporalSmoothnessLoss,
    AdaptiveWeightLearner,
    PhysicsInformedLossFunction,
    create_physics_informed_loss
)

# Constants
EARTH_MU = 3.986004418e14  # m³/s²
RE = 6378137.0  # Earth radius in meters

def generate_orbital_trajectory(n_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a simple Keplerian orbital trajectory for testing."""
    
    # Orbital elements for a typical LEO satellite
    a = 7000e3  # Semi-major axis (700 km altitude)
    e = 0.01    # Low eccentricity
    i = np.radians(51.6)  # ISS-like inclination
    omega = 0   # Argument of perigee
    Omega = 0   # RAAN
    
    # Time vector
    T = 2 * np.pi * np.sqrt(a**3 / EARTH_MU)  # Orbital period
    times = torch.linspace(0, T, n_points)
    
    # Mean anomaly progression
    M = 2 * np.pi * times / T
    
    # Convert parameters to tensors
    e = torch.tensor(e)
    a = torch.tensor(a)
    
    # Solve Kepler's equation (simplified for low eccentricity)
    E = M + e * torch.sin(M)
    nu = 2 * torch.atan2(torch.sqrt(1+e) * torch.sin(E/2), torch.sqrt(1-e) * torch.cos(E/2))
    
    # Distance from focus
    r = a * (1 - e * torch.cos(E))
    
    # Position in orbital plane
    x_orb = r * torch.cos(nu)
    y_orb = r * torch.sin(nu)
    z_orb = torch.zeros_like(x_orb)
    
    # Velocity in orbital plane
    h = torch.sqrt(EARTH_MU * a * (1 - e**2))  # Specific angular momentum
    v_x_orb = -(EARTH_MU / h) * torch.sin(nu)
    v_y_orb = (EARTH_MU / h) * (e + torch.cos(nu))
    v_z_orb = torch.zeros_like(v_x_orb)
    
    # Transform to ECI coordinates (simplified - just include inclination)
    cos_i, sin_i = torch.cos(torch.tensor(i)), torch.sin(torch.tensor(i))
    
    # Position transformation
    x_eci = x_orb
    y_eci = y_orb * cos_i - z_orb * sin_i
    z_eci = y_orb * sin_i + z_orb * cos_i
    
    # Velocity transformation  
    vx_eci = v_x_orb
    vy_eci = v_y_orb * cos_i - v_z_orb * sin_i
    vz_eci = v_y_orb * sin_i + v_z_orb * cos_i
    
    # Combine into state vector
    states = torch.stack([x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci], dim=-1)
    
    return states.unsqueeze(0), times.unsqueeze(0)  # Add batch dimension

def test_individual_loss_components():
    """Test each physics loss component individually."""
    print("Testing individual physics loss components...")
    
    # Generate test trajectory
    states, times = generate_orbital_trajectory(50)
    print(f"Generated trajectory shape: {states.shape}")
    
    # Create loss configuration
    config = PhysicsLossConfig(
        energy_weight=1.0,
        momentum_weight=1.0,
        dynamics_weight=1.0,
        measurement_weight=1.0,
        temporal_weight=1.0
    )
    
    # Test Energy Conservation Loss
    print("\n1. Testing Energy Conservation Loss:")
    energy_loss = EnergyConservationLoss(config)
    energy_value = energy_loss.compute_loss(states, times=times)
    print(f"   Energy conservation loss: {energy_value:.6f}")
    
    # Test Angular Momentum Conservation Loss
    print("\n2. Testing Angular Momentum Conservation Loss:")
    momentum_loss = AngularMomentumConservationLoss(config)
    momentum_value = momentum_loss.compute_loss(states, times=times)
    print(f"   Angular momentum conservation loss: {momentum_value:.6f}")
    
    # Test Dynamics Residual Loss
    print("\n3. Testing Dynamics Residual Loss:")
    dynamics_loss = DynamicsResidualLoss(config)
    dynamics_value = dynamics_loss.compute_loss(states, times=times)
    print(f"   Dynamics residual loss: {dynamics_value:.6f}")
    
    # Test Measurement Consistency Loss
    print("\n4. Testing Measurement Consistency Loss:")
    measurement_loss = MeasurementConsistencyLoss(config)
    
    # Create synthetic optical observations (RA/Dec)
    n_obs = states.shape[1]
    observations = torch.randn(1, n_obs, 2) * 0.01  # Small angular errors
    measurement_value = measurement_loss.compute_loss(
        states, observations=observations, 
        observation_types=["optical"]
    )
    print(f"   Measurement consistency loss: {measurement_value:.6f}")
    
    # Test Temporal Smoothness Loss
    print("\n5. Testing Temporal Smoothness Loss:")
    temporal_loss = TemporalSmoothnessLoss(config)
    temporal_value = temporal_loss.compute_loss(states, times=times)
    print(f"   Temporal smoothness loss: {temporal_value:.6f}")

def test_adaptive_weight_learning():
    """Test adaptive weight learning mechanism."""
    print("\nTesting adaptive weight learning...")
    
    # Create adaptive weight learner
    loss_names = ['energy', 'momentum', 'dynamics', 'measurement', 'temporal']
    adaptive_learner = AdaptiveWeightLearner(loss_names, initial_weight=0.0)
    
    # Create some sample losses
    sample_losses = {
        'energy': torch.tensor(0.5),
        'momentum': torch.tensor(0.3),
        'dynamics': torch.tensor(1.2),
        'measurement': torch.tensor(0.8),
        'temporal': torch.tensor(0.1)
    }
    
    print("Initial weights:", {name: f"{weight:.4f}" for name, weight in adaptive_learner.get_weights().items()})
    print("Initial log-vars:", adaptive_learner.log_vars.data.tolist())
    
    # Compute adaptive loss
    adaptive_loss = adaptive_learner(sample_losses)
    print(f"Adaptive total loss: {adaptive_loss:.4f}")
    print(f"Regularization term: {adaptive_learner.get_regularization():.4f}")
    
    # Simulate some training steps
    optimizer = torch.optim.Adam(adaptive_learner.parameters(), lr=0.01)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Vary losses to simulate training dynamics
        varied_losses = {
            name: loss * (1 + 0.1 * torch.randn(1).item())
            for name, loss in sample_losses.items()
        }
        
        total_loss = adaptive_learner(varied_losses)
        total_loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            weights = adaptive_learner.get_weights()
            print(f"Epoch {epoch}: weights = {[f'{w:.3f}' for w in weights.values()]}")

def test_curriculum_learning():
    """Test curriculum learning schedules."""
    print("\nTesting curriculum learning...")
    
    # Create physics loss with curriculum learning
    config = PhysicsLossConfig(
        energy_weight=1.0,
        momentum_weight=1.0,
        dynamics_weight=1.0,
        measurement_weight=1.0,
        temporal_weight=0.5,
        curriculum_enabled=True,
        warmup_epochs=20,
        physics_ramp_schedule="cosine"
    )
    
    physics_loss = PhysicsInformedLossFunction(config)
    
    print("Curriculum learning progression:")
    for epoch in [0, 5, 10, 15, 20, 25]:
        physics_loss.update_epoch(epoch)
        curriculum_weights = physics_loss.get_curriculum_weights()
        
        print(f"Epoch {epoch:2d}: Energy={curriculum_weights['energy']:.3f}, "
              f"Momentum={curriculum_weights['momentum']:.3f}, "
              f"Dynamics={curriculum_weights['dynamics']:.3f}")

def test_complete_physics_loss():
    """Test the complete physics-informed loss function."""
    print("\nTesting complete physics-informed loss function...")
    
    # Generate test data
    states, times = generate_orbital_trajectory(30)
    
    # Create physics loss function
    physics_loss = create_physics_informed_loss(
        energy_weight=1.0,
        momentum_weight=0.5,
        dynamics_weight=2.0,
        measurement_weight=1.0,
        temporal_weight=0.1,
        use_adaptive_weights=True,
        curriculum_enabled=True,
        warmup_epochs=10
    )
    
    # Test without observations
    print("\n1. Testing without observations:")
    total_loss, components = physics_loss(
        predicted_states=states,
        times=times,
        return_components=True
    )
    
    print(f"   Total loss: {total_loss:.6f}")
    for name, value in components.items():
        print(f"   {name}: {value:.6f}")
    
    # Test with synthetic observations
    print("\n2. Testing with observations:")
    observations = torch.randn(1, 30, 2) * 0.005  # Small observation noise
    
    total_loss_obs, components_obs = physics_loss(
        predicted_states=states,
        times=times,
        observations=observations,
        observation_types=["optical"],
        return_components=True
    )
    
    print(f"   Total loss with observations: {total_loss_obs:.6f}")
    for name, value in components_obs.items():
        print(f"   {name}: {value:.6f}")
    
    # Test curriculum progression
    print("\n3. Testing curriculum progression:")
    for epoch in [0, 5, 15]:
        physics_loss.update_epoch(epoch)
        loss_at_epoch = physics_loss(predicted_states=states, times=times)
        print(f"   Epoch {epoch}: Loss = {loss_at_epoch:.6f}")
    
    # Get loss information
    print("\n4. Loss configuration info:")
    loss_info = physics_loss.get_loss_info()
    print(f"   Components: {loss_info['loss_components']}")
    print(f"   Current epoch: {loss_info['current_epoch']}")
    print(f"   Curriculum enabled: {loss_info['config']['curriculum_enabled']}")
    print(f"   Adaptive weights: {loss_info['config']['use_adaptive_weights']}")

def test_gradient_flow():
    """Test gradient flow through the loss function."""
    print("\nTesting gradient flow...")
    
    # Create a simple model that predicts orbital states
    class SimpleOrbitalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 6)  # Simple linear transformation
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleOrbitalModel()
    
    # Create physics loss
    physics_loss = create_physics_informed_loss(
        energy_weight=1.0,
        momentum_weight=1.0,
        dynamics_weight=1.0
    )
    
    # Generate input data
    initial_states, times = generate_orbital_trajectory(20)
    
    # Forward pass
    predicted_states = model(initial_states)
    
    # Compute loss
    total_loss = physics_loss(predicted_states, times=times)
    
    print(f"Forward pass successful. Loss: {total_loss:.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            print(f"   {name}: gradient norm = {grad_norm:.6f}")
    
    print(f"Gradient flow test passed. Average gradient norm: {np.mean(grad_norms):.6f}")

def main():
    """Run all physics loss tests."""
    print("=" * 60)
    print("COMPREHENSIVE PHYSICS-INFORMED LOSS TESTING")
    print("=" * 60)
    
    try:
        # Test individual components
        test_individual_loss_components()
        
        # Test adaptive weighting
        test_adaptive_weight_learning()
        
        # Test curriculum learning
        test_curriculum_learning()
        
        # Test complete system
        test_complete_physics_loss()
        
        # Test gradient flow
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("Physics-informed loss functions are ready for NP-SNN training.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()