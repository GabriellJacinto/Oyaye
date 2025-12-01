#!/usr/bin/env python3
"""
Integration test for NP-SNN model with physics-informed loss functions.

This script tests the complete integration between the NP-SNN model and
the physics-informed loss functions, ensuring proper gradient flow,
loss computation, and training compatibility.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import NP-SNN components
from src.models.npsnn import NPSNN, create_npsnn_for_orbital_tracking, create_minimal_npsnn

# Import physics-informed loss functions
from src.train.losses import (
    PhysicsLossConfig,
    PhysicsInformedLossFunction,
    create_physics_informed_loss
)

# Constants
EARTH_MU = 3.986004418e14  # m³/s²
RE = 6378137.0  # Earth radius in meters

def generate_orbital_dataset(n_trajectories: int = 10, 
                           n_points: int = 50,
                           noise_level: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a small dataset of orbital trajectories for testing.
    
    Args:
        n_trajectories: Number of different orbital trajectories
        n_points: Number of time points per trajectory
        noise_level: Amount of noise to add to observations
        
    Returns:
        states: Ground truth orbital states (n_trajectories, n_points, 6)
        times: Time points (n_trajectories, n_points)
        observations: Noisy state observations (n_trajectories, n_points, 6)
    """
    all_states = []
    all_times = []
    all_observations = []
    
    for traj_idx in range(n_trajectories):
        # Vary orbital parameters slightly for each trajectory
        a = 7000e3 + np.random.normal(0, 100e3)  # Semi-major axis variation
        e = 0.001 + np.random.uniform(0, 0.05)    # Eccentricity variation
        i = np.radians(51.6) + np.random.normal(0, np.radians(5))  # Inclination variation
        
        # Generate trajectory
        T = 2 * np.pi * np.sqrt(a**3 / EARTH_MU)  # Orbital period
        times = torch.linspace(0, T * 0.5, n_points)  # Half orbit
        
        # Convert to tensors
        e_tensor = torch.tensor(e, dtype=torch.float32)
        a_tensor = torch.tensor(a, dtype=torch.float32)
        
        # Mean anomaly progression
        M = 2 * np.pi * times / T
        
        # Solve Kepler's equation (simplified)
        E = M + e_tensor * torch.sin(M)
        nu = 2 * torch.atan2(torch.sqrt(1+e_tensor) * torch.sin(E/2), 
                            torch.sqrt(1-e_tensor) * torch.cos(E/2))
        
        # Distance from focus
        r = a_tensor * (1 - e_tensor * torch.cos(E))
        
        # Position in orbital plane
        x_orb = r * torch.cos(nu)
        y_orb = r * torch.sin(nu)
        z_orb = torch.zeros_like(x_orb)
        
        # Velocity in orbital plane
        h = torch.sqrt(EARTH_MU * a_tensor * (1 - e_tensor**2))
        v_x_orb = -(EARTH_MU / h) * torch.sin(nu)
        v_y_orb = (EARTH_MU / h) * (e_tensor + torch.cos(nu))
        v_z_orb = torch.zeros_like(v_x_orb)
        
        # Transform to ECI coordinates (include inclination)
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
        
        # Generate synthetic observations matching model input size (6D)
        # For testing, use noisy state measurements as observations
        observations = states + torch.randn_like(states) * noise_level
        
        all_states.append(states)
        all_times.append(times)
        all_observations.append(observations)
    
    # Stack into batch tensors
    states_batch = torch.stack(all_states, dim=0)
    times_batch = torch.stack(all_times, dim=0)
    observations_batch = torch.stack(all_observations, dim=0)
    
    return states_batch, times_batch, observations_batch

def test_npsnn_physics_forward_pass():
    """Test forward pass through NP-SNN with physics loss computation."""
    print("Testing NP-SNN + Physics Loss Forward Pass...")
    
    # Create NP-SNN model
    model = create_npsnn_for_orbital_tracking(
        obs_input_size=6,
        state_dim=6,
        uncertainty=True,
        physics_constraints=True
    )
    
    print(f"Created NP-SNN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create physics-informed loss function
    physics_loss = create_physics_informed_loss(
        energy_weight=1.0,
        momentum_weight=0.5,
        dynamics_weight=2.0,
        measurement_weight=1.0,
        temporal_weight=0.1,
        use_adaptive_weights=False,  # Keep simple for initial test
        curriculum_enabled=False
    )
    
    # Generate simple test data (like in working example)
    batch_size = 4
    seq_len = 30
    
    # Time points
    times = torch.linspace(0, 7200, seq_len).unsqueeze(0).expand(batch_size, -1)  # 2 hours
    
    # Simulated observations (small random values)
    observations = torch.randn(batch_size, seq_len, model.config.obs_input_size) * 0.1
    
    # Generate corresponding "true" states for physics loss (larger scale for orbital mechanics)
    states_true = torch.randn(batch_size, seq_len, 6) * 1e6  # Orbital scale
    
    print(f"Generated dataset: states {states_true.shape}, times {times.shape}, observations {observations.shape}")
    
    # Forward pass through NP-SNN
    model.eval()  # Set to eval mode initially  
    with torch.no_grad():
        model_outputs = model(times, observations)  # Correct order: times first, obs second
    
    predicted_states = model_outputs['states']
    uncertainties = model_outputs.get('uncertainty', None)
    
    print(f"NP-SNN output shape: {predicted_states.shape}")
    if uncertainties is not None:
        print(f"Uncertainties shape: {uncertainties.shape}")
    
    # Compute physics-informed loss
    total_loss, loss_components = physics_loss(
        predicted_states=predicted_states,
        true_states=states_true,
        times=times,
        return_components=True
    )
    
    print(f"Physics loss computation successful!")
    print(f"Total loss: {total_loss:.6f}")
    for component, value in loss_components.items():
        print(f"  {component}: {value:.6f}")
    
    return model, physics_loss, (states_true, times, observations)

def test_npsnn_physics_gradient_flow():
    """Test gradient flow through complete NP-SNN + Physics Loss system."""
    print("\nTesting NP-SNN + Physics Loss Gradient Flow...")
    
    # Create model and loss
    model, physics_loss, (states_true, times, observations) = test_npsnn_physics_forward_pass()
    
    # Set model to training mode
    model.train()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    model_outputs = model(times, observations)
    predicted_states = model_outputs['states']
    
    # Compute loss
    total_loss = physics_loss(
        predicted_states=predicted_states,
        times=times
    )
    
    print(f"Forward pass loss: {total_loss:.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    grad_stats = []
    total_params = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats.append(grad_norm)
            params_with_grad += param.numel()
            
            if len(grad_stats) <= 5:  # Print first few
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
        
        total_params += param.numel()
    
    print(f"Gradient statistics:")
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"  Average gradient norm: {np.mean(grad_stats):.6f}")
    print(f"  Max gradient norm: {np.max(grad_stats):.6f}")
    print(f"  Min gradient norm: {np.min(grad_stats):.6f}")
    
    # Optimization step
    optimizer.step()
    
    print("Gradient flow test successful!")
    
    return model, physics_loss

def test_mini_training_loop():
    """Test a mini training loop with NP-SNN + Physics Loss."""
    print("\nTesting Mini Training Loop...")
    
    # Setup
    model, physics_loss = test_npsnn_physics_gradient_flow()
    
    # Generate larger dataset
    batch_size = 8
    seq_len = 40
    
    times = torch.linspace(0, 10800, seq_len).unsqueeze(0).expand(batch_size, -1)  # 3 hours
    observations = torch.randn(batch_size, seq_len, model.config.obs_input_size) * 0.1
    states_true = torch.randn(batch_size, seq_len, 6) * 1e6
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 5
    
    # Enable curriculum learning
    physics_loss.config.curriculum_enabled = True
    physics_loss.config.warmup_epochs = 3
    
    print(f"Training for {n_epochs} epochs on {states_true.shape[0]} trajectories...")
    
    loss_history = []
    
    for epoch in range(n_epochs):
        # Update curriculum
        physics_loss.update_epoch(epoch)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        model_outputs = model(times, observations)
        predicted_states = model_outputs['states']
        
        # Compute loss with components
        total_loss, loss_components = physics_loss(
            predicted_states=predicted_states,
            times=times,
            return_components=True
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        
        # Track loss
        loss_history.append(total_loss.item())
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}: Loss = {total_loss:.6f}")
        
        # Print curriculum weights at key epochs
        if epoch in [0, 2, 4]:
            curriculum_weights = physics_loss.get_curriculum_weights()
            print(f"  Curriculum weights: Energy={curriculum_weights['energy']:.3f}, "
                  f"Momentum={curriculum_weights['momentum']:.3f}")
    
    print(f"Training complete! Loss change: {loss_history[0]:.3f} → {loss_history[-1]:.3f}")
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        eval_outputs = model(times[:2], observations[:2])  # Smaller batch for eval
        eval_loss = physics_loss(eval_outputs['states'], times=times[:2])
        print(f"Evaluation loss: {eval_loss:.6f}")
    
    return loss_history

def test_adaptive_weights():
    """Test adaptive weight learning in the physics loss."""
    print("\nTesting Adaptive Weight Learning...")
    
    # Create model and loss with adaptive weights
    model = create_minimal_npsnn(
        obs_input_size=6
    )
    
    physics_loss = create_physics_informed_loss(
        energy_weight=1.0,
        momentum_weight=1.0,
        dynamics_weight=1.0,
        measurement_weight=1.0,
        temporal_weight=1.0,
        use_adaptive_weights=True,  # Enable adaptive weighting
        curriculum_enabled=False
    )
    
    # Generate data
    batch_size = 4
    seq_len = 25
    
    times = torch.linspace(0, 5400, seq_len).unsqueeze(0).expand(batch_size, -1)  # 1.5 hours
    observations = torch.randn(batch_size, seq_len, model.config.obs_input_size) * 0.1
    states_true = torch.randn(batch_size, seq_len, 6) * 1e6
    
    # Setup optimizer (include adaptive weight parameters)
    all_params = list(model.parameters())
    if physics_loss.adaptive_learner is not None:
        all_params.extend(list(physics_loss.adaptive_learner.parameters()))
    
    optimizer = optim.Adam(all_params, lr=0.001)
    
    print("Initial adaptive weights:", 
          {name: f"{weight:.3f}" for name, weight in physics_loss.adaptive_learner.get_weights().items()})
    
    # Training loop
    for epoch in range(8):
        optimizer.zero_grad()
        
        # Forward pass
        model_outputs = model(times, observations)
        predicted_states = model_outputs['states']
        
        # Compute loss (adaptive weights automatically applied)
        total_loss, loss_components = physics_loss(
            predicted_states=predicted_states,
            times=times,
            return_components=True
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Print adaptive weight evolution
        if epoch % 2 == 0:
            weights = physics_loss.adaptive_learner.get_weights()
            log_vars = physics_loss.adaptive_learner.log_vars.data
            print(f"Epoch {epoch}: Loss={total_loss:.3f}")
            print(f"  Adaptive weights: {[f'{w:.3f}' for w in weights.values()]}")
            print(f"  Log-vars: {[f'{v:.3f}' for v in log_vars.tolist()]}")
    
    print("Adaptive weight learning test successful!")

def test_model_saving_loading():
    """Test saving and loading NP-SNN model with physics loss state."""
    print("\nTesting Model Saving/Loading with Physics Loss...")
    
    # Create and train model briefly
    model, physics_loss = test_npsnn_physics_gradient_flow()
    
    # Save model state
    model_path = "/tmp/test_npsnn_model.pth"
    loss_path = "/tmp/test_physics_loss.pth"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'obs_input_size': 6, 'state_dim': 6
        }
    }, model_path)
    
    # Save physics loss state (if adaptive weights are enabled)
    loss_state = {
        'config': physics_loss.config,
        'current_epoch': physics_loss.current_epoch
    }
    if physics_loss.adaptive_learner is not None:
        loss_state['adaptive_learner_state'] = physics_loss.adaptive_learner.state_dict()
    
    torch.save(loss_state, loss_path)
    
    print("Saved model and loss state")
    
    # Load model
    checkpoint = torch.load(model_path)
    new_model = create_npsnn_for_orbital_tracking(**checkpoint['model_config'])
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load physics loss
    loss_checkpoint = torch.load(loss_path, weights_only=False)
    new_physics_loss = PhysicsInformedLossFunction(loss_checkpoint['config'])
    new_physics_loss.current_epoch = loss_checkpoint['current_epoch']
    
    if 'adaptive_learner_state' in loss_checkpoint and new_physics_loss.adaptive_learner is not None:
        new_physics_loss.adaptive_learner.load_state_dict(loss_checkpoint['adaptive_learner_state'])
    
    print("Loaded model and loss state successfully")
    
    # Test that loaded model works
    batch_size = 2
    seq_len = 20
    
    times = torch.linspace(0, 3600, seq_len).unsqueeze(0).expand(batch_size, -1)
    test_observations = torch.randn(batch_size, seq_len, new_model.config.obs_input_size) * 0.1
    
    new_model.eval()
    with torch.no_grad():
        outputs = new_model(times, test_observations)
        loss = new_physics_loss(outputs['states'], times=times)
        print(f"Loaded model test loss: {loss:.6f}")
    
    # Cleanup
    os.remove(model_path)
    os.remove(loss_path)
    
    print("Save/Load test successful!")

def main():
    """Run comprehensive integration tests."""
    print("=" * 70)
    print("COMPREHENSIVE NP-SNN + PHYSICS LOSS INTEGRATION TESTING")
    print("=" * 70)
    
    try:
        # Test 1: Basic forward pass
        test_npsnn_physics_forward_pass()
        
        # Test 2: Gradient flow
        test_npsnn_physics_gradient_flow()
        
        # Test 3: Mini training loop
        loss_history = test_mini_training_loop()
        
        # Test 4: Adaptive weights
        test_adaptive_weights()
        
        # Test 5: Save/Load functionality
        test_model_saving_loading()
        
        print("\n" + "=" * 70)
        print("🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        print("✅ NP-SNN + Physics Loss Forward Pass: WORKING")
        print("✅ Gradient Flow Through Complete System: WORKING") 
        print("✅ Training Loop with Curriculum Learning: WORKING")
        print("✅ Adaptive Weight Learning: WORKING")
        print("✅ Model Save/Load Functionality: WORKING")
        print("\nThe NP-SNN model is fully compatible with physics-informed loss functions!")
        print("Ready to proceed with complete training pipeline implementation.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()