#!/usr/bin/env python3
"""
NP-SNN Complete Model Example

This example demonstrates how to use the complete Neural Physics-Informed
Spiking Neural Network for orbital mechanics applications.

Usage:
    python examples/complete_npsnn_example.py

Features demonstrated:
- Model creation with different configurations
- Forward pass and trajectory prediction
- Physics-informed loss computation  
- Save/load functionality
- Integration with orbital mechanics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.npsnn import (
    create_npsnn_for_orbital_tracking,
    create_npsnn_for_debris_tracking,
    create_minimal_npsnn
)

def demonstrate_model_creation():
    """Show different ways to create NP-SNN models."""
    print("🔧 NP-SNN Model Creation Examples")
    print("=" * 40)
    
    # 1. Orbital tracking model (full-featured)
    model_orbital = create_npsnn_for_orbital_tracking(
        obs_input_size=6,        # Position + velocity measurements
        uncertainty=True,        # Enable uncertainty quantification
        physics_constraints=True # Enable physics-aware constraints
    )
    
    info = model_orbital.get_model_info()
    print(f"📡 Orbital Tracking Model:")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Memory: {info['memory_usage_mb']:.1f} MB")
    print(f"   Decoder: {info['decoder_type']}")
    
    # 2. Debris tracking model (sensor fusion)
    model_debris = create_npsnn_for_debris_tracking(
        sensor_types=["optical", "radar"],  # Multiple sensor types
        multi_object=False                  # Single object tracking
    )
    
    info = model_debris.get_model_info()
    print(f"\n🛰️  Debris Tracking Model:")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Input size: {model_debris.config.obs_input_size} (optical + radar)")
    
    # 3. Minimal model (for testing/prototyping)
    model_minimal = create_minimal_npsnn(obs_input_size=3)
    
    info = model_minimal.get_model_info()
    print(f"\n⚡ Minimal Model:")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Physics constraints: {model_minimal.config.enable_physics_constraints}")
    
    return model_orbital, model_debris, model_minimal

def demonstrate_forward_pass(model):
    """Show how to use the model for prediction."""
    print("\n🚀 Forward Pass Example")
    print("=" * 25)
    
    # Create sample data
    batch_size = 3
    seq_len = 10
    
    # Time points (1 hour with 6-minute intervals)
    t = torch.linspace(0, 3600, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Simulated sensor observations (small random values)
    obs = torch.randn(batch_size, seq_len, model.config.obs_input_size) * 0.1
    
    print(f"Input shapes:")
    print(f"  Time: {t.shape}")
    print(f"  Observations: {obs.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model.forward(t=t, obs=obs, return_uncertainty=True)
    
    states = outputs['states']
    print(f"\nOutput shapes:")
    print(f"  States: {states.shape}")
    
    if 'uncertainty' in outputs:
        uncertainty = outputs['uncertainty']
        print(f"  Uncertainty: {uncertainty.shape}")
        print(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    
    # Physics violations (if physics-constrained model)
    if outputs['physics_violations']:
        print(f"\nPhysics constraint violations:")
        for key, value in outputs['physics_violations'].items():
            print(f"  {key}: {value.item():.6f}")
    
    return outputs

def demonstrate_trajectory_prediction(model):
    """Show trajectory prediction capabilities."""
    print("\n🛰️ Trajectory Prediction Example")
    print("=" * 35)
    
    # Define time span for prediction (2 hours, 20 steps)
    t_span = torch.linspace(0, 7200, 20)  # 2 hours in seconds
    
    # Initial observations (simulated sensor data)
    initial_obs = torch.randn(2, model.config.obs_input_size) * 0.1
    
    print(f"Predicting trajectory over {t_span[-1]/3600:.1f} hours")
    print(f"Time steps: {len(t_span)}")
    print(f"Initial observations: {initial_obs.shape}")
    
    # Predict trajectory
    trajectory = model.predict_trajectory(
        t_span=t_span,
        initial_obs=initial_obs,
        return_uncertainty=model.config.uncertainty_quantification
    )
    
    states = trajectory['states']
    print(f"\nTrajectory prediction:")
    print(f"  States shape: {states.shape}")
    print(f"  Position range: [{states[..., :3].min():.3f}, {states[..., :3].max():.3f}]")
    print(f"  Velocity range: [{states[..., 3:].min():.3f}, {states[..., 3:].max():.3f}]")
    
    # Physics consistency metrics
    consistency = trajectory['physics_consistency']
    if consistency:
        print(f"\nPhysics consistency:")
        print(f"  Energy conservation: {consistency['energy_conservation_violation']:.6f}")
        print(f"  Angular momentum conservation: {consistency['angular_momentum_conservation_violation']:.6f}")
        print(f"  Mean altitude: {consistency['mean_altitude']:.0f} m")
    
    return trajectory

def demonstrate_physics_informed_training(model):
    """Show physics-informed loss computation for training."""
    print("\n🔬 Physics-Informed Loss Example")
    print("=" * 35)
    
    # Enable gradients for training
    model.train()
    
    # Sample training data
    batch_size = 4
    t = torch.randn(batch_size, requires_grad=True)
    obs = torch.randn(batch_size, model.config.obs_input_size) * 0.1
    
    # Simulated ground truth orbital states
    true_states = torch.randn(batch_size, model.config.decoder.state_dim) * 1e5
    
    print(f"Training batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  True states range: [{true_states.min():.0f}, {true_states.max():.0f}]")
    
    # Compute physics-informed losses
    losses = model.physics_informed_loss(t=t, obs=obs, true_states=true_states)
    
    print(f"\nLoss components:")
    total_loss = 0
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
        total_loss += value
    
    print(f"  Total loss: {total_loss.item():.6f}")
    
    # Zero gradients first
    model.zero_grad()
    
    # Demonstrate gradient computation
    total_loss.backward()
    
    # Check gradient norms
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if grad_norms:
        print(f"\nGradient statistics:")
        print(f"  Max gradient norm: {max(grad_norms):.6f}")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        
        # Apply gradient clipping
        clipped_norm = model.clip_gradients(max_norm=1.0)
        print(f"  Clipped gradient norm: {clipped_norm:.6f}")
    else:
        print(f"\nNo gradients computed (likely due to graph reuse)")
    
    return losses

def demonstrate_save_load(model):
    """Show model save and load functionality."""
    print("\n💾 Save/Load Example")
    print("=" * 20)
    
    # Save model
    checkpoint_path = "outputs/models/npsnn_example.pth"
    
    model.save_checkpoint(
        filepath=checkpoint_path,
        optimizer_state={'lr': 0.001, 'momentum': 0.9},
        epoch=100,
        metrics={'train_loss': 0.123, 'val_loss': 0.456}
    )
    
    print(f"✓ Model saved to: {checkpoint_path}")
    
    # Load model
    loaded_model, checkpoint = type(model).load_checkpoint(checkpoint_path)
    
    print(f"✓ Model loaded successfully")
    print(f"  Saved epoch: {checkpoint['epoch']}")
    print(f"  Saved metrics: {checkpoint['metrics']}")
    
    # Verify loaded model works
    test_input = torch.randn(2, model.config.obs_input_size) * 0.1
    test_time = torch.randn(2)
    
    with torch.no_grad():
        original_output = model.forward(t=test_time, obs=test_input)
        loaded_output = loaded_model.forward(t=test_time, obs=test_input)
    
    # Check outputs match
    state_diff = torch.abs(original_output['states'] - loaded_output['states']).max()
    print(f"✓ Output difference: {state_diff:.2e} (should be ~0)")
    
    # Cleanup
    os.remove(checkpoint_path)
    
    return loaded_model

def visualize_trajectory(trajectory):
    """Create simple trajectory visualization."""
    print("\n📊 Trajectory Visualization")
    print("=" * 28)
    
    states = trajectory['states']
    
    # Extract positions and velocities
    positions = states[0, :, :3].numpy()  # First trajectory, all timesteps, xyz
    velocities = states[0, :, 3:6].numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position components over time
    axes[0, 0].plot(positions[:, 0], label='X')
    axes[0, 0].plot(positions[:, 1], label='Y')
    axes[0, 0].plot(positions[:, 2], label='Z')
    axes[0, 0].set_title('Position Components')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Velocity components over time
    axes[0, 1].plot(velocities[:, 0], label='Vx')
    axes[0, 1].plot(velocities[:, 1], label='Vy') 
    axes[0, 1].plot(velocities[:, 2], label='Vz')
    axes[0, 1].set_title('Velocity Components')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3D trajectory (if positions are reasonable)
    if np.abs(positions).max() > 1e-3:
        axes[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
        axes[1, 0].plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
        axes[1, 0].scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=50, label='Start')
        axes[1, 0].scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=50, label='End')
        axes[1, 0].set_title('3D Trajectory')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Positions too small\\nto visualize meaningfully', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('3D Trajectory (N/A)')
    
    # Uncertainty visualization (if available)
    if 'uncertainty' in trajectory:
        uncertainty = trajectory['uncertainty'][0, :, :].numpy()
        axes[1, 1].plot(uncertainty.mean(axis=1), label='Mean Uncertainty')
        axes[1, 1].fill_between(range(len(uncertainty)), 
                               uncertainty.min(axis=1), 
                               uncertainty.max(axis=1), 
                               alpha=0.3, label='Uncertainty Range')
        axes[1, 1].set_title('Uncertainty Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No uncertainty\\ninformation available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Uncertainty (N/A)')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "outputs/visualizations/npsnn_trajectory.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Trajectory plot saved to: {plot_path}")
    
    # Don't display - just save
    plt.close()

def main():
    """Run complete NP-SNN demonstration."""
    print("🧠 Neural Physics-Informed SNN - Complete Example")
    print("=" * 55)
    
    # 1. Model Creation
    models = demonstrate_model_creation()
    orbital_model = models[0]
    
    # 2. Forward Pass
    outputs = demonstrate_forward_pass(orbital_model)
    
    # 3. Trajectory Prediction
    trajectory = demonstrate_trajectory_prediction(orbital_model)
    
    # 4. Physics-Informed Training
    losses = demonstrate_physics_informed_training(orbital_model)
    
    # 5. Save/Load
    loaded_model = demonstrate_save_load(orbital_model)
    
    # 6. Visualization
    try:
        visualize_trajectory(trajectory)
    except ImportError:
        print("\n📊 Skipping visualization (matplotlib not available)")
    
    print("\n🎉 Complete NP-SNN demonstration finished!")
    print("\nKey Features Demonstrated:")
    print("✓ Model creation with different configurations")
    print("✓ Forward pass and state prediction")
    print("✓ Trajectory prediction with physics consistency")
    print("✓ Physics-informed loss computation")
    print("✓ Gradient stability and clipping")  
    print("✓ Model save/load functionality")
    print("✓ Uncertainty quantification")
    print("✓ Physics constraint enforcement")
    
    print(f"\nModel Summary:")
    info = orbital_model.get_model_info()
    print(f"  Architecture: {info['model_type']}")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Components: {', '.join(info['components'].values())}")
    print(f"  Physics: {orbital_model.config.enable_physics_constraints}")
    print(f"  Uncertainty: {orbital_model.config.uncertainty_quantification}")

if __name__ == "__main__":
    main()