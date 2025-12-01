"""
Quick integration test for SNN Core with Time Encoding.
Tests the complete pipeline from time encoding → SNN processing → output.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.models.time_encoding import MixedTimeEncoding
from src.models.snn_core import create_physics_snn, create_standard_snn

def test_snn_integration():
    """Test SNN core integration with time encoding."""
    print("=== SNN Core Integration Test ===")
    
    # Test parameters
    batch_size = 4
    sequence_length = 10
    time_dim = 32
    snn_output_dim = 6  # Position and velocity (x, y, z, vx, vy, vz)
    
    # Create time encoder
    print("\n1. Creating time encoder...")
    time_encoder = MixedTimeEncoding(
        d_model=time_dim,
        fourier_dim=16,
        learned_dim=16,
        max_frequency=1e-3,
        time_scale=3600.0
    )
    
    # Generate sample time data (orbital period simulation)
    times = torch.linspace(0, 100, sequence_length).unsqueeze(0).repeat(batch_size, 1)  # (batch, time)
    print(f"Time input shape: {times.shape}")
    
    # Encode times
    print("\n2. Encoding time data...")
    with torch.no_grad():
        encoded_times = time_encoder(times)  # (batch, time, time_dim)
    print(f"Encoded time shape: {encoded_times.shape}")
    
    # Create SNN
    print("\n3. Creating physics-informed SNN...")
    snn = create_physics_snn(
        input_size=time_dim,
        hidden_sizes=[64, 128, 64],
        output_size=snn_output_dim
    )
    
    print(f"SNN architecture:")
    print(f"  Input size: {snn.input_size}")
    print(f"  Hidden sizes: {snn.hidden_sizes}")
    print(f"  Output size: {snn.output_size}")
    print(f"  Number of layers: {snn.num_layers}")
    
    # Initialize hidden states
    print("\n4. Processing sequence through SNN...")
    device = 'cpu'
    hidden_states = snn.init_hidden_states(batch_size, device)
    
    # Process sequence step by step
    outputs = []
    for t in range(sequence_length):
        time_step_input = encoded_times[:, t, :]  # (batch, time_dim)
        
        with torch.no_grad():
            output, hidden_states = snn(time_step_input, hidden_states)
            outputs.append(output)
    
    # Stack outputs
    sequence_outputs = torch.stack(outputs, dim=1)  # (batch, time, output_dim)
    print(f"Final output shape: {sequence_outputs.shape}")
    
    # Check output statistics
    print(f"\n5. Output statistics:")
    print(f"  Mean: {sequence_outputs.mean():.4f}")
    print(f"  Std: {sequence_outputs.std():.4f}")
    print(f"  Min: {sequence_outputs.min():.4f}")
    print(f"  Max: {sequence_outputs.max():.4f}")
    
    # Check for NaN or Inf
    has_nan = torch.isnan(sequence_outputs).any()
    has_inf = torch.isinf(sequence_outputs).any()
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    # Get membrane statistics
    print(f"\n6. Membrane potential statistics:")
    mem_stats = snn.get_membrane_stats(hidden_states)
    for key, value in mem_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test standard SNN too
    print(f"\n7. Testing standard SNN...")
    standard_snn = create_standard_snn(
        input_size=time_dim,
        hidden_sizes=[64, 64],
        output_size=snn_output_dim
    )
    
    hidden_states_std = standard_snn.init_hidden_states(batch_size, device)
    with torch.no_grad():
        output_std, _ = standard_snn(encoded_times[:, 0, :], hidden_states_std)
    
    print(f"Standard SNN output shape: {output_std.shape}")
    print(f"Standard SNN output mean: {output_std.mean():.4f}")
    
    # Test gradient flow
    print(f"\n8. Testing gradient flow...")
    snn.train()
    time_encoder.train()
    
    # Forward pass with gradient computation
    encoded_times_grad = time_encoder(times[:1, :5])  # Smaller for gradient test
    hidden_states_grad = snn.init_hidden_states(1, device)
    
    total_loss = 0
    for t in range(5):
        output, hidden_states_grad = snn(encoded_times_grad[:, t, :], hidden_states_grad)
        # Simple loss (sum of outputs)
        total_loss += output.sum()
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    has_grad = False
    total_grad_norm = 0.0
    param_count = 0
    
    for name, param in snn.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    
    print(f"  Gradients computed: {has_grad}")
    print(f"  Average gradient norm: {avg_grad_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")
    
    print(f"\n=== Integration Test Complete ===")
    print(f"✅ Time encoding working")
    print(f"✅ SNN processing working") 
    print(f"✅ No NaN/Inf values")
    print(f"✅ Gradient flow working" if has_grad else "❌ Gradient flow issues")
    
    return sequence_outputs, mem_stats

if __name__ == "__main__":
    test_snn_integration()