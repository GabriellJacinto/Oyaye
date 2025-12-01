#!/usr/bin/env python3
"""
Debug NP-SNN input dimensions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.npsnn import create_npsnn_for_orbital_tracking

def debug_model_input():
    """Debug model input dimensions."""
    print("Debugging NP-SNN input dimensions...")
    
    # Create model
    model = create_npsnn_for_orbital_tracking(obs_input_size=6, state_dim=6)
    
    print(f"Model obs_input_size: {model.config.obs_input_size}")
    print(f"Model obs_encoding_dim: {model.config.obs_encoding_dim}")
    
    # Create test data
    batch_size = 2
    seq_len = 10
    obs_dim = 6
    
    observations = torch.randn(batch_size, seq_len, obs_dim)
    times = torch.linspace(0, 100, seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    print(f"\nInput shapes:")
    print(f"  observations: {observations.shape}")
    print(f"  times: {times.shape}")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(observations, times)
        
        print(f"\nForward pass successful!")
        print(f"  predicted_states: {outputs['predicted_states'].shape}")
        if 'uncertainties' in outputs:
            print(f"  uncertainties: {outputs['uncertainties'].shape}")
            
    except Exception as e:
        print(f"\nForward pass failed: {e}")
        
        # Let's trace through the forward pass manually
        print("\nManual forward pass debug:")
        seq_len = observations.shape[1] 
        print(f"  seq_len extracted: {seq_len}")
        
        for step in range(min(3, seq_len)):  # Check first few steps
            t_current = times[:, step]
            obs_current = observations[:, step, :]
            print(f"  Step {step}: t_current={t_current.shape}, obs_current={obs_current.shape}")

if __name__ == "__main__":
    debug_model_input()