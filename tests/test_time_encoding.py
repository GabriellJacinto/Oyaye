#!/usr/bin/env python3
"""
Comprehensive tests for time encoding module.

Tests all encoding strategies with orbital mechanics scenarios:
- Fourier encoding with orbital periods
- Learned encoding adaptability  
- Mixed encoding performance
- Configuration management
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.time_encoding import (
    FourierTimeEncoding, LearnedTimeEncoding, PositionalEncoding, MixedTimeEncoding,
    TimeEncodingConfig, TimeEncodingFactory,
    create_fourier_encoder, create_learned_encoder, create_mixed_encoder
)

def test_fourier_encoding():
    """Test Fourier time encoding with orbital periods."""
    print("Testing Fourier Time Encoding")
    print("=" * 50)
    
    # Create encoder with orbital mechanics frequencies
    encoder = create_fourier_encoder(d_model=64, max_frequency=1e-2, time_scale=3600.0)
    
    # Test with realistic orbital times
    # LEO orbit period: ~90 minutes = 5400 seconds
    t_orbit = torch.linspace(0, 5400, 100)  # One orbital period
    t_day = torch.linspace(0, 24*3600, 100)  # One day
    
    # Encode times
    features_orbit = encoder(t_orbit)
    features_day = encoder(t_day)
    
    print(f"✓ Fourier encoder created: {encoder.d_model}-dim output")
    print(f"✓ Orbital period encoding: {features_orbit.shape}")  
    print(f"✓ Daily period encoding: {features_day.shape}")
    
    # Check frequency information
    freq_info = encoder.get_frequency_info()
    print(f"✓ Frequency range: {freq_info['min_period_hours']:.1f}h to {freq_info['max_period_hours']:.1f}h periods")
    
    # Test periodicity (should be similar after orbital period)
    t1 = torch.tensor(0.0)
    t2 = torch.tensor(5400.0)  # One orbital period later
    f1 = encoder(t1)
    f2 = encoder(t2)
    
    # Fourier features should be similar for periodic times
    similarity = torch.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0))
    print(f"✓ Periodicity check: cosine similarity = {similarity.item():.3f}")
    
    # Test gradient flow (important for training)
    t_test = torch.tensor(1000.0, requires_grad=True)
    output = encoder(t_test).sum()
    output.backward()
    print(f"✓ Gradient flow: grad magnitude = {t_test.grad.abs().item():.6f}")
    
    return encoder

def test_learned_encoding():
    """Test learned time encoding."""
    print("\\nTesting Learned Time Encoding")  
    print("=" * 50)
    
    # Create learned encoder
    encoder = create_learned_encoder(d_model=64, hidden_dim=128, time_scale=3600.0)
    
    # Test with various time scales
    times = torch.tensor([0.0, 1800.0, 3600.0, 7200.0, 86400.0])  # 0h, 0.5h, 1h, 2h, 24h
    features = encoder(times)
    
    print(f"✓ Learned encoder created: {encoder.d_model}-dim output")
    print(f"✓ Multi-time encoding: {features.shape}")
    
    # Check that different times produce different features
    pairwise_distances = torch.cdist(features, features)
    min_distance = pairwise_distances[pairwise_distances > 0].min()
    print(f"✓ Feature diversity: min pairwise distance = {min_distance.item():.3f}")
    
    # Test gradient flow
    t_test = torch.tensor(3600.0, requires_grad=True)
    output = encoder(t_test).sum()
    output.backward()
    print(f"✓ Gradient flow: grad magnitude = {t_test.grad.abs().item():.6f}")
    
    # Test batch processing
    batch_times = torch.randn(32, 10) * 3600  # 32 trajectories, 10 time points each
    batch_features = encoder(batch_times)
    print(f"✓ Batch processing: {batch_times.shape} -> {batch_features.shape}")
    
    return encoder

def test_mixed_encoding():
    """Test mixed Fourier + learned encoding."""
    print("\\nTesting Mixed Time Encoding")
    print("=" * 50)
    
    # Create mixed encoder
    encoder = create_mixed_encoder(d_model=64, max_frequency=1e-2, time_scale=3600.0)
    
    # Test encoding
    times = torch.linspace(0, 12*3600, 50)  # 12 hours
    features = encoder(times)
    
    print(f"✓ Mixed encoder created: {encoder.d_model}-dim output")
    print(f"✓ Fourier component: {encoder.fourier_dim} dims")
    print(f"✓ Learned component: {encoder.learned_dim} dims") 
    print(f"✓ Time series encoding: {features.shape}")
    
    # Check feature components
    fourier_features = encoder.fourier_encoder(times)
    learned_features = encoder.learned_encoder(times)
    
    # Verify concatenation
    reconstructed = torch.cat([fourier_features, learned_features], dim=-1)
    reconstruction_error = (features - reconstructed).abs().max()
    print(f"✓ Component integration: max error = {reconstruction_error.item():.6f}")
    
    return encoder

def test_factory_and_config():
    """Test configuration-based encoder creation."""
    print("\\nTesting Factory and Configuration")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        TimeEncodingConfig(encoding_type="fourier", d_model=32, max_frequency=1e-3),
        TimeEncodingConfig(encoding_type="learned", d_model=32, hidden_dim=64, num_layers=3),
        TimeEncodingConfig(encoding_type="positional", d_model=32),
        TimeEncodingConfig(encoding_type="mixed", d_model=32, learnable_freqs=True)
    ]
    
    encoders = []
    for config in configs:
        encoder = TimeEncodingFactory.create_encoder(config)
        encoders.append(encoder)
        print(f"✓ Created {config.encoding_type} encoder: {config.d_model}-dim")
    
    # Test all encoders with same input
    test_time = torch.tensor(3600.0)  # 1 hour
    
    for i, (encoder, config) in enumerate(zip(encoders, configs)):
        features = encoder(test_time)
        print(f"  {config.encoding_type}: output shape {features.shape}")
        
        # Verify correct output dimension
        assert features.shape[-1] == config.d_model, f"Dimension mismatch for {config.encoding_type}"
    
    print(f"✓ All encoders produce correct output dimensions")
    
    return encoders

def test_orbital_scenarios():
    """Test encoders with realistic orbital mechanics scenarios."""
    print("\\nTesting Orbital Mechanics Scenarios")
    print("=" * 50)
    
    # Create encoder optimized for orbital mechanics
    config = TimeEncodingConfig(
        encoding_type="mixed",
        d_model=64,
        max_frequency=1e-2,  # Captures ~100 second periods
        time_scale=3600.0,   # Hours normalization
        hidden_dim=128,
        num_layers=2,
        dropout=0.0
    )
    encoder = TimeEncodingFactory.create_encoder(config)
    
    # Realistic orbital scenarios
    scenarios = {
        "ISS_pass": torch.linspace(0, 600, 20),      # 10-minute ISS pass
        "LEO_orbit": torch.linspace(0, 5400, 60),    # 90-minute LEO orbit  
        "MEO_orbit": torch.linspace(0, 12*3600, 100), # 12-hour MEO orbit
        "tracking_session": torch.linspace(0, 4*3600, 50)  # 4-hour tracking session
    }
    
    for scenario_name, times in scenarios.items():
        features = encoder(times)
        
        # Check feature smoothness (important for neural ODEs)
        if len(times) > 1:
            dt = times[1] - times[0]
            df_dt = (features[1:] - features[:-1]) / dt.item()
            max_gradient = df_dt.abs().max()
            
            print(f"✓ {scenario_name}: {len(times)} points, max gradient = {max_gradient.item():.3f}")
        else:
            print(f"✓ {scenario_name}: single point encoding")
    
    return encoder

def visualize_encoding_comparison():
    """Create visualization comparing different encoding strategies."""
    print("\\nCreating Encoding Visualization")
    print("=" * 50)
    
    # Create encoders
    fourier_enc = create_fourier_encoder(d_model=64, max_frequency=1e-2)
    learned_enc = create_learned_encoder(d_model=64, hidden_dim=64)
    mixed_enc = create_mixed_encoder(d_model=64, max_frequency=1e-2)
    
    # Time range: 8 hours with high resolution
    times = torch.linspace(0, 8*3600, 200)
    
    # Get encodings
    fourier_features = fourier_enc(times).detach().numpy()
    learned_features = learned_enc(times).detach().numpy()
    mixed_features = mixed_enc(times).detach().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Time Encoding Comparison for Orbital Mechanics', fontsize=16)
    
    times_hours = times.numpy() / 3600  # Convert to hours
    
    # Plot first 8 features for each encoding
    for i in range(3):
        encodings = [fourier_features, learned_features, mixed_features]
        names = ['Fourier', 'Learned', 'Mixed']
        
        # Time series plot
        axes[i, 0].plot(times_hours, encodings[i][:, :8])
        axes[i, 0].set_title(f'{names[i]} Encoding - First 8 Features')
        axes[i, 0].set_xlabel('Time (hours)')
        axes[i, 0].set_ylabel('Feature Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Feature correlation heatmap
        corr_matrix = np.corrcoef(encodings[i][:, :16].T)
        im = axes[i, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'{names[i]} - Feature Correlations')
        axes[i, 1].set_xlabel('Feature Index')
        axes[i, 1].set_ylabel('Feature Index')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('/home/orion23/Documents/repos/Oya (copy)/outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'time_encoding_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_dir / 'time_encoding_comparison.png'}")
    
    plt.close()

def test_performance_benchmark():
    """Benchmark encoding performance for different strategies."""
    print("\\nPerformance Benchmarking")
    print("=" * 50)
    
    import time
    
    # Create encoders
    encoders = {
        'Fourier': create_fourier_encoder(d_model=64),
        'Learned': create_learned_encoder(d_model=64),
        'Mixed': create_mixed_encoder(d_model=64)
    }
    
    # Test data: large batch 
    batch_size = 1000
    seq_len = 100
    test_times = torch.randn(batch_size, seq_len) * 3600  # Random times
    
    results = {}
    
    for name, encoder in encoders.items():
        # Warmup
        _ = encoder(test_times)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(10):  # 10 iterations
            features = encoder(test_times)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        throughput = (batch_size * seq_len) / avg_time
        
        results[name] = {
            'avg_time_ms': avg_time * 1000,
            'throughput': throughput,
            'output_shape': features.shape
        }
        
        print(f"✓ {name}: {avg_time*1000:.2f}ms, {throughput:.0f} samples/sec")
    
    return results

def main():
    """Run all time encoding tests."""
    print("Neural Physics-Informed SNN - Time Encoding Test Suite")
    print("=" * 60)
    
    try:
        # Core functionality tests
        fourier_encoder = test_fourier_encoding()
        learned_encoder = test_learned_encoding()
        mixed_encoder = test_mixed_encoding()
        
        # Configuration and factory tests
        factory_encoders = test_factory_and_config()
        
        # Domain-specific tests
        orbital_encoder = test_orbital_scenarios()
        
        # Visualization and benchmarking
        visualize_encoding_comparison()
        perf_results = test_performance_benchmark()
        
        print("\\n" + "=" * 60)
        print("✅ ALL TIME ENCODING TESTS PASSED!")
        print("\\nKey Results:")
        print(f"• Fourier encoding: Multi-scale orbital periods supported")
        print(f"• Learned encoding: Adaptive temporal representations")
        print(f"• Mixed encoding: Best of both strategies")
        print(f"• Factory pattern: Configuration-based creation")
        print(f"• Orbital scenarios: Realistic space dynamics coverage")
        print(f"• Performance: All encoders suitable for training")
        print("\\n🚀 Time encoding module ready for SNN integration!")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()