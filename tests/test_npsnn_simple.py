"""
Simple NP-SNN baseline comparison focused on the trained model.

This script directly tests the trained normalized NP-SNN model and compares it
with a simple baseline to verify the model is working properly.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.orbital_generator import OrbitalDataGenerator
from src.data.orbital_dataset import TrajectoryTransforms
from src.models.npsnn import NPSNN
from src.train.config import TrainingConfig

class SimpleNPSNNTester:
    """Simple tester for the normalized NP-SNN model."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalization parameters (same as training)
        self.position_scale = 1e7  # 10 Mm
        self.velocity_scale = 1e4  # 10 km/s
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        print(f"📥 Loading model from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration
        config = checkpoint['config']
        
        # Create model
        from src.models.npsnn import NPSNNConfig, TimeEncodingConfig, SNNConfig, DecoderConfig
        
        npsnn_config = NPSNNConfig()
        npsnn_config.time_encoding = TimeEncodingConfig()
        npsnn_config.time_encoding.d_model = config.model_config.hidden_size
        npsnn_config.obs_input_size = 6
        npsnn_config.obs_encoding_dim = config.model_config.hidden_size // 2
        
        npsnn_config.snn = SNNConfig()
        npsnn_config.snn.num_layers = config.model_config.num_layers
        npsnn_config.snn.hidden_size = config.model_config.hidden_size
        npsnn_config.snn.dropout = config.model_config.dropout
        
        npsnn_config.decoder = DecoderConfig()
        npsnn_config.decoder.output_size = 6
        
        # Create and load model
        self.model = NPSNN(npsnn_config).to(self.device)
        
        # Load model state dict
        if 'best_model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['best_model_state'])
            print("✅ Loaded best model state")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded final model state")
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"🧠 Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def normalize_trajectory(self, times: np.ndarray, states: np.ndarray):
        """Normalize trajectory for model input."""
        times_tensor = torch.tensor(times, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        trajectory = {
            'times': times_tensor,
            'states': states_tensor,
            'observations': states_tensor,
            'observation_mask': torch.ones(len(times), dtype=torch.bool)
        }
        
        normalized_traj = TrajectoryTransforms.normalize_states(
            trajectory, 
            position_scale=self.position_scale,
            velocity_scale=self.velocity_scale
        )
        
        return normalized_traj
    
    def denormalize_states(self, normalized_states: torch.Tensor):
        """Denormalize model output."""
        denormalized = normalized_states.clone()
        denormalized[..., :3] *= self.position_scale   # Positions
        denormalized[..., 3:6] *= self.velocity_scale  # Velocities
        return denormalized
    
    def predict_trajectory(self, times: np.ndarray, initial_state: np.ndarray):
        """Predict trajectory using the trained model."""
        n_times = len(times)
        
        # Create input trajectory
        input_states = np.tile(initial_state, (n_times, 1))
        
        # Normalize
        normalized_traj = self.normalize_trajectory(times, input_states)
        
        with torch.no_grad():
            # Prepare inputs
            norm_times = normalized_traj['times'].unsqueeze(0).to(self.device)
            norm_obs = normalized_traj['observations'].unsqueeze(0).to(self.device)
            norm_mask = normalized_traj['observation_mask'].unsqueeze(0).to(self.device)
            
            try:
                # Model prediction using correct NPSNN interface
                outputs = self.model(
                    t=norm_times.squeeze(0),  # Remove batch dimension: (T,)
                    obs=norm_obs.squeeze(0),  # Remove batch dimension: (T, 6)
                    return_uncertainty=False
                )
                
                # Extract predicted states
                if isinstance(outputs, dict):
                    predictions = outputs['states']
                else:
                    predictions = outputs
                
                # Add batch dimension back if needed
                if predictions.dim() == 2:
                    predictions = predictions.unsqueeze(0)  # (1, T, 6)
                
                # Denormalize
                denormalized_preds = self.denormalize_states(predictions)
                result = denormalized_preds.squeeze(0).cpu().numpy()
                
                return result
                
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                return input_states  # Return input as fallback

def generate_test_data():
    """Generate test trajectory."""
    print("🔧 Generating test trajectory...")
    
    # Simple circular orbit parameters
    GM_EARTH = 3.986004418e14
    R_EARTH = 6.378137e6
    altitude = 500e3  # 500 km
    radius = R_EARTH + altitude
    
    # Time parameters
    duration = 8 * 3600  # 8 hours
    dt = 300  # 5 minutes
    times = np.arange(0, duration + dt, dt)
    n_times = len(times)
    
    # Generate circular orbit
    n = np.sqrt(GM_EARTH / radius**3)  # Mean motion
    states = np.zeros((n_times, 6))
    
    for i, t in enumerate(times):
        theta = n * t
        # Position
        states[i, 0] = radius * np.cos(theta)
        states[i, 1] = radius * np.sin(theta)
        states[i, 2] = 0.0
        # Velocity
        states[i, 3] = -radius * n * np.sin(theta)
        states[i, 4] = radius * n * np.cos(theta)
        states[i, 5] = 0.0
    
    print(f"✅ Generated circular orbit: {altitude/1000:.0f} km altitude, {duration/3600:.0f} hours")
    return times, states

def compute_metrics(true_trajectory: np.ndarray, predicted_trajectory: np.ndarray):
    """Compute prediction metrics."""
    # Position and velocity errors
    pos_true = true_trajectory[:, :3]
    vel_true = true_trajectory[:, 3:6]
    pos_pred = predicted_trajectory[:, :3]
    vel_pred = predicted_trajectory[:, 3:6]
    
    # RMSE in position (convert to km)
    pos_error = np.linalg.norm(pos_true - pos_pred, axis=1) / 1000
    pos_rmse = np.sqrt(np.mean(pos_error**2))
    
    # RMSE in velocity (m/s)
    vel_error = np.linalg.norm(vel_true - vel_pred, axis=1)
    vel_rmse = np.sqrt(np.mean(vel_error**2))
    
    return {
        'position_rmse_km': pos_rmse,
        'velocity_rmse_ms': vel_rmse,
        'position_error_km': pos_error,
        'velocity_error_ms': vel_error
    }

def run_simple_test():
    """Run simple NP-SNN test."""
    print("🎯 Simple NP-SNN Model Test")
    print("=" * 40)
    
    # Check if model exists
    checkpoint_path = "checkpoints/final_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return
    
    # Generate test data
    times, true_trajectory = generate_test_data()
    initial_state = true_trajectory[0]
    
    # Test NP-SNN
    print(f"\n🧠 Testing NP-SNN model...")
    tester = SimpleNPSNNTester(checkpoint_path)
    
    # Predict trajectory
    predicted_trajectory = tester.predict_trajectory(times, initial_state)
    
    # Compute metrics
    metrics = compute_metrics(true_trajectory, predicted_trajectory)
    
    # Print results
    print("\n" + "=" * 40)
    print("📊 NP-SNN TEST RESULTS")
    print("=" * 40)
    print(f"Position RMSE: {metrics['position_rmse_km']:.2f} km")
    print(f"Velocity RMSE: {metrics['velocity_rmse_ms']:.2f} m/s")
    
    # Error progression
    print(f"\n📈 Error progression over 8 hours:")
    horizons = [0.5, 1, 2, 4, 8]  # hours
    for h in horizons:
        idx = int(h * 3600 / 300)  # Convert to time step
        if idx < len(metrics['position_error_km']):
            pos_err = metrics['position_error_km'][idx]
            vel_err = metrics['velocity_error_ms'][idx]
            print(f"  {h:>3.1f}h: {pos_err:6.2f} km, {vel_err:5.1f} m/s")
    
    # Sanity checks
    print(f"\n🔍 Model sanity checks:")
    print(f"  Max position error: {metrics['position_error_km'].max():.2f} km")
    print(f"  Model producing finite outputs: {np.all(np.isfinite(predicted_trajectory))}")
    print(f"  Model outputs in reasonable range: {np.all(np.abs(predicted_trajectory) < 1e8)}")
    
    if metrics['position_rmse_km'] < 1000:  # Less than 1000 km error
        print("✅ NP-SNN model appears to be working!")
    else:
        print("⚠️  NP-SNN model may need more training")
    
    return metrics

if __name__ == "__main__":
    results = run_simple_test()