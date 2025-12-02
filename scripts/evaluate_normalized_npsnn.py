"""
Baseline evaluation script with normalized NP-SNN model.

This script evaluates the trained normalized NP-SNN against classical baselines
using properly normalized data and denormalized outputs for fair comparison.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from typing import Dict, List, Any

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import baseline comparison framework
from src.baselines import (
    BaselineEvaluator,
    SGP4Baseline,
    EKFBaseline,
    UKFBaseline, 
    MLPBaseline,
    ParticleFilterBaseline,
    NPSNNBaselineWrapper
)
from src.data.orbital_generator import OrbitalDataGenerator
from src.data.orbital_dataset import TrajectoryTransforms
from src.models.npsnn import NPSNN
from src.train.config import TrainingConfig

class NormalizedNPSNNWrapper(NPSNNBaselineWrapper):
    """
    NP-SNN wrapper that handles normalization/denormalization for fair comparison.
    """
    
    def __init__(self, model_path: str, config_path: str, device: torch.device = None):
        super().__init__(model_path, config_path, device)
        
        # Normalization parameters (same as training)
        self.position_scale = 1e7  # 10 Mm
        self.velocity_scale = 1e4  # 10 km/s
        
    def normalize_trajectory(self, times: np.ndarray, states: np.ndarray) -> Dict:
        """Normalize trajectory for model input."""
        # Convert to tensors
        times_tensor = torch.tensor(times, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        # Create trajectory dict
        trajectory = {
            'times': times_tensor,
            'states': states_tensor,
            'observations': states_tensor,  # Use states as observations
            'observation_mask': torch.ones(len(times), dtype=torch.bool)
        }
        
        # Apply normalization
        normalized_traj = TrajectoryTransforms.normalize_states(
            trajectory, 
            position_scale=self.position_scale,
            velocity_scale=self.velocity_scale
        )
        
        return normalized_traj
    
    def denormalize_states(self, normalized_states: torch.Tensor) -> torch.Tensor:
        """Denormalize model output back to physical units."""
        denormalized = normalized_states.clone()
        
        # Denormalize positions and velocities
        denormalized[..., :3] *= self.position_scale   # Positions
        denormalized[..., 3:6] *= self.velocity_scale  # Velocities
        
        return denormalized
    
    def predict_trajectory(self, initial_state: np.ndarray, times: np.ndarray, 
                          observations: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Predict trajectory with proper normalization handling.
        
        Args:
            initial_state: Initial orbital state [x,y,z,vx,vy,vz] in physical units
            times: Time points
            observations: Observations (optional)
            
        Returns:
            Predicted trajectory in physical units
        """
        # Create input trajectory with initial state repeated
        n_times = len(times)
        input_states = np.tile(initial_state, (n_times, 1))
        
        # Normalize input
        normalized_traj = self.normalize_trajectory(times, input_states)
        
        try:
            # Run model prediction on normalized data
            with torch.no_grad():
                self.model.eval()
                
                # Extract normalized inputs
                norm_times = normalized_traj['times'].unsqueeze(0).to(self.device)  # (1, T)
                norm_observations = normalized_traj['observations'].unsqueeze(0).to(self.device)  # (1, T, 6)
                norm_mask = normalized_traj['observation_mask'].unsqueeze(0).to(self.device)  # (1, T)
                
                # Model prediction
                predictions = self.model(
                    context_times=norm_times,
                    context_observations=norm_observations,
                    context_mask=norm_mask,
                    target_times=norm_times,
                    return_uncertainty=False
                )
                
                # Denormalize predictions
                denormalized_preds = self.denormalize_states(predictions)
                
                # Convert back to numpy
                result = denormalized_preds.squeeze(0).cpu().numpy()  # (T, 6)
                
        except Exception as e:
            print(f"Warning: NP-SNN prediction failed: {e}")
            # Fallback: return input trajectory
            result = input_states
            
        return result

def create_test_scenarios(n_scenarios: int = 15) -> List[Dict]:
    """Create test scenarios for evaluation."""
    print(f"🔧 Generating {n_scenarios} test scenarios...")
    
    # Initialize data generator
    data_gen = OrbitalDataGenerator(
        n_objects=1,
        time_horizon_hours=8.0,  # 8-hour prediction horizon
        dt_minutes=5.0,          # 5-minute time steps
        observation_noise_std=0.001,
        device=torch.device('cpu')
    )
    
    scenarios = []
    for i in range(n_scenarios):
        try:
            # Generate trajectory (unnormalized for baselines)
            traj = data_gen.generate_single_trajectory(
                add_observation_gaps=True,
                gap_probability=0.1
            )
            
            # Convert tensors to numpy for baseline evaluation
            times = traj['times'].numpy()
            states = traj['states'].numpy()
            observations = traj['observations'].numpy()
            observation_mask = traj['observation_mask'].numpy()
            
            scenarios.append({
                'scenario_id': f"test_{i:03d}",
                'times': times,
                'true_trajectory': states,
                'observations': observations,
                'observation_mask': observation_mask,
                'initial_state': states[0],  # First state as initial condition
                'metadata': {
                    'duration_hours': 8.0,
                    'time_step_minutes': 5.0,
                    'n_timesteps': len(times)
                }
            })
            
        except Exception as e:
            print(f"Warning: Failed to generate scenario {i}: {e}")
            continue
    
    print(f"✅ Generated {len(scenarios)} valid test scenarios")
    return scenarios

def run_evaluation():
    """Run comprehensive baseline evaluation with normalized NP-SNN."""
    print("🎯 NP-SNN vs Classical Baselines Evaluation")
    print("=" * 60)
    
    # Configuration
    model_path = "checkpoints/final_checkpoint.pt"
    config_path = "configs/npsnn_training.yaml" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📱 Device: {device}")
    print(f"🧠 Model: {model_path}")
    
    # Check if trained model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model not found at {model_path}")
        print("Please ensure the normalized training completed successfully.")
        return
    
    # Create test scenarios
    scenarios = create_test_scenarios(n_scenarios=15)
    if not scenarios:
        print("❌ Error: No valid test scenarios generated")
        return
    
    # Initialize baselines
    print("\n🔧 Initializing baselines...")
    baselines = {}
    
    try:
        # Classical baselines
        baselines['SGP4'] = SGP4Baseline()
        baselines['EKF_J2'] = EKFBaseline(include_j2=True)
        baselines['UKF_J2'] = UKFBaseline(include_j2=True)
        baselines['MLP'] = MLPBaseline()
        baselines['ParticleFilter'] = ParticleFilterBaseline(n_particles=100)
        
        # Normalized NP-SNN
        baselines['NPSNN_Normalized'] = NormalizedNPSNNWrapper(
            model_path=model_path,
            config_path=config_path, 
            device=device
        )
        
        print(f"✅ Initialized {len(baselines)} baselines")
        
    except Exception as e:
        print(f"❌ Error initializing baselines: {e}")
        return
    
    # Create evaluator
    evaluator = BaselineEvaluator(
        prediction_horizons=[1800, 3600, 7200, 14400, 28800],  # 0.5h, 1h, 2h, 4h, 8h
        metrics=['rmse_position', 'rmse_velocity']
    )
    
    # Run evaluation
    print("\n🚀 Running baseline evaluation...")
    results = evaluator.evaluate_baselines(
        baselines=baselines,
        scenarios=scenarios,
        save_path="results/normalized_npsnn_comparison/",
        parallel=False  # Disable parallel for debugging
    )
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    if results and 'summary_stats' in results:
        summary = results['summary_stats']
        
        # Position RMSE comparison at 8-hour horizon
        print("\n🎯 Position RMSE at 8-hour horizon:")
        for baseline_name in ['SGP4', 'EKF_J2', 'UKF_J2', 'MLP', 'ParticleFilter', 'NPSNN_Normalized']:
            if baseline_name in summary:
                horizon_key = '28800'  # 8 hours in seconds
                if horizon_key in summary[baseline_name] and 'rmse_position' in summary[baseline_name][horizon_key]:
                    rmse_pos = summary[baseline_name][horizon_key]['rmse_position']['mean']
                    rmse_std = summary[baseline_name][horizon_key]['rmse_position']['std']
                    print(f"  {baseline_name:20}: {rmse_pos:8.2f} ± {rmse_std:6.2f} km")
        
        # Performance comparison
        print("\n📈 Performance Analysis:")
        if 'NPSNN_Normalized' in summary and 'UKF_J2' in summary:
            horizon_key = '28800'
            try:
                npsnn_rmse = summary['NPSNN_Normalized'][horizon_key]['rmse_position']['mean']
                ukf_rmse = summary['UKF_J2'][horizon_key]['rmse_position']['mean']
                
                if ukf_rmse > 0:
                    improvement_factor = ukf_rmse / npsnn_rmse
                    if improvement_factor > 1:
                        print(f"  🎉 NP-SNN is {improvement_factor:.1f}× BETTER than UKF")
                    else:
                        print(f"  ⚠️  NP-SNN is {1/improvement_factor:.1f}× WORSE than UKF")
                
            except KeyError as e:
                print(f"  ❌ Could not compute performance comparison: {e}")
    
    print(f"\n📁 Detailed results saved to: results/normalized_npsnn_comparison/")
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    run_evaluation()