"""
Simplified baseline evaluation script using existing infrastructure.

Compares the trained normalized NP-SNN against classical baselines using 
the existing evaluation framework from scripts/evaluate_baselines.py.
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import baseline comparison framework
from src.baselines import (
    BaselineEvaluator,
    SGP4Baseline,
    EKFBaseline,
    UKFBaseline, 
    MLPBaseline,
    ParticleFilterBaseline
)

# Constants and helpers from the existing evaluation script
GM_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6.378137e6  # m

class SimpleOrbitalDynamics:
    """Simple orbital dynamics for test data generation."""
    
    def __init__(self):
        self.GM_EARTH = GM_EARTH
        self.R_EARTH = R_EARTH
    
    def propagate_circular_orbit(self, radius, inclination, duration, dt):
        """Propagate a simple circular orbit."""
        n_steps = int(duration / dt) + 1
        times = np.linspace(0, duration, n_steps)
        
        # Mean motion
        n = np.sqrt(self.GM_EARTH / radius**3)
        
        # Initialize trajectory
        trajectory = np.zeros((n_steps, 6))
        
        for i, t in enumerate(times):
            # True anomaly
            nu = n * t
            
            # Position in orbital plane
            x = radius * np.cos(nu)
            y = radius * np.sin(nu)
            z = 0.0
            
            # Velocity in orbital plane  
            vx = -radius * n * np.sin(nu)
            vy = radius * n * np.cos(nu)
            vz = 0.0
            
            # Apply inclination rotation
            cos_i = np.cos(inclination)
            sin_i = np.sin(inclination)
            
            # Rotated position and velocity
            trajectory[i, 0] = x
            trajectory[i, 1] = y * cos_i
            trajectory[i, 2] = y * sin_i
            trajectory[i, 3] = vx
            trajectory[i, 4] = vy * cos_i
            trajectory[i, 5] = vy * sin_i
            
        return trajectory

def create_test_scenarios(n_scenarios: int = 15):
    """Create test scenarios using the simplified dynamics."""
    print(f"🔧 Generating {n_scenarios} test scenarios...")
    
    dynamics = SimpleOrbitalDynamics()
    scenarios = []
    
    for i in range(n_scenarios):
        # Random orbital parameters (LEO orbits)
        altitude = np.random.uniform(300e3, 800e3)  # 300-800 km altitude
        inclination = np.random.uniform(0, np.pi/3)   # 0-60 degrees inclination
        
        radius = dynamics.R_EARTH + altitude
        
        # Generate trajectory (8-hour prediction horizon)
        duration = 8 * 3600  # 8 hours in seconds
        dt = 300  # 5-minute time steps
        
        trajectory = dynamics.propagate_circular_orbit(radius, inclination, duration, dt)
        times = np.arange(0, duration + dt, dt)
        
        scenarios.append({
            'scenario_id': f"test_{i:03d}",
            'times': times,
            'true_trajectory': trajectory,
            'observations': trajectory + np.random.normal(0, 0.001 * np.abs(trajectory), trajectory.shape),
            'observation_mask': np.ones(len(times), dtype=bool),
            'initial_state': trajectory[0],
            'metadata': {
                'altitude_km': altitude / 1000,
                'inclination_deg': np.degrees(inclination),
                'duration_hours': 8.0,
                'n_timesteps': len(times)
            }
        })
    
    print(f"✅ Generated {len(scenarios)} test scenarios")
    return scenarios

def run_baseline_comparison():
    """Run comparison using only classical baselines (no NP-SNN yet)."""
    print("🎯 Classical Baselines Comparison")
    print("=" * 50)
    
    # Create test scenarios
    scenarios = create_test_scenarios(n_scenarios=15)
    
    # Initialize classical baselines only
    print("\n🔧 Initializing classical baselines...")
    baselines = {
        'SGP4': SGP4Baseline(),
        'EKF_J2': EKFBaseline(include_j2=True),
        'UKF_J2': UKFBaseline(include_j2=True),
        'MLP': MLPBaseline(),
        'ParticleFilter': ParticleFilterBaseline(n_particles=100)
    }
    
    print(f"✅ Initialized {len(baselines)} classical baselines")
    
    # Create evaluator
    evaluator = BaselineEvaluator(
        horizons=[0.5, 1.0, 2.0, 4.0, 8.0],  # 0.5h to 8h
        significance_level=0.05
    )
    
    # Run evaluation
    print("\n🚀 Running classical baseline evaluation...")
    results = evaluator.evaluate_baselines(
        baselines=baselines,
        scenarios=scenarios,
        save_path="results/classical_baselines_comparison/",
        parallel=False
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 CLASSICAL BASELINES RESULTS")
    print("=" * 50)
    
    if results and 'summary_stats' in results:
        summary = results['summary_stats']
        
        print("\n🎯 Position RMSE at different horizons:")
        horizons = ['1800', '3600', '7200', '14400', '28800']
        horizon_labels = ['0.5h', '1h', '2h', '4h', '8h']
        
        for baseline_name in ['SGP4', 'EKF_J2', 'UKF_J2', 'MLP', 'ParticleFilter']:
            if baseline_name in summary:
                print(f"\n{baseline_name}:")
                for horizon, label in zip(horizons, horizon_labels):
                    if horizon in summary[baseline_name] and 'rmse_position' in summary[baseline_name][horizon]:
                        rmse_pos = summary[baseline_name][horizon]['rmse_position']['mean']
                        rmse_std = summary[baseline_name][horizon]['rmse_position']['std']
                        print(f"  {label:>4}: {rmse_pos:8.2f} ± {rmse_std:6.2f} km")
    
    print(f"\n📁 Detailed results saved to: results/classical_baselines_comparison/")
    print("✅ Classical baseline evaluation complete!")
    
    return results

if __name__ == "__main__":
    # First run classical baselines to establish benchmark
    classical_results = run_baseline_comparison()
    
    print("\n" + "=" * 50)
    print("📝 SUMMARY")
    print("=" * 50)
    print("✅ Classical baseline evaluation completed successfully")
    print("🔄 To add NP-SNN comparison, load the trained model and extend this evaluation")
    print("📈 Current benchmark: UKF_J2 and EKF_J2 performance established")