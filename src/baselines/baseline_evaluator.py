"""
Baseline Evaluator for Comprehensive Model Comparison.

Provides statistical comparison framework for NP-SNN against
classical orbital propagation, filtering, and neural network baselines.
Includes significance testing and performance metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, kruskal
import pandas as pd
from pathlib import Path
import json
import warnings

from .sgp4_baseline import SGP4Baseline
from .ekf_baseline import EKFBaseline
from .ukf_baseline import UKFBaseline
from .mlp_baseline import MLPBaseline
from .particle_filter_baseline import ParticleFilterBaseline
from .npsnn_baseline import NPSNNBaselineWrapper


class BaselineEvaluator:
    """
    Comprehensive evaluation framework for comparing baselines.
    
    Handles multiple baseline methods, statistical significance testing,
    performance metrics computation, and result visualization.
    """
    
    def __init__(self, 
                 horizons: List[float] = [0.5, 1.0, 2.0, 6.0, 12.0, 24.0],
                 significance_level: float = 0.05,
                 bootstrap_samples: int = 1000):
        """
        Initialize baseline evaluator.
        
        Args:
            horizons: Prediction horizons in hours
            significance_level: Statistical significance level
            bootstrap_samples: Number of bootstrap samples
        """
        self.horizons = np.array(horizons)
        self.significance_level = significance_level
        self.bootstrap_samples = bootstrap_samples
        
        # Available baselines
        self.baselines = {
            'SGP4': SGP4Baseline(),
            'EKF': EKFBaseline(include_j2=False),
            'EKF_J2': EKFBaseline(include_j2=True),
            'UKF': UKFBaseline(include_j2=False),
            'UKF_J2': UKFBaseline(include_j2=True),
            'MLP': MLPBaseline(),
            'PF500': ParticleFilterBaseline(n_particles=500),
            'PF1000': ParticleFilterBaseline(n_particles=1000),
            'NPSNN': NPSNNBaselineWrapper(
                model_type='orbital_tracking',
                uncertainty=True,
                physics_constraints=True,
                device='cpu'
            ),
            'NPSNN_minimal': NPSNNBaselineWrapper(
                model_type='minimal',
                uncertainty=False,
                physics_constraints=False,
                device='cpu'
            )
        }
        
        # Results storage
        self.results = {}
    
    def evaluate_baseline(self,
                         baseline_name: str,
                         test_trajectories: List[np.ndarray],
                         test_times: List[np.ndarray],
                         measurements: Optional[List[List[np.ndarray]]] = None,
                         measurement_times: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Evaluate single baseline on test data.
        
        Args:
            baseline_name: Name of baseline to evaluate
            test_trajectories: List of true trajectories
            test_times: List of time arrays for each trajectory
            measurements: Optional measurements for filtering methods
            measurement_times: Optional measurement times
            
        Returns:
            Dictionary of evaluation results
        """
        if baseline_name not in self.baselines:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        
        baseline = self.baselines[baseline_name]
        n_trajectories = len(test_trajectories)
        
        # Initialize results
        results = {
            'position_rmse': [],
            'velocity_rmse': [],
            'position_mae': [],
            'velocity_mae': [],
            'prediction_times': [],
            'uncertainties': [],
            'success_rate': 0.0
        }
        
        successful_predictions = 0
        
        print(f"Evaluating {baseline_name} on {n_trajectories} trajectories...")
        
        for i in range(n_trajectories):
            try:
                true_traj = test_trajectories[i]
                times = test_times[i]
                
                # Get initial state
                initial_state = true_traj[0]
                
                # Get measurements if available
                traj_measurements = measurements[i] if measurements else None
                traj_meas_times = measurement_times[i] if measurement_times else None
                
                # Predict trajectory
                pred_states, uncertainties = baseline.predict_trajectory(
                    initial_state, times, traj_measurements, traj_meas_times
                )
                
                # Compute errors at each horizon
                traj_pos_rmse = []
                traj_vel_rmse = []
                traj_pos_mae = []
                traj_vel_mae = []
                
                for horizon in self.horizons:
                    # Find closest time index
                    time_idx = np.argmin(np.abs(times - horizon))
                    
                    if time_idx < len(pred_states) and time_idx < len(true_traj):
                        # Position errors
                        pos_error = true_traj[time_idx, :3] - pred_states[time_idx, :3]
                        pos_rmse = np.sqrt(np.mean(pos_error**2))
                        pos_mae = np.mean(np.abs(pos_error))
                        
                        # Velocity errors  
                        vel_error = true_traj[time_idx, 3:] - pred_states[time_idx, 3:]
                        vel_rmse = np.sqrt(np.mean(vel_error**2))
                        vel_mae = np.mean(np.abs(vel_error))
                        
                        traj_pos_rmse.append(pos_rmse)
                        traj_vel_rmse.append(vel_rmse)
                        traj_pos_mae.append(pos_mae)
                        traj_vel_mae.append(vel_mae)
                    else:
                        # Pad with NaN for missing predictions
                        traj_pos_rmse.append(np.nan)
                        traj_vel_rmse.append(np.nan)
                        traj_pos_mae.append(np.nan)
                        traj_vel_mae.append(np.nan)
                
                results['position_rmse'].append(traj_pos_rmse)
                results['velocity_rmse'].append(traj_vel_rmse)
                results['position_mae'].append(traj_pos_mae)
                results['velocity_mae'].append(traj_vel_mae)
                results['uncertainties'].append(uncertainties)
                
                successful_predictions += 1
                
            except Exception as e:
                warnings.warn(f"Baseline {baseline_name} failed on trajectory {i}: {e}")
                # Add NaN results for failed predictions
                nan_results = [np.nan] * len(self.horizons)
                results['position_rmse'].append(nan_results)
                results['velocity_rmse'].append(nan_results)
                results['position_mae'].append(nan_results)
                results['velocity_mae'].append(nan_results)
                results['uncertainties'].append(None)
        
        results['success_rate'] = successful_predictions / n_trajectories
        
        # Convert to numpy arrays for easier manipulation
        results['position_rmse'] = np.array(results['position_rmse'])
        results['velocity_rmse'] = np.array(results['velocity_rmse'])
        results['position_mae'] = np.array(results['position_mae'])
        results['velocity_mae'] = np.array(results['velocity_mae'])
        
        print(f"✅ {baseline_name} evaluation complete. Success rate: {results['success_rate']:.2%}")
        
        return results
    
    def compare_baselines(self,
                         baseline_names: List[str],
                         test_trajectories: List[np.ndarray],
                         test_times: List[np.ndarray],
                         measurements: Optional[List[List[np.ndarray]]] = None,
                         measurement_times: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Compare multiple baselines with statistical testing.
        
        Args:
            baseline_names: List of baseline names to compare
            test_trajectories: List of true trajectories
            test_times: List of time arrays
            measurements: Optional measurements
            measurement_times: Optional measurement times
            
        Returns:
            Comprehensive comparison results
        """
        print(f"🔬 Comparing {len(baseline_names)} baselines...")
        
        # Evaluate each baseline
        baseline_results = {}
        for name in baseline_names:
            baseline_results[name] = self.evaluate_baseline(
                name, test_trajectories, test_times, measurements, measurement_times
            )
        
        # Compute comparison statistics
        comparison_results = {
            'baseline_results': baseline_results,
            'statistical_tests': self._perform_statistical_tests(baseline_results),
            'performance_summary': self._compute_performance_summary(baseline_results),
            'rankings': self._compute_rankings(baseline_results)
        }
        
        return comparison_results
    
    def _perform_statistical_tests(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        
        test_results = {}
        baseline_names = list(baseline_results.keys())
        
        # Position RMSE comparisons
        pos_rmse_data = []
        for name in baseline_names:
            results = baseline_results[name]
            # Flatten across trajectories and horizons, remove NaNs
            flat_rmse = results['position_rmse'].flatten()
            pos_rmse_data.append(flat_rmse[~np.isnan(flat_rmse)])
        
        # Velocity RMSE comparisons  
        vel_rmse_data = []
        for name in baseline_names:
            results = baseline_results[name]
            flat_rmse = results['velocity_rmse'].flatten()
            vel_rmse_data.append(flat_rmse[~np.isnan(flat_rmse)])
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        if len(baseline_names) > 2:
            try:
                pos_h_stat, pos_p_val = kruskal(*pos_rmse_data)
                vel_h_stat, vel_p_val = kruskal(*vel_rmse_data)
                
                test_results['kruskal_wallis'] = {
                    'position': {'h_statistic': pos_h_stat, 'p_value': pos_p_val},
                    'velocity': {'h_statistic': vel_h_stat, 'p_value': vel_p_val}
                }
            except Exception as e:
                warnings.warn(f"Kruskal-Wallis test failed: {e}")
        
        # Pairwise comparisons
        pairwise_tests = {}
        
        for i in range(len(baseline_names)):
            for j in range(i + 1, len(baseline_names)):
                name1, name2 = baseline_names[i], baseline_names[j]
                
                # Wilcoxon signed-rank test for paired samples
                try:
                    # Match samples by taking minimum length
                    min_len = min(len(pos_rmse_data[i]), len(pos_rmse_data[j]))
                    
                    if min_len > 5:  # Need sufficient samples
                        pos_stat, pos_p = wilcoxon(
                            pos_rmse_data[i][:min_len], 
                            pos_rmse_data[j][:min_len]
                        )
                        vel_stat, vel_p = wilcoxon(
                            vel_rmse_data[i][:min_len], 
                            vel_rmse_data[j][:min_len]
                        )
                        
                        pairwise_tests[f"{name1}_vs_{name2}"] = {
                            'position': {'statistic': pos_stat, 'p_value': pos_p},
                            'velocity': {'statistic': vel_stat, 'p_value': vel_p}
                        }
                        
                except Exception as e:
                    warnings.warn(f"Wilcoxon test failed for {name1} vs {name2}: {e}")
        
        test_results['pairwise_comparisons'] = pairwise_tests
        
        return test_results
    
    def _compute_performance_summary(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute performance summary statistics."""
        
        summary = {}
        
        for name, results in baseline_results.items():
            # Mean metrics across trajectories and horizons
            pos_rmse_mean = np.nanmean(results['position_rmse'])
            pos_rmse_std = np.nanstd(results['position_rmse'])
            vel_rmse_mean = np.nanmean(results['velocity_rmse'])
            vel_rmse_std = np.nanstd(results['velocity_rmse'])
            
            pos_mae_mean = np.nanmean(results['position_mae'])
            vel_mae_mean = np.nanmean(results['velocity_mae'])
            
            # Horizon-wise performance
            horizon_performance = {}
            for h_idx, horizon in enumerate(self.horizons):
                pos_rmse_h = np.nanmean(results['position_rmse'][:, h_idx])
                vel_rmse_h = np.nanmean(results['velocity_rmse'][:, h_idx])
                
                horizon_performance[f"{horizon}h"] = {
                    'position_rmse': pos_rmse_h,
                    'velocity_rmse': vel_rmse_h
                }
            
            summary[name] = {
                'overall': {
                    'position_rmse_mean': pos_rmse_mean,
                    'position_rmse_std': pos_rmse_std,
                    'velocity_rmse_mean': vel_rmse_mean,
                    'velocity_rmse_std': vel_rmse_std,
                    'position_mae_mean': pos_mae_mean,
                    'velocity_mae_mean': vel_mae_mean,
                    'success_rate': results['success_rate']
                },
                'by_horizon': horizon_performance
            }
        
        return summary
    
    def _compute_rankings(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute baseline rankings by different metrics."""
        
        baseline_names = list(baseline_results.keys())
        
        rankings = {}
        
        # Overall position RMSE ranking
        pos_rmse_means = []
        for name in baseline_names:
            mean_rmse = np.nanmean(baseline_results[name]['position_rmse'])
            pos_rmse_means.append((name, mean_rmse))
        
        pos_rmse_means.sort(key=lambda x: x[1])
        rankings['position_rmse'] = [name for name, _ in pos_rmse_means]
        
        # Overall velocity RMSE ranking
        vel_rmse_means = []
        for name in baseline_names:
            mean_rmse = np.nanmean(baseline_results[name]['velocity_rmse'])
            vel_rmse_means.append((name, mean_rmse))
        
        vel_rmse_means.sort(key=lambda x: x[1])
        rankings['velocity_rmse'] = [name for name, _ in vel_rmse_means]
        
        # Success rate ranking
        success_rates = []
        for name in baseline_names:
            success_rate = baseline_results[name]['success_rate']
            success_rates.append((name, success_rate))
        
        success_rates.sort(key=lambda x: x[1], reverse=True)
        rankings['success_rate'] = [name for name, _ in success_rates]
        
        return rankings
    
    def create_comparison_plots(self,
                               comparison_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Create comprehensive comparison visualizations."""
        
        figures = {}
        
        # Performance by horizon plot
        fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        baseline_names = list(comparison_results['baseline_results'].keys())
        
        # Position RMSE by horizon
        ax = axes[0, 0]
        for name in baseline_names:
            results = comparison_results['baseline_results'][name]
            mean_rmse = np.nanmean(results['position_rmse'], axis=0)
            std_rmse = np.nanstd(results['position_rmse'], axis=0)
            
            ax.plot(self.horizons, mean_rmse / 1000, 'o-', label=name, linewidth=2)
            ax.fill_between(self.horizons, 
                           (mean_rmse - std_rmse) / 1000,
                           (mean_rmse + std_rmse) / 1000,
                           alpha=0.2)
        
        ax.set_xlabel('Prediction Horizon (hours)')
        ax.set_ylabel('Position RMSE (km)')
        ax.set_title('Position RMSE by Prediction Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Velocity RMSE by horizon
        ax = axes[0, 1]
        for name in baseline_names:
            results = comparison_results['baseline_results'][name]
            mean_rmse = np.nanmean(results['velocity_rmse'], axis=0)
            std_rmse = np.nanstd(results['velocity_rmse'], axis=0)
            
            ax.plot(self.horizons, mean_rmse, 'o-', label=name, linewidth=2)
            ax.fill_between(self.horizons, 
                           mean_rmse - std_rmse,
                           mean_rmse + std_rmse,
                           alpha=0.2)
        
        ax.set_xlabel('Prediction Horizon (hours)')
        ax.set_ylabel('Velocity RMSE (m/s)')
        ax.set_title('Velocity RMSE by Prediction Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Overall performance comparison
        ax = axes[1, 0]
        pos_means = []
        pos_stds = []
        labels = []
        
        for name in baseline_names:
            results = comparison_results['baseline_results'][name]
            mean_val = np.nanmean(results['position_rmse']) / 1000
            std_val = np.nanstd(results['position_rmse']) / 1000
            pos_means.append(mean_val)
            pos_stds.append(std_val)
            labels.append(name)
        
        bars = ax.bar(labels, pos_means, yerr=pos_stds, capsize=5)
        ax.set_ylabel('Position RMSE (km)')
        ax.set_title('Overall Position RMSE Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        
        # Success rate comparison
        ax = axes[1, 1]
        success_rates = []
        for name in baseline_names:
            success_rates.append(comparison_results['baseline_results'][name]['success_rate'])
        
        bars = ax.bar(labels, success_rates)
        ax.set_ylabel('Success Rate')
        ax.set_title('Baseline Success Rates')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1)
        
        # Add values on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        figures['performance_comparison'] = fig1
        
        if save_path:
            fig1.savefig(f"{save_path}_performance.png", dpi=300, bbox_inches='tight')
        
        return figures
    
    def save_results(self, comparison_results: Dict[str, Any], save_path: str):
        """Save comparison results to files."""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        summary_data = {
            'performance_summary': comparison_results['performance_summary'],
            'rankings': comparison_results['rankings'],
            'horizons': self.horizons.tolist()
        }
        
        with open(save_path / 'comparison_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save statistical tests
        with open(save_path / 'statistical_tests.json', 'w') as f:
            json.dump(comparison_results['statistical_tests'], f, indent=2, default=str)
        
        print(f"📊 Results saved to {save_path}")


def test_baseline_evaluator():
    """Test baseline evaluator with synthetic data."""
    
    print("🧪 Testing Baseline Evaluator...")
    
    # Create synthetic test data
    n_trajectories = 5
    test_trajectories = []
    test_times = []
    
    for i in range(n_trajectories):
        # Create circular orbit
        initial_state = np.array([
            6.778e6 + i * 1000, 0.0, 0.0,    # position (m)
            0.0, 7660.0 + i * 10, 0.0        # velocity (m/s)
        ])
        
        times = np.linspace(0, 2, 25)  # 2 hours
        
        # Simple propagation for ground truth
        trajectory = np.zeros((len(times), 6))
        trajectory[0] = initial_state
        
        # Integrate using EKF dynamics
        ekf = EKFBaseline()
        for t_idx in range(1, len(times)):
            dt = (times[t_idx] - times[t_idx-1]) * 3600
            trajectory[t_idx] = ekf.integrate_state(trajectory[t_idx-1], dt)
        
        test_trajectories.append(trajectory)
        test_times.append(times)
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(
        horizons=[0.5, 1.0, 2.0],
        significance_level=0.05
    )
    
    # Test baseline comparison
    baseline_names = ['SGP4', 'EKF_J2', 'MLP']
    
    comparison_results = evaluator.compare_baselines(
        baseline_names, test_trajectories, test_times
    )
    
    print("✅ Baseline evaluation successful!")
    print(f"   Baselines tested: {baseline_names}")
    print(f"   Test trajectories: {n_trajectories}")
    
    # Print rankings
    rankings = comparison_results['rankings']
    print(f"   Position RMSE ranking: {rankings['position_rmse']}")
    print(f"   Velocity RMSE ranking: {rankings['velocity_rmse']}")
    print(f"   Success rate ranking: {rankings['success_rate']}")
    
    # Create plots
    figures = evaluator.create_comparison_plots(comparison_results)
    
    return comparison_results


if __name__ == "__main__":
    test_baseline_evaluator()