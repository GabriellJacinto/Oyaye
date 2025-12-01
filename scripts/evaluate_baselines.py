#!/usr/bin/env python3
"""
Comprehensive NP-SNN vs Baselines Evaluation Script.

This script performs a complete scientific comparison of NP-SNN against
classical orbital propagation, modern filtering, and neural network baselines.
Generates publication-ready results with statistical significance testing.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

# Imports for data generation will be included inline

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_test_scenarios(n_scenarios: int = 50, 
                         prediction_horizon_hours: float = 24.0,
                         time_resolution_minutes: float = 30.0) -> tuple:
    """
    Create diverse orbital test scenarios for comprehensive evaluation.
    
    Args:
        n_scenarios: Number of test scenarios to generate
        prediction_horizon_hours: Prediction horizon in hours
        time_resolution_minutes: Time resolution in minutes
        
    Returns:
        Tuple of (trajectories, times, initial_states, scenario_info)
    """
    print(f"🛰️ Generating {n_scenarios} test scenarios...")
    
    trajectories = []
    times_list = []
    initial_states = []
    scenario_info = []
    
    # Time array
    n_points = int(prediction_horizon_hours * 60 / time_resolution_minutes) + 1
    times = np.linspace(0, prediction_horizon_hours, n_points)  # hours
    
    # Physical constants
    GM_EARTH = 3.986004418e14  # m^3/s^2
    R_EARTH = 6.378137e6  # Earth radius (m)
    
    for i in range(n_scenarios):
        # Generate diverse orbital scenarios
        scenario_type = np.random.choice([
            'leo_circular', 'leo_elliptical', 'meo_circular'
        ])
        
        try:
            # Generate initial state based on scenario type
            if scenario_type == 'leo_circular':
                # Low Earth Orbit - Circular
                altitude = np.random.uniform(300e3, 800e3)  # 300-800 km
                r = R_EARTH + altitude
                v = np.sqrt(GM_EARTH / r)  # Circular velocity
                
                # Random inclination and RAAN
                inclination = np.random.uniform(0, np.pi/3)
                raan = np.random.uniform(0, 2*np.pi)
                
                # Initial position and velocity (simplified)
                initial_state = np.array([
                    r * np.cos(raan),
                    r * np.sin(raan),
                    0,
                    -v * np.sin(raan) * np.cos(inclination),
                    v * np.cos(raan) * np.cos(inclination),
                    v * np.sin(inclination)
                ])
                
            elif scenario_type == 'leo_elliptical':
                # Low Earth Orbit - Elliptical
                perigee_alt = np.random.uniform(300e3, 600e3)
                apogee_alt = np.random.uniform(800e3, 2000e3)
                
                r_perigee = R_EARTH + perigee_alt
                r_apogee = R_EARTH + apogee_alt
                a = (r_perigee + r_apogee) / 2  # Semi-major axis
                
                # Velocity at perigee
                v_perigee = np.sqrt(GM_EARTH * (2/r_perigee - 1/a))
                
                # Random orientation
                raan = np.random.uniform(0, 2*np.pi)
                inclination = np.random.uniform(0, np.pi/3)
                
                initial_state = np.array([
                    r_perigee * np.cos(raan),
                    r_perigee * np.sin(raan),
                    0,
                    -v_perigee * np.sin(raan) * np.cos(inclination),
                    v_perigee * np.cos(raan) * np.cos(inclination),
                    v_perigee * np.sin(inclination)
                ])
                
            else:  # meo_circular
                # Medium Earth Orbit
                altitude = np.random.uniform(8000e3, 15000e3)  # MEO range
                r = R_EARTH + altitude
                v = np.sqrt(GM_EARTH / r)
                
                raan = np.random.uniform(0, 2*np.pi)
                inclination = np.random.uniform(np.pi/6, np.pi/2)
                
                initial_state = np.array([
                    r * np.cos(raan),
                    r * np.sin(raan),
                    0,
                    -v * np.sin(raan) * np.cos(inclination),
                    v * np.cos(raan) * np.cos(inclination),
                    v * np.sin(inclination)
                ])
            
            # Simple propagation using two-body dynamics
            times_seconds = times * 3600  # Convert to seconds
            true_trajectory = propagate_simple_orbit(initial_state, times_seconds)
            
            # Store results
            trajectories.append(true_trajectory)
            times_list.append(times)
            initial_states.append(initial_state)
            scenario_info.append({
                'type': scenario_type,
                'initial_altitude': np.linalg.norm(initial_state[:3]) - R_EARTH,
                'initial_velocity': np.linalg.norm(initial_state[3:]),
                'scenario_id': i
            })
            
        except Exception as e:
            warnings.warn(f"Failed to generate scenario {i} ({scenario_type}): {e}")
            # Use fallback circular LEO orbit
            r_fallback = R_EARTH + 500e3
            v_fallback = np.sqrt(GM_EARTH / r_fallback)
            
            fallback_state = np.array([
                r_fallback, 0, 0,
                0, v_fallback, 0
            ])
            
            times_seconds = times * 3600
            fallback_trajectory = propagate_simple_orbit(fallback_state, times_seconds)
            
            trajectories.append(fallback_trajectory)
            times_list.append(times)
            initial_states.append(fallback_state)
            scenario_info.append({
                'type': 'fallback_leo',
                'initial_altitude': 500e3,
                'initial_velocity': v_fallback,
                'scenario_id': i
            })
    
    print(f"✅ Generated {len(trajectories)} test scenarios")
    
    # Print scenario distribution
    scenario_types = [info['type'] for info in scenario_info]
    from collections import Counter
    type_counts = Counter(scenario_types)
    print("   Scenario distribution:")
    for scenario_type, count in type_counts.items():
        print(f"     {scenario_type}: {count}")
    
    return trajectories, times_list, initial_states, scenario_info


def propagate_simple_orbit(initial_state: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Simple orbital propagation using two-body dynamics.
    
    Args:
        initial_state: Initial orbital state [x, y, z, vx, vy, vz]
        times: Time points (seconds)
        
    Returns:
        Trajectory array (n_times, 6)
    """
    from scipy.integrate import solve_ivp
    
    GM_EARTH = 3.986004418e14  # m^3/s^2
    
    def dynamics(t, state):
        """Two-body orbital dynamics."""
        r = state[:3]
        v = state[3:]
        
        r_mag = np.linalg.norm(r)
        a = -GM_EARTH * r / (r_mag**3)
        
        return np.concatenate([v, a])
    
    # Integrate
    solution = solve_ivp(
        dynamics, 
        [times[0], times[-1]], 
        initial_state,
        t_eval=times,
        rtol=1e-8,
        atol=1e-11,
        method='DOP853'
    )
    
    if solution.success:
        return solution.y.T  # Transpose to (n_times, 6)
    else:
        # Fallback: return initial state for all times
        warnings.warn("Orbit propagation failed, using constant state")
        trajectory = np.zeros((len(times), 6))
        trajectory[:] = initial_state
        return trajectory


def run_comprehensive_evaluation(n_scenarios: int = 20,
                                prediction_horizon_hours: float = 12.0,
                                baseline_selection: list = None,
                                save_results: bool = True,
                                results_dir: str = "results/baseline_comparison") -> dict:
    """
    Run comprehensive evaluation of NP-SNN against all baselines.
    
    Args:
        n_scenarios: Number of test scenarios
        prediction_horizon_hours: Prediction horizon in hours
        baseline_selection: List of baseline names to evaluate (None = all)
        save_results: Whether to save results to disk
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing complete evaluation results
    """
    
    print("🔬 Starting Comprehensive NP-SNN vs Baselines Evaluation")
    print("=" * 60)
    
    # Create test scenarios
    trajectories, times_list, initial_states, scenario_info = create_test_scenarios(
        n_scenarios=n_scenarios,
        prediction_horizon_hours=prediction_horizon_hours
    )
    
    # Initialize baseline evaluator
    evaluator = BaselineEvaluator(
        horizons=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        significance_level=0.05,
        bootstrap_samples=1000
    )
    
    # Select baselines to evaluate
    if baseline_selection is None:
        # Default comprehensive selection
        baseline_selection = [
            'SGP4',           # Classical orbital propagation
            'EKF_J2',         # Extended Kalman Filter with J2
            'UKF_J2',         # Unscented Kalman Filter with J2
            'PF500',          # Particle Filter (500 particles)
            'MLP',            # Multi-Layer Perceptron
            'NPSNN',          # NP-SNN (full configuration)
            'NPSNN_minimal'   # NP-SNN (minimal configuration)
        ]
    
    print(f"\n📊 Evaluating {len(baseline_selection)} baselines:")
    for baseline in baseline_selection:
        print(f"   • {baseline}")
    
    # Since we don't have actual measurements, we'll evaluate without them
    # This tests the pure prediction capability of each method
    print(f"\n⚠️  Note: Evaluating prediction-only capability (no measurements)")
    
    # Run baseline comparison
    try:
        comparison_results = evaluator.compare_baselines(
            baseline_names=baseline_selection,
            test_trajectories=trajectories,
            test_times=times_list,
            measurements=None,  # No measurements - pure prediction test
            measurement_times=None
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        
        # Print summary results
        print_evaluation_summary(comparison_results)
        
        # Create comprehensive visualizations
        figures = create_comprehensive_visualizations(
            comparison_results, scenario_info, evaluator
        )
        
        # Save results if requested
        if save_results:
            save_path = Path(results_dir)
            save_evaluation_results(
                comparison_results, figures, scenario_info, save_path
            )
        
        return {
            'comparison_results': comparison_results,
            'figures': figures,
            'scenario_info': scenario_info,
            'evaluator': evaluator
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise e


def print_evaluation_summary(comparison_results: dict):
    """Print summary of evaluation results."""
    
    print("\n📋 EVALUATION SUMMARY")
    print("=" * 30)
    
    # Performance rankings
    rankings = comparison_results['rankings']
    
    print("\n🏆 Performance Rankings:")
    print(f"   Position RMSE: {' > '.join(rankings['position_rmse'])}")
    print(f"   Velocity RMSE: {' > '.join(rankings['velocity_rmse'])}")
    print(f"   Success Rate:  {' > '.join(rankings['success_rate'])}")
    
    # Success rates
    print("\n✅ Success Rates:")
    baseline_results = comparison_results['baseline_results']
    for name in sorted(baseline_results.keys()):
        success_rate = baseline_results[name]['success_rate']
        print(f"   {name:15s}: {success_rate:6.1%}")
    
    # Overall performance
    print("\n📊 Overall Performance (Position RMSE):")
    performance_summary = comparison_results['performance_summary']
    
    performance_list = []
    for name, summary in performance_summary.items():
        mean_rmse = summary['overall']['position_rmse_mean']
        if not np.isnan(mean_rmse):
            performance_list.append((name, mean_rmse))
    
    # Sort by performance (ascending RMSE)
    performance_list.sort(key=lambda x: x[1])
    
    if performance_list:
        best_rmse = performance_list[0][1]
        print(f"   {'Method':15s} {'RMSE (km)':>10s} {'Relative':>10s}")
        print("   " + "-" * 37)
        
        for name, rmse in performance_list:
            relative_perf = rmse / best_rmse if best_rmse > 0 else float('inf')
            print(f"   {name:15s} {rmse/1000:9.2f} {relative_perf:9.1f}x")
    
    # Statistical significance
    statistical_tests = comparison_results['statistical_tests']
    if 'kruskal_wallis' in statistical_tests:
        kw_test = statistical_tests['kruskal_wallis']
        print(f"\n🔬 Statistical Significance (Kruskal-Wallis):")
        print(f"   Position p-value: {kw_test['position']['p_value']:.6f}")
        print(f"   Velocity p-value: {kw_test['velocity']['p_value']:.6f}")
        
        alpha = 0.05
        if kw_test['position']['p_value'] < alpha:
            print(f"   ✅ Significant position differences detected (α = {alpha})")
        else:
            print(f"   ⚠️  No significant position differences (α = {alpha})")


def create_comprehensive_visualizations(comparison_results: dict, 
                                      scenario_info: list,
                                      evaluator: BaselineEvaluator) -> dict:
    """Create comprehensive visualization suite."""
    
    print("\n📊 Creating comprehensive visualizations...")
    
    figures = {}
    
    # 1. Performance comparison plots from evaluator
    try:
        eval_figures = evaluator.create_comparison_plots(comparison_results)
        figures.update(eval_figures)
    except Exception as e:
        warnings.warn(f"Failed to create evaluator plots: {e}")
    
    # 2. Detailed performance analysis
    try:
        fig_perf = create_detailed_performance_plot(comparison_results)
        figures['detailed_performance'] = fig_perf
    except Exception as e:
        warnings.warn(f"Failed to create detailed performance plot: {e}")
    
    # 3. Physics consistency analysis (for methods that support it)
    try:
        fig_physics = create_physics_analysis_plot(comparison_results)
        if fig_physics is not None:
            figures['physics_analysis'] = fig_physics
    except Exception as e:
        warnings.warn(f"Failed to create physics analysis plot: {e}")
    
    # 4. Scenario-based performance analysis
    try:
        fig_scenarios = create_scenario_performance_plot(comparison_results, scenario_info)
        if fig_scenarios is not None:
            figures['scenario_performance'] = fig_scenarios
    except Exception as e:
        warnings.warn(f"Failed to create scenario performance plot: {e}")
    
    print(f"✅ Created {len(figures)} visualization figures")
    
    return figures


def create_detailed_performance_plot(comparison_results: dict):
    """Create detailed performance analysis plot."""
    
    baseline_results = comparison_results['baseline_results']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Baseline Performance Analysis', fontsize=16, fontweight='bold')
    
    baseline_names = list(baseline_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_names)))
    
    # 1. Position RMSE distribution (log scale)
    ax = axes[0, 0]
    pos_data = []
    labels = []
    for name in baseline_names:
        data = baseline_results[name]['position_rmse'].flatten()
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            pos_data.append(valid_data / 1000)  # Convert to km
            labels.append(name)
    
    if pos_data:
        bp = ax.boxplot(pos_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(pos_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Position RMSE (km)')
        ax.set_title('Position RMSE Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # 2. Velocity RMSE distribution
    ax = axes[0, 1]
    vel_data = []
    labels = []
    for name in baseline_names:
        data = baseline_results[name]['velocity_rmse'].flatten()
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vel_data.append(valid_data)
            labels.append(name)
    
    if vel_data:
        bp = ax.boxplot(vel_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(vel_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Velocity RMSE (m/s)')
        ax.set_title('Velocity RMSE Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # 3. Success rates
    ax = axes[0, 2]
    success_rates = []
    names = []
    for name in baseline_names:
        rate = baseline_results[name]['success_rate']
        success_rates.append(rate)
        names.append(name)
    
    bars = ax.bar(names, success_rates, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Success Rate')
    ax.set_title('Baseline Success Rates')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 1.1)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{rate:.0%}', ha='center', va='bottom')
    
    # 4. Performance vs Horizon (Position)
    ax = axes[1, 0]
    horizons = comparison_results['performance_summary'][baseline_names[0]]['by_horizon'].keys()
    horizons = [float(h.replace('h', '')) for h in horizons]
    
    for name, color in zip(baseline_names, colors):
        summary = comparison_results['performance_summary'][name]
        horizon_rmse = []
        
        for horizon_key in summary['by_horizon']:
            rmse = summary['by_horizon'][horizon_key]['position_rmse']
            horizon_rmse.append(rmse / 1000 if not np.isnan(rmse) else np.nan)
        
        # Only plot if we have valid data
        valid_indices = ~np.isnan(horizon_rmse)
        if np.any(valid_indices):
            valid_horizons = np.array(horizons)[valid_indices]
            valid_rmse = np.array(horizon_rmse)[valid_indices]
            ax.plot(valid_horizons, valid_rmse, 'o-', label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('Position RMSE (km)')
    ax.set_title('Performance vs Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. Relative performance comparison
    ax = axes[1, 1]
    
    # Get overall position RMSE for each baseline
    baseline_rmse = []
    baseline_names_valid = []
    
    for name in baseline_names:
        rmse = comparison_results['performance_summary'][name]['overall']['position_rmse_mean']
        if not np.isnan(rmse):
            baseline_rmse.append(rmse)
            baseline_names_valid.append(name)
    
    if baseline_rmse:
        # Normalize to best performer
        min_rmse = min(baseline_rmse)
        relative_rmse = [rmse / min_rmse for rmse in baseline_rmse]
        
        bars = ax.bar(baseline_names_valid, relative_rmse, color=colors[:len(baseline_names_valid)], alpha=0.7)
        ax.set_ylabel('Relative Performance (1 = best)')
        ax.set_title('Relative Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Best Performance')
        ax.legend()
        
        # Add values on bars
        for bar, rel_perf in zip(bars, relative_rmse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{rel_perf:.1f}x', ha='center', va='bottom')
    
    # 6. Statistical significance heatmap
    ax = axes[1, 2]
    
    # Create pairwise comparison matrix if we have statistical tests
    statistical_tests = comparison_results['statistical_tests']
    if 'pairwise_comparisons' in statistical_tests:
        pairwise = statistical_tests['pairwise_comparisons']
        
        # Create matrix of p-values
        n_baselines = len(baseline_names)
        p_matrix = np.ones((n_baselines, n_baselines))
        
        for i, name1 in enumerate(baseline_names):
            for j, name2 in enumerate(baseline_names):
                if i != j:
                    key1 = f"{name1}_vs_{name2}"
                    key2 = f"{name2}_vs_{name1}"
                    
                    if key1 in pairwise and 'position' in pairwise[key1]:
                        p_value = pairwise[key1]['position']['p_value']
                        p_matrix[i, j] = p_value
                    elif key2 in pairwise and 'position' in pairwise[key2]:
                        p_value = pairwise[key2]['position']['p_value']
                        p_matrix[i, j] = p_value
        
        # Create heatmap
        im = ax.imshow(p_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
        ax.set_xticks(range(len(baseline_names)))
        ax.set_yticks(range(len(baseline_names)))
        ax.set_xticklabels(baseline_names, rotation=45)
        ax.set_yticklabels(baseline_names)
        ax.set_title('Statistical Significance\n(p-values, darker = more significant)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value')
        
        # Add significance markers
        for i in range(len(baseline_names)):
            for j in range(len(baseline_names)):
                if p_matrix[i, j] < 0.05 and i != j:
                    ax.text(j, i, '*', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=16)
    else:
        ax.text(0.5, 0.5, 'No statistical\ntests available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Statistical Significance')
    
    plt.tight_layout()
    return fig


def create_physics_analysis_plot(comparison_results: dict):
    """Create physics consistency analysis plot (placeholder for now)."""
    # This would analyze physics consistency for methods that support it
    # For now, return None as we'd need additional data collection
    return None


def create_scenario_performance_plot(comparison_results: dict, scenario_info: list):
    """Create scenario-based performance analysis."""
    # This would break down performance by orbital scenario type
    # For now, return None as we'd need to match results to scenarios
    return None


def save_evaluation_results(comparison_results: dict,
                          figures: dict,
                          scenario_info: list,
                          save_path: Path):
    """Save comprehensive evaluation results."""
    
    print(f"\n💾 Saving evaluation results to {save_path}...")
    
    # Create directories
    save_path.mkdir(parents=True, exist_ok=True)
    figures_path = save_path / "figures"
    figures_path.mkdir(exist_ok=True)
    
    # Save comparison results
    results_file = save_path / "comparison_results.json"
    
    # Convert numpy arrays and torch tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in comparison_results.items():
        if key == 'baseline_results':
            serializable_baseline_results = {}
            for baseline, results in value.items():
                serializable_baseline_results[baseline] = {
                    'success_rate': results['success_rate']
                }
                # Skip numpy arrays for now
            serializable_results[key] = serializable_baseline_results
        else:
            serializable_results[key] = value
    
    try:
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"✅ Saved results to {results_file}")
    except Exception as e:
        warnings.warn(f"Failed to save results JSON: {e}")
    
    # Save figures
    for name, fig in figures.items():
        try:
            fig_path = figures_path / f"{name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved figure: {fig_path}")
        except Exception as e:
            warnings.warn(f"Failed to save figure {name}: {e}")
    
    # Save scenario info
    try:
        scenario_file = save_path / "scenario_info.json"
        with open(scenario_file, 'w') as f:
            json.dump(scenario_info, f, indent=2, default=str)
        print(f"✅ Saved scenario info to {scenario_file}")
    except Exception as e:
        warnings.warn(f"Failed to save scenario info: {e}")
    
    # Create summary report
    create_summary_report(comparison_results, save_path)


def create_summary_report(comparison_results: dict, save_path: Path):
    """Create markdown summary report."""
    
    report_file = save_path / "evaluation_summary.md"
    
    with open(report_file, 'w') as f:
        f.write("# NP-SNN vs Baselines - Evaluation Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance rankings
        rankings = comparison_results['rankings']
        f.write("## Performance Rankings\n\n")
        f.write(f"**Position RMSE:** {' > '.join(rankings['position_rmse'])}\n\n")
        f.write(f"**Velocity RMSE:** {' > '.join(rankings['velocity_rmse'])}\n\n")
        f.write(f"**Success Rate:** {' > '.join(rankings['success_rate'])}\n\n")
        
        # Success rates table
        f.write("## Success Rates\n\n")
        f.write("| Method | Success Rate |\n")
        f.write("|--------|-------------|\n")
        
        baseline_results = comparison_results['baseline_results']
        for name in sorted(baseline_results.keys()):
            success_rate = baseline_results[name]['success_rate']
            f.write(f"| {name} | {success_rate:.1%} |\n")
        
        f.write("\n")
        
        # Performance summary
        f.write("## Overall Performance (Position RMSE)\n\n")
        f.write("| Method | RMSE (km) | Relative Performance |\n")
        f.write("|--------|-----------|--------------------|\n")
        
        performance_summary = comparison_results['performance_summary']
        
        performance_list = []
        for name, summary in performance_summary.items():
            mean_rmse = summary['overall']['position_rmse_mean']
            if not np.isnan(mean_rmse):
                performance_list.append((name, mean_rmse))
        
        # Sort by performance (ascending RMSE)
        performance_list.sort(key=lambda x: x[1])
        
        if performance_list:
            best_rmse = performance_list[0][1]
            for name, rmse in performance_list:
                relative_perf = rmse / best_rmse if best_rmse > 0 else float('inf')
                f.write(f"| {name} | {rmse/1000:.2f} | {relative_perf:.1f}x |\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- `comparison_results.json` - Complete evaluation results\n")
        f.write("- `scenario_info.json` - Test scenario information\n")
        f.write("- `figures/` - Visualization plots\n")
    
    print(f"✅ Created summary report: {report_file}")


def main():
    """Main evaluation script."""
    
    parser = argparse.ArgumentParser(description='NP-SNN vs Baselines Evaluation')
    parser.add_argument('--n-scenarios', type=int, default=20,
                       help='Number of test scenarios (default: 20)')
    parser.add_argument('--horizon-hours', type=float, default=12.0,
                       help='Prediction horizon in hours (default: 12.0)')
    parser.add_argument('--baselines', nargs='+', default=None,
                       help='Specific baselines to evaluate (default: all)')
    parser.add_argument('--results-dir', type=str, default='results/baseline_comparison',
                       help='Results directory (default: results/baseline_comparison)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to disk')
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        evaluation_results = run_comprehensive_evaluation(
            n_scenarios=args.n_scenarios,
            prediction_horizon_hours=args.horizon_hours,
            baseline_selection=args.baselines,
            save_results=not args.no_save,
            results_dir=args.results_dir
        )
        
        print(f"\n🎉 Evaluation completed successfully!")
        
        # Show plots if running interactively
        if not args.no_save:
            print(f"📊 Results saved to: {args.results_dir}")
        
        return evaluation_results
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise e


if __name__ == "__main__":
    # Set up for better error reporting
    import traceback
    
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit(1)