"""
Comprehensive evaluation framework for NP-SNN orbital prediction.

This module provides:
- Multi-horizon evaluation pipeline
- Statistical significance testing
- Batch evaluation of model checkpoints
- Automated report generation
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import warnings

from .metrics import (
    position_rmse, velocity_rmse, orbital_energy_error, 
    angular_momentum_error, trajectory_error_growth,
    uncertainty_calibration_metrics, compute_comprehensive_metrics,
    create_error_plots
)
# from ..data.orbital_dataset import OrbitalDataset  # Commented for standalone testing
# from ..models.npsnn import NPSNN  # Commented for standalone testing


class NPSNNEvaluator:
    """
    Comprehensive evaluation framework for NP-SNN models.
    
    Provides multi-horizon evaluation, statistical testing, and report generation
    following implementation plan section 4.5 requirements.
    """
    
    def __init__(self, 
                 model: Any,  # NPSNN
                 device: torch.device = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained NP-SNN model
            device: Device for computation
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Standard evaluation horizons (hours)
        self.eval_horizons = [
            1/60,    # 1 minute
            10/60,   # 10 minutes  
            1.0,     # 1 hour
            6.0,     # 6 hours
            24.0     # 24 hours
        ]
        
    def evaluate_single_trajectory(self, 
                                 trajectory: Dict[str, torch.Tensor],
                                 max_horizon_hours: float = 24.0) -> Dict[str, Any]:
        """
        Evaluate model on single trajectory with comprehensive metrics.
        
        Args:
            trajectory: Single trajectory from dataset
            max_horizon_hours: Maximum prediction horizon
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        with torch.no_grad():
            times = trajectory['times'].to(self.device)
            states_true = trajectory['states'].to(self.device)
            observations = trajectory['observations'].to(self.device)
            obs_mask = trajectory['observation_mask'].to(self.device)
            
            # Add batch dimension if needed
            if len(times.shape) == 1:
                times = times.unsqueeze(0)
                states_true = states_true.unsqueeze(0)
                observations = observations.unsqueeze(0)
                obs_mask = obs_mask.unsqueeze(0)
            
            # Model prediction
            output = self.model(times, observations, obs_mask)
            states_pred = output['states']
            uncertainties = output.get('uncertainty', None)
            
            # Convert to numpy for metrics computation
            times_np = times.cpu().numpy().flatten()
            states_pred_np = states_pred.cpu().numpy().squeeze()
            states_true_np = states_true.cpu().numpy().squeeze()
            
            if uncertainties is not None:
                uncertainties_np = uncertainties.cpu().numpy().squeeze()
            else:
                uncertainties_np = None
            
            # Convert times to hours
            times_hours = times_np / 3600.0
            
            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(
                states_pred_np, states_true_np, times_hours, uncertainties_np
            )
            
            # Add additional trajectory-level metrics
            metrics.update({
                'trajectory_length_hours': float(times_hours[-1] - times_hours[0]),
                'n_observations': int(obs_mask.sum().item()),
                'observation_rate': float(obs_mask.float().mean().item()),
                'final_position_error': float(np.linalg.norm(
                    states_pred_np[-1, :3] - states_true_np[-1, :3]
                )),
                'final_velocity_error': float(np.linalg.norm(
                    states_pred_np[-1, 3:] - states_true_np[-1, 3:]
                ))
            })
            
            return metrics
    
    def evaluate_dataset_old(self, 
                        dataset: Any,  # OrbitalDataset
                        max_trajectories: Optional[int] = None,
                        save_individual: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset.
        
        Args:
            dataset: Orbital dataset to evaluate
            max_trajectories: Limit number of trajectories (for speed)
            save_individual: Whether to save individual trajectory results
            
        Returns:
            Aggregated evaluation results
        """
        print(f"📊 Evaluating NP-SNN on {len(dataset)} trajectories...")
        
        all_metrics = []
        individual_results = []
        
        n_eval = min(len(dataset), max_trajectories or len(dataset))
        
        for i in range(n_eval):
            try:
                trajectory = dataset[i]
                metrics = self.evaluate_single_trajectory(trajectory)
                
                metrics['trajectory_id'] = i
                all_metrics.append(metrics)
                
                if save_individual:
                    individual_results.append({
                        'trajectory_id': i,
                        'metrics': metrics
                    })
                    
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{n_eval} trajectories")
                    
            except Exception as e:
                print(f"   Warning: Failed to evaluate trajectory {i}: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("No trajectories were successfully evaluated")
        
        print(f"✅ Successfully evaluated {len(all_metrics)} trajectories")
        
        # Aggregate statistics
        aggregated = self._aggregate_metrics(all_metrics)
        
        if save_individual:
            aggregated['individual_results'] = individual_results
            
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across trajectories with statistics."""
        
        def safe_extract(key, default=np.nan):
            values = []
            for m in all_metrics:
                val = m.get(key, default)
                if not np.isnan(val) and np.isfinite(val):
                    values.append(val)
            return np.array(values) if values else np.array([default])
        
        # Core metrics
        pos_rmse_values = safe_extract('position_rmse')
        vel_rmse_values = safe_extract('velocity_rmse')
        energy_rel_err = safe_extract('energy_relative_error')
        momentum_rel_err = safe_extract('angular_momentum_relative_error')
        
        # Trajectory characteristics
        traj_lengths = safe_extract('trajectory_length_hours')
        obs_rates = safe_extract('observation_rate')
        final_pos_errs = safe_extract('final_position_error')
        
        # Error growth aggregation
        error_growth_data = {horizon: [] for horizon in self.eval_horizons}
        for metrics in all_metrics:
            if 'error_growth' in metrics:
                growth = metrics['error_growth']
                for i, horizon in enumerate(growth.get('horizons_hours', [])):
                    if horizon in error_growth_data:
                        pos_rmse = growth['position_rmse'][i]
                        if not np.isnan(pos_rmse):
                            error_growth_data[horizon].append(pos_rmse)
        
        # Uncertainty calibration (if available)
        calib_metrics = {}
        nll_values = safe_extract('nll')
        calib_err_values = safe_extract('calibration_error')
        
        if len(nll_values) > 0 and not np.isnan(nll_values[0]):
            calib_metrics = {
                'nll_mean': float(np.mean(nll_values)),
                'nll_std': float(np.std(nll_values)),
                'calibration_error_mean': float(np.mean(calib_err_values)),
                'calibration_error_std': float(np.std(calib_err_values))
            }
        
        # Compile aggregated results
        aggregated = {
            'n_trajectories_evaluated': len(all_metrics),
            'evaluation_timestamp': datetime.now().isoformat(),
            
            # Position metrics
            'position_rmse': {
                'mean': float(np.mean(pos_rmse_values)),
                'std': float(np.std(pos_rmse_values)),
                'median': float(np.median(pos_rmse_values)),
                'p90': float(np.percentile(pos_rmse_values, 90)),
                'p95': float(np.percentile(pos_rmse_values, 95))
            },
            
            # Velocity metrics  
            'velocity_rmse': {
                'mean': float(np.mean(vel_rmse_values)),
                'std': float(np.std(vel_rmse_values)),
                'median': float(np.median(vel_rmse_values)),
                'p90': float(np.percentile(vel_rmse_values, 90))
            },
            
            # Physics conservation
            'energy_conservation': {
                'relative_error_mean': float(np.mean(energy_rel_err)),
                'relative_error_std': float(np.std(energy_rel_err)),
                'relative_error_median': float(np.median(energy_rel_err))
            },
            
            'momentum_conservation': {
                'relative_error_mean': float(np.mean(momentum_rel_err)),
                'relative_error_std': float(np.std(momentum_rel_err)),
                'relative_error_median': float(np.median(momentum_rel_err))
            },
            
            # Error growth by horizon
            'error_growth_by_horizon': {
                str(h): {
                    'mean': float(np.mean(errors)) if errors else np.nan,
                    'std': float(np.std(errors)) if errors else np.nan,
                    'median': float(np.median(errors)) if errors else np.nan,
                    'n_samples': len(errors)
                } for h, errors in error_growth_data.items()
            },
            
            # Dataset characteristics
            'dataset_characteristics': {
                'trajectory_length_hours': {
                    'mean': float(np.mean(traj_lengths)),
                    'std': float(np.std(traj_lengths))
                },
                'observation_rate': {
                    'mean': float(np.mean(obs_rates)),
                    'std': float(np.std(obs_rates))
                },
                'final_position_errors': {
                    'mean': float(np.mean(final_pos_errs)),
                    'median': float(np.median(final_pos_errs))
                }
            }
        }
        
        # Add uncertainty metrics if available
        if calib_metrics:
            aggregated['uncertainty_calibration'] = calib_metrics
        
        return aggregated
    
    def generate_evaluation_report(self, 
                                  results: Dict[str, Any],
                                  output_dir: Path,
                                  model_name: str = "NP-SNN") -> Path:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results from evaluate_dataset
            output_dir: Directory to save report
            model_name: Model name for report
            
        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"{model_name}_evaluation_report_{timestamp}"
        report_path.mkdir(exist_ok=True)
        
        # Save raw results
        with open(report_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary plots
        self._create_summary_plots(results, report_path, model_name)
        
        # Generate markdown report
        self._generate_markdown_report(results, report_path, model_name)
        
        print(f"📄 Evaluation report saved to: {report_path}")
        return report_path
    
    def _create_summary_plots(self, 
                            results: Dict[str, Any], 
                            output_dir: Path,
                            model_name: str):
        """Create summary visualization plots."""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Error growth plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        horizons = []
        mean_errors = []
        std_errors = []
        
        for horizon_str, stats in results['error_growth_by_horizon'].items():
            if not np.isnan(stats['mean']):
                horizons.append(float(horizon_str))
                mean_errors.append(stats['mean'])
                std_errors.append(stats['std'])
        
        if horizons:
            horizons = np.array(horizons)
            mean_errors = np.array(mean_errors)
            std_errors = np.array(std_errors)
            
            ax.loglog(horizons, mean_errors, 'bo-', linewidth=2, markersize=8, label='Mean RMSE')
            ax.fill_between(horizons, 
                          mean_errors - std_errors, 
                          mean_errors + std_errors,
                          alpha=0.3, label='±1σ')
            
        ax.set_xlabel('Prediction Horizon (hours)', fontsize=12)
        ax.set_ylabel('Position RMSE (m)', fontsize=12)
        ax.set_title(f'{model_name} - Error Growth Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_growth_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Evaluation Metrics Distribution', fontsize=16, fontweight='bold')
        
        # Position RMSE stats
        pos_stats = results['position_rmse']
        axes[0, 0].bar(['Mean', 'Median', 'P90', 'P95'], 
                      [pos_stats['mean'], pos_stats['median'], pos_stats['p90'], pos_stats['p95']],
                      color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Position RMSE Statistics (m)')
        axes[0, 0].set_ylabel('RMSE (m)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Velocity RMSE stats
        vel_stats = results['velocity_rmse']
        axes[0, 1].bar(['Mean', 'Median', 'P90'], 
                      [vel_stats['mean'], vel_stats['median'], vel_stats['p90']],
                      color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Velocity RMSE Statistics (m/s)')
        axes[0, 1].set_ylabel('RMSE (m/s)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Energy conservation
        energy_stats = results['energy_conservation']
        axes[1, 0].bar(['Mean', 'Median'], 
                      [energy_stats['relative_error_mean'], energy_stats['relative_error_median']],
                      color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Energy Conservation (Relative Error)')
        axes[1, 0].set_ylabel('Relative Error')
        axes[1, 0].set_yscale('log')
        
        # Angular momentum conservation
        momentum_stats = results['momentum_conservation']
        axes[1, 1].bar(['Mean', 'Median'], 
                      [momentum_stats['relative_error_mean'], momentum_stats['relative_error_median']],
                      color='gold', alpha=0.8)
        axes[1, 1].set_title('Angular Momentum Conservation (Relative Error)')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, 
                                results: Dict[str, Any], 
                                output_dir: Path,
                                model_name: str):
        """Generate markdown evaluation report."""
        
        report_content = f"""# {model_name} Evaluation Report

**Generated:** {results['evaluation_timestamp']}  
**Trajectories Evaluated:** {results['n_trajectories_evaluated']}

## Executive Summary

This report presents comprehensive evaluation results for the {model_name} model on orbital trajectory prediction tasks.

## Key Performance Metrics

### Position Accuracy
- **Mean RMSE:** {results['position_rmse']['mean']:.2f} m
- **Median RMSE:** {results['position_rmse']['median']:.2f} m  
- **90th Percentile:** {results['position_rmse']['p90']:.2f} m
- **95th Percentile:** {results['position_rmse']['p95']:.2f} m

### Velocity Accuracy
- **Mean RMSE:** {results['velocity_rmse']['mean']:.2f} m/s
- **Median RMSE:** {results['velocity_rmse']['median']:.2f} m/s
- **90th Percentile:** {results['velocity_rmse']['p90']:.2f} m/s

### Physics Conservation

#### Energy Conservation
- **Mean Relative Error:** {results['energy_conservation']['relative_error_mean']:.2e}
- **Median Relative Error:** {results['energy_conservation']['relative_error_median']:.2e}

#### Angular Momentum Conservation  
- **Mean Relative Error:** {results['momentum_conservation']['relative_error_mean']:.2e}
- **Median Relative Error:** {results['momentum_conservation']['relative_error_median']:.2e}

## Error Growth Analysis

| Horizon | Mean RMSE (m) | Std Dev (m) | Samples |
|---------|---------------|-------------|---------|
"""

        # Add error growth table
        for horizon_str, stats in results['error_growth_by_horizon'].items():
            if not np.isnan(stats['mean']):
                horizon = float(horizon_str)
                if horizon < 1:
                    horizon_label = f"{horizon*60:.0f} min"
                else:
                    horizon_label = f"{horizon:.1f} hr"
                
                report_content += f"| {horizon_label} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['n_samples']} |\n"

        report_content += f"""
## Dataset Characteristics

- **Average Trajectory Length:** {results['dataset_characteristics']['trajectory_length_hours']['mean']:.1f} ± {results['dataset_characteristics']['trajectory_length_hours']['std']:.1f} hours
- **Average Observation Rate:** {results['dataset_characteristics']['observation_rate']['mean']:.2%} ± {results['dataset_characteristics']['observation_rate']['std']:.2%}
- **Mean Final Position Error:** {results['dataset_characteristics']['final_position_errors']['mean']:.2f} m

"""

        # Add uncertainty calibration if available
        if 'uncertainty_calibration' in results:
            calib = results['uncertainty_calibration']
            report_content += f"""## Uncertainty Calibration

- **Negative Log-Likelihood:** {calib['nll_mean']:.3f} ± {calib['nll_std']:.3f}
- **Calibration Error:** {calib['calibration_error_mean']:.3f} ± {calib['calibration_error_std']:.3f}

"""

        report_content += """## Visualizations

![Error Growth Analysis](error_growth_analysis.png)

![Metrics Distribution](metrics_distribution.png)

## Conclusions

The evaluation results demonstrate the model's performance across multiple dimensions of orbital trajectory prediction accuracy and physics consistency.

---
*Report generated by NP-SNN Evaluation Framework*
"""

        # Save markdown report
        with open(output_dir / "README.md", 'w') as f:
            f.write(report_content)


def evaluate_model_checkpoint(checkpoint_path: Path,
                            test_dataset: Any,  # OrbitalDataset
                            output_dir: Path,
                            config: Any = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dataset: Test dataset
        output_dir: Output directory for results
        config: Model configuration (if needed for loading)
        
    Returns:
        Evaluation results
    """
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # TODO: Implement proper model loading from checkpoint
    # This depends on the exact checkpoint format used in training
    
    print(f"📁 Loaded checkpoint: {checkpoint_path}")
    print(f"📊 Evaluating on {len(test_dataset)} trajectories...")
    
    # Create evaluator and run evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For now, return placeholder - this needs actual model loading logic
    return {
        'checkpoint_path': str(checkpoint_path),
        'evaluation_status': 'needs_model_loading_implementation'
    }


def compare_multiple_models(model_results: Dict[str, Dict[str, Any]],
                          output_dir: Path) -> None:
    """
    Compare evaluation results from multiple models.
    
    Args:
        model_results: Dictionary of {model_name: evaluation_results}
        output_dir: Output directory for comparison report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    model_names = list(model_results.keys())
    colors = sns.color_palette("husl", len(model_names))
    
    # Position RMSE comparison
    for i, (model_name, results) in enumerate(model_results.items()):
        pos_rmse = results['position_rmse']['mean']
        vel_rmse = results['velocity_rmse']['mean']
        
        axes[0, 0].bar(model_name, pos_rmse, color=colors[i], alpha=0.7)
        axes[0, 1].bar(model_name, vel_rmse, color=colors[i], alpha=0.7)
    
    axes[0, 0].set_title('Position RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE (m)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].set_title('Velocity RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE (m/s)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Error growth comparison
    for i, (model_name, results) in enumerate(model_results.items()):
        horizons = []
        mean_errors = []
        
        for horizon_str, stats in results['error_growth_by_horizon'].items():
            if not np.isnan(stats['mean']):
                horizons.append(float(horizon_str))
                mean_errors.append(stats['mean'])
        
        if horizons:
            axes[1, 0].loglog(horizons, mean_errors, 'o-', 
                            color=colors[i], linewidth=2, label=model_name)
    
    axes[1, 0].set_xlabel('Prediction Horizon (hours)')
    axes[1, 0].set_ylabel('Position RMSE (m)')
    axes[1, 0].set_title('Error Growth Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Physics conservation comparison
    energy_errors = []
    momentum_errors = []
    
    for model_name, results in model_results.items():
        energy_errors.append(results['energy_conservation']['relative_error_mean'])
        momentum_errors.append(results['momentum_conservation']['relative_error_mean'])
    
    x_pos = np.arange(len(model_names))
    axes[1, 1].bar(x_pos - 0.2, energy_errors, 0.4, label='Energy', color='lightblue')
    axes[1, 1].bar(x_pos + 0.2, momentum_errors, 0.4, label='Angular Momentum', color='lightcoral')
    
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_title('Physics Conservation Comparison')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison saved to: {output_dir / 'model_comparison.png'}")