"""
MLflow logging utilities for experiment tracking.

This module provides:
- Standardized MLflow logging for NP-SNN experiments
- Artifact management and model versioning
- Reproducibility tracking (configs, seeds, environment)
"""

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
except ImportError:
    print("Warning: MLflow not available. Install mlflow for experiment tracking.")
    mlflow = None

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import git

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = nn = None

class NPSNNLogger:
    """MLflow logger for NP-SNN experiments."""
    
    def __init__(self, 
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None):
        
        if mlflow is None:
            raise ImportError("MLflow not available")
            
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        # Setup MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        mlflow.set_experiment(experiment_name)
        
        # Git information for reproducibility
        self.git_info = self._get_git_info()
        
        # Run context
        self.run_active = False
        self.run_id = None
        
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information for reproducibility."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                'commit_hash': repo.head.object.hexsha,
                'branch': repo.active_branch.name,
                'is_dirty': repo.is_dirty(),
                'remote_url': repo.remotes.origin.url if repo.remotes else None
            }
        except Exception as e:
            print(f"Warning: Could not get git info: {e}")
            return {}
    
    def start_run(self, 
                  config: Dict[str, Any],
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start MLflow run with configuration logging."""
        
        # Start run
        run = mlflow.start_run(run_name=self.run_name)
        self.run_id = run.info.run_id
        self.run_active = True
        
        # Log git info and reproducibility
        if self.git_info:
            mlflow.log_params({f"git_{k}": str(v) for k, v in self.git_info.items()})
            
        # Log environment info
        mlflow.log_params({
            "python_version": os.sys.version,
            "platform": os.name,
            "working_directory": os.getcwd()
        })
        
        # Log configuration
        self._log_config(config)
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
            
        return self.run_id
    
    def _log_config(self, config: Dict[str, Any], prefix: str = "") -> None:
        """Recursively log configuration parameters."""
        
        for key, value in config.items():
            param_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                self._log_config(value, f"{param_key}.")
            else:
                # Convert to string for MLflow
                param_value = str(value)
                if len(param_value) > 250:  # MLflow param limit
                    param_value = param_value[:247] + "..."
                mlflow.log_param(param_key, param_value)
    
    def log_metrics(self, 
                   metrics: Dict[str, Union[float, int]], 
                   step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        
        if not self.run_active:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
            
        # Filter out non-numeric values
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_metrics[key] = float(value)
            else:
                print(f"Warning: Skipping non-numeric metric {key}: {value}")
        
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
    
    def log_model(self, 
                  model: nn.Module,
                  model_name: str = "npsnn_model",
                  save_state_dict: bool = True) -> None:
        """Log PyTorch model to MLflow."""
        
        if torch is None:
            raise ImportError("PyTorch not available")
            
        if not self.run_active:
            raise RuntimeError("No active MLflow run")
            
        # Log model using MLflow PyTorch integration
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            registered_model_name=f"{self.experiment_name}_{model_name}"
        )
        
        # Also save state dict separately for easier loading
        if save_state_dict:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                mlflow.log_artifact(f.name, f"{model_name}_state_dict")
                os.unlink(f.name)
    
    def log_dataset_info(self, 
                        dataset_path: Union[str, Path],
                        dataset_metadata: Optional[Dict] = None) -> None:
        """Log dataset information and fingerprint."""
        
        dataset_path = Path(dataset_path)
        
        # Compute dataset fingerprint
        fingerprint = self._compute_file_hash(dataset_path)
        
        # Log dataset info
        mlflow.log_params({
            "dataset_path": str(dataset_path),
            "dataset_size_mb": dataset_path.stat().st_size / 1024 / 1024,
            "dataset_fingerprint": fingerprint
        })
        
        # Log metadata if provided
        if dataset_metadata:
            metadata_path = Path(f"dataset_metadata_{self.run_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_path))
            metadata_path.unlink()
    
    def log_plots(self, 
                  figures: Dict[str, Any],
                  artifact_path: str = "plots") -> None:
        """Log matplotlib figures to MLflow."""
        
        import tempfile
        import matplotlib.pyplot as plt
        
        for name, fig in figures.items():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(f.name, f"{artifact_path}/{name}.png")
                os.unlink(f.name)
                plt.close(fig)
    
    def log_trajectory_comparison(self,
                                 predicted_trajectories: np.ndarray,
                                 true_trajectories: np.ndarray,
                                 times: np.ndarray,
                                 object_names: Optional[List[str]] = None) -> None:
        """Log trajectory comparison plots and metrics."""
        
        import matplotlib.pyplot as plt
        
        n_objects = predicted_trajectories.shape[0] if predicted_trajectories.ndim > 2 else 1
        
        # Create comparison plots
        figs = {}
        
        for i in range(min(n_objects, 5)):  # Limit to 5 objects for clarity
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Trajectory Comparison - Object {object_names[i] if object_names else i}")
            
            if predicted_trajectories.ndim > 2:
                pred_traj = predicted_trajectories[i]
                true_traj = true_trajectories[i]
            else:
                pred_traj = predicted_trajectories
                true_traj = true_trajectories
            
            # Position components
            for j, label in enumerate(['X', 'Y', 'Z']):
                axes[0, j].plot(times, true_traj[:, j], 'b-', label='True', alpha=0.8)
                axes[0, j].plot(times, pred_traj[:, j], 'r--', label='Predicted', alpha=0.8)
                axes[0, j].set_title(f'Position {label}')
                axes[0, j].set_xlabel('Time (s)')
                axes[0, j].set_ylabel(f'{label} Position (m)')
                axes[0, j].legend()
                axes[0, j].grid(True, alpha=0.3)
            
            # Velocity components  
            for j, label in enumerate(['VX', 'VY', 'VZ']):
                axes[1, j].plot(times, true_traj[:, j+3], 'b-', label='True', alpha=0.8)
                axes[1, j].plot(times, pred_traj[:, j+3], 'r--', label='Predicted', alpha=0.8)
                axes[1, j].set_title(f'Velocity {label}')
                axes[1, j].set_xlabel('Time (s)')
                axes[1, j].set_ylabel(f'{label} Velocity (m/s)')
                axes[1, j].legend()
                axes[1, j].grid(True, alpha=0.3)
            
            plt.tight_layout()
            obj_name = object_names[i] if object_names else f"object_{i}"
            figs[f"trajectory_comparison_{obj_name}"] = fig
        
        # Log plots
        self.log_plots(figs)
        
        # Compute and log trajectory metrics
        position_rmse = np.sqrt(np.mean((predicted_trajectories[..., :3] - true_trajectories[..., :3])**2))
        velocity_rmse = np.sqrt(np.mean((predicted_trajectories[..., 3:] - true_trajectories[..., 3:])**2))
        
        self.log_metrics({
            "trajectory_position_rmse_m": float(position_rmse),
            "trajectory_velocity_rmse_ms": float(velocity_rmse)
        })
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file for fingerprinting."""
        
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def end_run(self) -> None:
        """End current MLflow run."""
        
        if self.run_active:
            mlflow.end_run()
            self.run_active = False
            self.run_id = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

def log_hyperparameter_sweep(experiment_name: str,
                            param_grid: Dict[str, List],
                            results: List[Dict]) -> None:
    """Log results from hyperparameter sweep."""
    
    if mlflow is None:
        raise ImportError("MLflow not available")
        
    mlflow.set_experiment(f"{experiment_name}_sweep")
    
    for i, result in enumerate(results):
        with mlflow.start_run(run_name=f"sweep_run_{i}"):
            
            # Log parameters
            params = result.get('params', {})
            mlflow.log_params(params)
            
            # Log metrics
            metrics = result.get('metrics', {})
            mlflow.log_metrics(metrics)
            
            # Log additional info
            if 'config' in result:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(result['config'], f, indent=2)
                    mlflow.log_artifact(f.name, "config.json")
                    os.unlink(f.name)

def create_experiment_summary(experiment_name: str) -> Dict[str, Any]:
    """Create summary of all runs in an experiment."""
    
    if mlflow is None:
        raise ImportError("MLflow not available")
        
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    summary = {
        'experiment_name': experiment_name,
        'num_runs': len(runs),
        'best_runs': {},
        'parameter_ranges': {},
        'metric_statistics': {}
    }
    
    # Find best runs for each metric
    for col in runs.columns:
        if col.startswith('metrics.'):
            metric_name = col.replace('metrics.', '')
            best_idx = runs[col].idxmin() if 'loss' in metric_name.lower() else runs[col].idxmax()
            summary['best_runs'][metric_name] = {
                'run_id': runs.loc[best_idx, 'run_id'],
                'value': runs.loc[best_idx, col]
            }
            
            # Metric statistics
            summary['metric_statistics'][metric_name] = {
                'mean': float(runs[col].mean()),
                'std': float(runs[col].std()),
                'min': float(runs[col].min()),
                'max': float(runs[col].max())
            }
    
    return summary