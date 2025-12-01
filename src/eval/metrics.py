"""
Evaluation metrics for orbital trajectory prediction.

This module provides:
- Standard trajectory error metrics (RMSE, MAE, etc.)
- Physics-based metrics (energy conservation, etc.)
- Uncertainty calibration metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import math

def position_rmse(predicted_states: np.ndarray, 
                 true_states: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """
    Root Mean Square Error for position predictions.
    
    Args:
        predicted_states: Predicted states (..., 6) with [x,y,z,vx,vy,vz]
        true_states: True states (..., 6) 
        mask: Optional mask for valid predictions
        
    Returns:
        RMSE in meters
    """
    pos_pred = predicted_states[..., :3]
    pos_true = true_states[..., :3]
    
    squared_errors = np.sum((pos_pred - pos_true)**2, axis=-1)
    
    if mask is not None:
        squared_errors = squared_errors[mask]
        
    return float(np.sqrt(np.mean(squared_errors)))

def velocity_rmse(predicted_states: np.ndarray,
                 true_states: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> float:
    """
    Root Mean Square Error for velocity predictions.
    
    Args:
        predicted_states: Predicted states (..., 6)
        true_states: True states (..., 6)
        mask: Optional mask for valid predictions
        
    Returns:
        RMSE in m/s
    """
    vel_pred = predicted_states[..., 3:]
    vel_true = true_states[..., 3:]
    
    squared_errors = np.sum((vel_pred - vel_true)**2, axis=-1)
    
    if mask is not None:
        squared_errors = squared_errors[mask]
        
    return float(np.sqrt(np.mean(squared_errors)))

def along_track_cross_track_errors(predicted_states: np.ndarray,
                                  true_states: np.ndarray) -> Dict[str, float]:
    """
    Compute along-track and cross-track errors in orbital frame.
    
    Args:
        predicted_states: Predicted states (..., 6)
        true_states: True states (..., 6)
        
    Returns:
        Dictionary with along-track, cross-track, and radial errors
    """
    # TODO: Implement RSW (radial-along-cross) frame transformation
    # For now, return placeholder
    return {
        'radial_rmse': 0.0,
        'along_track_rmse': 0.0,
        'cross_track_rmse': 0.0
    }

def orbital_energy_error(predicted_states: np.ndarray,
                        true_states: np.ndarray,
                        mu: float = 3.986004418e14) -> Dict[str, float]:
    """
    Compute orbital energy conservation errors.
    
    Args:
        predicted_states: Predicted states (..., 6)
        true_states: True states (..., 6) 
        mu: Gravitational parameter (m^3/s^2)
        
    Returns:
        Dictionary with energy error metrics
    """
    def compute_energy(states):
        r = states[..., :3]
        v = states[..., 3:]
        
        kinetic = 0.5 * np.sum(v**2, axis=-1)
        r_norm = np.linalg.norm(r, axis=-1)
        potential = -mu / r_norm
        
        return kinetic + potential
    
    energy_pred = compute_energy(predicted_states)
    energy_true = compute_energy(true_states)
    
    abs_error = np.abs(energy_pred - energy_true)
    rel_error = abs_error / np.abs(energy_true)
    
    return {
        'energy_mae': float(np.mean(abs_error)),
        'energy_rmse': float(np.sqrt(np.mean(abs_error**2))),
        'energy_relative_error': float(np.mean(rel_error))
    }

def angular_momentum_error(predicted_states: np.ndarray,
                          true_states: np.ndarray) -> Dict[str, float]:
    """
    Compute angular momentum conservation errors.
    
    Args:
        predicted_states: Predicted states (..., 6)
        true_states: True states (..., 6)
        
    Returns:
        Dictionary with angular momentum error metrics
    """
    def compute_angular_momentum_magnitude(states):
        r = states[..., :3] 
        v = states[..., 3:]
        
        h_vec = np.cross(r, v, axis=-1)
        return np.linalg.norm(h_vec, axis=-1)
    
    h_pred = compute_angular_momentum_magnitude(predicted_states)
    h_true = compute_angular_momentum_magnitude(true_states)
    
    abs_error = np.abs(h_pred - h_true)
    rel_error = abs_error / h_true
    
    return {
        'angular_momentum_mae': float(np.mean(abs_error)),
        'angular_momentum_rmse': float(np.sqrt(np.mean(abs_error**2))),
        'angular_momentum_relative_error': float(np.mean(rel_error))
    }

def trajectory_error_growth(predicted_states: np.ndarray,
                           true_states: np.ndarray,
                           times: np.ndarray,
                           horizon_hours: List[float] = [0.1, 1.0, 6.0, 24.0]) -> Dict[str, List[float]]:
    """
    Compute error growth over different prediction horizons.
    
    Args:
        predicted_states: Predicted trajectory (..., 6)
        true_states: True trajectory (..., 6)
        times: Time array (hours from initial time)
        horizon_hours: Horizons to evaluate at (hours)
        
    Returns:
        Dictionary with errors at each horizon
    """
    results = {
        'horizons_hours': horizon_hours,
        'position_rmse': [],
        'velocity_rmse': []
    }
    
    for horizon in horizon_hours:
        # Find closest time index
        time_idx = np.argmin(np.abs(times - horizon))
        
        if time_idx < len(predicted_states):
            pred_subset = predicted_states[:time_idx+1]
            true_subset = true_states[:time_idx+1]
            
            pos_rmse = position_rmse(pred_subset, true_subset)
            vel_rmse = velocity_rmse(pred_subset, true_subset)
            
            results['position_rmse'].append(pos_rmse)
            results['velocity_rmse'].append(vel_rmse)
        else:
            # Extrapolation beyond available data
            results['position_rmse'].append(np.nan)
            results['velocity_rmse'].append(np.nan)
    
    return results

def uncertainty_calibration_metrics(predictions: np.ndarray,
                                   uncertainties: np.ndarray,
                                   targets: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, float]:
    """
    Compute uncertainty calibration metrics.
    
    Args:
        predictions: Model predictions (N, D)
        uncertainties: Predicted uncertainties (N, D)
        targets: True targets (N, D)
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with calibration metrics
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    std_flat = uncertainties.flatten()
    target_flat = targets.flatten()
    
    # Remove invalid values
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(std_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[valid_mask]
    std_flat = std_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return {'nll': np.nan, 'calibration_error': np.nan, 'sharpness': np.nan}
    
    # Negative Log-Likelihood (assuming Gaussian)
    nll = 0.5 * np.mean((pred_flat - target_flat)**2 / std_flat**2 + np.log(2 * np.pi * std_flat**2))
    
    # Calibration error using reliability diagram
    errors = np.abs(pred_flat - target_flat)
    
    # Sort by predicted uncertainty
    sorted_indices = np.argsort(std_flat)
    sorted_errors = errors[sorted_indices]
    sorted_stds = std_flat[sorted_indices]
    
    bin_size = len(sorted_errors) // n_bins
    calibration_errors = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_errors)
        
        if end_idx > start_idx:
            bin_errors = sorted_errors[start_idx:end_idx]
            bin_stds = sorted_stds[start_idx:end_idx]
            
            # Expected vs observed error rate
            expected_error = np.mean(bin_stds)
            observed_error = np.mean(bin_errors)
            
            calibration_errors.append(abs(expected_error - observed_error))
    
    calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
    
    # Sharpness (average predicted uncertainty)
    sharpness = np.mean(std_flat)
    
    return {
        'nll': float(nll),
        'calibration_error': float(calibration_error),
        'sharpness': float(sharpness)
    }

def compute_comprehensive_metrics(predicted_states: np.ndarray,
                                 true_states: np.ndarray,
                                 times: np.ndarray,
                                 predicted_uncertainties: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict, List]]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predicted_states: Predicted trajectory
        true_states: True trajectory  
        times: Time array
        predicted_uncertainties: Optional uncertainty estimates
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Basic trajectory metrics
    metrics['position_rmse'] = position_rmse(predicted_states, true_states)
    metrics['velocity_rmse'] = velocity_rmse(predicted_states, true_states)
    
    # Physics-based metrics
    metrics.update(orbital_energy_error(predicted_states, true_states))
    metrics.update(angular_momentum_error(predicted_states, true_states))
    
    # Error growth analysis
    metrics['error_growth'] = trajectory_error_growth(
        predicted_states, true_states, times
    )
    
    # Frame-dependent errors
    metrics.update(along_track_cross_track_errors(predicted_states, true_states))
    
    # Uncertainty calibration (if available)
    if predicted_uncertainties is not None:
        metrics.update(uncertainty_calibration_metrics(
            predicted_states, predicted_uncertainties, true_states
        ))
    
    return metrics

def create_error_plots(predicted_states: np.ndarray,
                      true_states: np.ndarray, 
                      times: np.ndarray,
                      object_name: str = "Object") -> Dict[str, plt.Figure]:
    """
    Create comprehensive error analysis plots.
    
    Args:
        predicted_states: Predicted trajectory
        true_states: True trajectory
        times: Time array  
        object_name: Name for plot titles
        
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    # Position and velocity errors over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Trajectory Errors - {object_name}")
    
    # Position errors
    pos_errors = np.linalg.norm(predicted_states[:, :3] - true_states[:, :3], axis=1)
    axes[0, 0].plot(times, pos_errors, 'b-', linewidth=2)
    axes[0, 0].set_title('Position Error')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Position Error (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity errors  
    vel_errors = np.linalg.norm(predicted_states[:, 3:] - true_states[:, 3:], axis=1)
    axes[0, 1].plot(times, vel_errors, 'r-', linewidth=2)
    axes[0, 1].set_title('Velocity Error')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Velocity Error (m/s)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy conservation
    def orbital_energy(states):
        r = states[:, :3]
        v = states[:, 3:]
        kinetic = 0.5 * np.sum(v**2, axis=1)
        potential = -3.986004418e14 / np.linalg.norm(r, axis=1)
        return kinetic + potential
    
    energy_pred = orbital_energy(predicted_states)
    energy_true = orbital_energy(true_states)
    energy_error = np.abs(energy_pred - energy_true) / np.abs(energy_true)
    
    axes[1, 0].semilogy(times, energy_error, 'g-', linewidth=2)
    axes[1, 0].set_title('Relative Energy Error')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Relative Energy Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3D trajectory comparison
    axes[1, 1].remove()
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot trajectories (subsample for clarity)
    step = max(1, len(times) // 100)
    ax_3d.plot(true_states[::step, 0], true_states[::step, 1], true_states[::step, 2], 
               'b-', label='True', alpha=0.8, linewidth=2)
    ax_3d.plot(predicted_states[::step, 0], predicted_states[::step, 1], predicted_states[::step, 2],
               'r--', label='Predicted', alpha=0.8, linewidth=2)
    
    ax_3d.set_title('3D Trajectory')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)') 
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend()
    
    plt.tight_layout()
    figures['trajectory_errors'] = fig
    
    # Error growth plot
    horizons = [0.1, 0.5, 1.0, 2.0, 6.0, 12.0, 24.0]
    error_growth = trajectory_error_growth(predicted_states, true_states, times, horizons)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(error_growth['horizons_hours'], error_growth['position_rmse'], 
              'bo-', linewidth=2, markersize=6, label='Position RMSE')
    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title(f'Error Growth - {object_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    figures['error_growth'] = fig
    
    return figures