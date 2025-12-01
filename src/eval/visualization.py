"""
Advanced visualization tools for NP-SNN evaluation.

This module provides:
- Trajectory visualization in 3D and projected views
- Uncertainty quantification plots
- Physics violation analysis
- Interactive orbit plots
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.animation as animation
from matplotlib.patches import Circle
import warnings

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class TrajectoryVisualizer:
    """Advanced trajectory visualization tools."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.earth_radius = 6.371e6  # Earth radius in meters
        
    def plot_3d_trajectory_comparison(self, 
                                    predicted_states: np.ndarray,
                                    true_states: np.ndarray,
                                    times: np.ndarray,
                                    title: str = "Orbital Trajectory Comparison",
                                    save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create 3D trajectory comparison plot.
        
        Args:
            predicted_states: Predicted trajectory (N, 6)
            true_states: True trajectory (N, 6)
            times: Time array (hours)
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract positions (convert to km for visualization)
        pos_pred = predicted_states[:, :3] / 1000
        pos_true = true_states[:, :3] / 1000
        
        # Plot Earth (sphere)
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        earth_x = self.earth_radius/1000 * np.cos(u) * np.sin(v)
        earth_y = self.earth_radius/1000 * np.sin(u) * np.sin(v)
        earth_z = self.earth_radius/1000 * np.cos(v)
        
        ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue')
        
        # Plot trajectories with time-based coloring
        n_points = len(times)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        # Subsample for performance
        step = max(1, n_points // 200)
        indices = np.arange(0, n_points, step)
        
        for i in indices[:-1]:
            # True trajectory
            ax.plot(pos_true[i:i+2, 0], pos_true[i:i+2, 1], pos_true[i:i+2, 2],
                   color='blue', alpha=0.8, linewidth=2)
            
            # Predicted trajectory  
            ax.plot(pos_pred[i:i+2, 0], pos_pred[i:i+2, 1], pos_pred[i:i+2, 2],
                   color='red', alpha=0.8, linewidth=2, linestyle='--')
        
        # Mark start and end points
        ax.scatter(*pos_true[0], color='green', s=100, label='Start')
        ax.scatter(*pos_true[-1], color='red', s=100, label='End')
        
        # Set equal aspect ratio
        max_range = np.max(np.abs(pos_true)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='True Trajectory'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Predicted Trajectory'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Start'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='End')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ground_track(self,
                         predicted_states: np.ndarray,
                         true_states: np.ndarray,
                         times: np.ndarray,
                         title: str = "Ground Track Comparison",
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot ground track (latitude/longitude) comparison.
        
        Args:
            predicted_states: Predicted trajectory (N, 6)  
            true_states: True trajectory (N, 6)
            times: Time array
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        def cartesian_to_latlon(positions):
            """Convert cartesian to lat/lon."""
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            
            # Convert to spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            lat = np.arcsin(z / r) * 180 / np.pi
            lon = np.arctan2(y, x) * 180 / np.pi
            
            return lat, lon
        
        # Convert to lat/lon
        lat_true, lon_true = cartesian_to_latlon(true_states[:, :3])
        lat_pred, lon_pred = cartesian_to_latlon(predicted_states[:, :3])
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Plot world map outline (simplified)
        world_lon = np.linspace(-180, 180, 100)
        equator_lat = np.zeros_like(world_lon)
        
        ax.plot(world_lon, equator_lat, 'k--', alpha=0.3, label='Equator')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        
        # Plot trajectories
        ax.plot(lon_true, lat_true, 'b-', linewidth=2, label='True Ground Track', alpha=0.8)
        ax.plot(lon_pred, lat_pred, 'r--', linewidth=2, label='Predicted Ground Track', alpha=0.8)
        
        # Mark start and end
        ax.scatter(lon_true[0], lat_true[0], color='green', s=100, zorder=5, label='Start')
        ax.scatter(lon_true[-1], lat_true[-1], color='red', s=100, zorder=5, label='End')
        
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable limits
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_evolution(self,
                                 predicted_states: np.ndarray,
                                 uncertainties: np.ndarray,
                                 true_states: np.ndarray,
                                 times: np.ndarray,
                                 title: str = "Uncertainty Evolution",
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot uncertainty evolution with confidence bounds.
        
        Args:
            predicted_states: Predicted states (N, 6)
            uncertainties: Predicted uncertainties (N, 6)
            true_states: True states (N, 6)
            times: Time array (hours)
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        state_labels = ['X (m)', 'Y (m)', 'Z (m)', 'VX (m/s)', 'VY (m/s)', 'VZ (m/s)']
        
        for i in range(6):
            ax = axes[i//3, i%3]
            
            # Predictions with uncertainty bounds
            pred = predicted_states[:, i]
            true = true_states[:, i]
            std = uncertainties[:, i]
            
            # Plot true values
            ax.plot(times, true, 'b-', linewidth=2, label='True', alpha=0.8)
            
            # Plot predictions with uncertainty
            ax.plot(times, pred, 'r--', linewidth=2, label='Predicted', alpha=0.8)
            ax.fill_between(times, pred - 2*std, pred + 2*std, 
                          color='red', alpha=0.2, label='±2σ Uncertainty')
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(state_labels[i])
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Add legend to first subplot
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_physics_violations(self,
                              predicted_states: np.ndarray,
                              true_states: np.ndarray, 
                              times: np.ndarray,
                              title: str = "Physics Violation Analysis",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot physics violations (energy, angular momentum).
        
        Args:
            predicted_states: Predicted states (N, 6)
            true_states: True states (N, 6)
            times: Time array (hours)
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        def compute_orbital_energy(states):
            """Compute specific orbital energy."""
            r = states[:, :3]
            v = states[:, 3:]
            
            kinetic = 0.5 * np.sum(v**2, axis=1)
            r_norm = np.linalg.norm(r, axis=1)
            potential = -3.986004418e14 / r_norm  # GM_earth
            
            return kinetic + potential
        
        def compute_angular_momentum_magnitude(states):
            """Compute angular momentum magnitude."""
            r = states[:, :3]
            v = states[:, 3:]
            
            h = np.cross(r, v, axis=1)
            return np.linalg.norm(h, axis=1)
        
        # Compute physics quantities
        energy_pred = compute_orbital_energy(predicted_states)
        energy_true = compute_orbital_energy(true_states)
        
        momentum_pred = compute_angular_momentum_magnitude(predicted_states)
        momentum_true = compute_angular_momentum_magnitude(true_states)
        
        # Compute relative errors
        energy_rel_error = np.abs(energy_pred - energy_true) / np.abs(energy_true)
        momentum_rel_error = np.abs(momentum_pred - momentum_true) / momentum_true
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Energy evolution
        axes[0, 0].plot(times, energy_true, 'b-', linewidth=2, label='True Energy', alpha=0.8)
        axes[0, 0].plot(times, energy_pred, 'r--', linewidth=2, label='Predicted Energy', alpha=0.8)
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Specific Energy (J/kg)')
        axes[0, 0].set_title('Orbital Energy Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Angular momentum evolution
        axes[0, 1].plot(times, momentum_true, 'b-', linewidth=2, label='True |h|', alpha=0.8)
        axes[0, 1].plot(times, momentum_pred, 'r--', linewidth=2, label='Predicted |h|', alpha=0.8)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Angular Momentum Magnitude (m²/s)')
        axes[0, 1].set_title('Angular Momentum Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy conservation error
        axes[1, 0].semilogy(times, energy_rel_error, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Relative Energy Error')
        axes[1, 0].set_title('Energy Conservation Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Angular momentum conservation error
        axes[1, 1].semilogy(times, momentum_rel_error, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Relative Angular Momentum Error')
        axes[1, 1].set_title('Angular Momentum Conservation Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residual_analysis(self,
                             predicted_states: np.ndarray,
                             true_states: np.ndarray,
                             times: np.ndarray,
                             title: str = "Residual Analysis",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot detailed residual analysis.
        
        Args:
            predicted_states: Predicted states (N, 6)
            true_states: True states (N, 6)
            times: Time array (hours)
            title: Plot title
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Compute residuals
        residuals = predicted_states - true_states
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        state_labels = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        
        # Time series residuals
        for i in range(6):
            row = i // 3
            col = i % 3
            
            axes[row, col].plot(times, residuals[:, i], 'b-', linewidth=1.5, alpha=0.8)
            axes[row, col].set_xlabel('Time (hours)')
            axes[row, col].set_ylabel(f'{state_labels[i]} Residual')
            axes[row, col].set_title(f'{state_labels[i]} Residuals vs Time')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add zero line
            axes[row, col].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Overall position and velocity residual magnitudes
        pos_residual_norm = np.linalg.norm(residuals[:, :3], axis=1)
        vel_residual_norm = np.linalg.norm(residuals[:, 3:], axis=1)
        
        axes[2, 0].plot(times, pos_residual_norm, 'g-', linewidth=2)
        axes[2, 0].set_xlabel('Time (hours)')
        axes[2, 0].set_ylabel('Position Residual Norm (m)')
        axes[2, 0].set_title('Position Residual Magnitude')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(times, vel_residual_norm, 'orange', linewidth=2)
        axes[2, 1].set_xlabel('Time (hours)')
        axes[2, 1].set_ylabel('Velocity Residual Norm (m/s)')
        axes[2, 1].set_title('Velocity Residual Magnitude')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Residual distribution (histogram)
        axes[2, 2].hist(pos_residual_norm, bins=30, alpha=0.7, color='green', 
                       label='Position', density=True)
        axes[2, 2].hist(vel_residual_norm, bins=30, alpha=0.7, color='orange',
                       label='Velocity', density=True)
        axes[2, 2].set_xlabel('Residual Magnitude')
        axes[2, 2].set_ylabel('Density')
        axes[2, 2].set_title('Residual Distribution')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_animation(self,
                        predicted_states: np.ndarray,
                        true_states: np.ndarray,
                        times: np.ndarray,
                        save_path: Optional[Path] = None,
                        fps: int = 10) -> animation.FuncAnimation:
        """
        Create animated trajectory comparison.
        
        Args:
            predicted_states: Predicted trajectory (N, 6)
            true_states: True trajectory (N, 6)
            times: Time array
            save_path: Optional save path for animation
            fps: Frames per second
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert to km
        pos_pred = predicted_states[:, :3] / 1000
        pos_true = true_states[:, :3] / 1000
        
        # Initialize empty lines
        line_true, = ax.plot([], [], [], 'b-', linewidth=2, label='True')
        line_pred, = ax.plot([], [], [], 'r--', linewidth=2, label='Predicted')
        point_true, = ax.plot([], [], [], 'bo', markersize=8)
        point_pred, = ax.plot([], [], [], 'ro', markersize=8)
        
        # Set up the axes
        max_range = np.max(np.abs(pos_true)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.legend()
        
        def animate(frame):
            # Update trajectory lines
            line_true.set_data_3d(pos_true[:frame+1, 0], 
                                 pos_true[:frame+1, 1], 
                                 pos_true[:frame+1, 2])
            line_pred.set_data_3d(pos_pred[:frame+1, 0], 
                                 pos_pred[:frame+1, 1], 
                                 pos_pred[:frame+1, 2])
            
            # Update current position points
            if frame < len(pos_true):
                point_true.set_data_3d([pos_true[frame, 0]], 
                                     [pos_true[frame, 1]], 
                                     [pos_true[frame, 2]])
                point_pred.set_data_3d([pos_pred[frame, 0]], 
                                     [pos_pred[frame, 1]], 
                                     [pos_pred[frame, 2]])
            
            ax.set_title(f'Orbital Trajectory (t = {times[frame]:.2f} hours)')
            
            return line_true, line_pred, point_true, point_pred
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                     interval=1000//fps, blit=True, repeat=True)
        
        if save_path:
            # Save as MP4 (requires ffmpeg)
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
                print(f"Animation saved to: {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
                # Fallback: save as GIF
                try:
                    gif_path = save_path.with_suffix('.gif')
                    anim.save(gif_path, writer='pillow', fps=fps)
                    print(f"Animation saved as GIF: {gif_path}")
                except Exception as e2:
                    print(f"Could not save GIF either: {e2}")
        
        return anim


def create_comprehensive_visualization_report(predicted_states: np.ndarray,
                                           true_states: np.ndarray,
                                           uncertainties: Optional[np.ndarray],
                                           times: np.ndarray,
                                           output_dir: Path,
                                           object_name: str = "Trajectory") -> None:
    """
    Create comprehensive visualization report with all plots.
    
    Args:
        predicted_states: Predicted trajectory
        true_states: True trajectory
        uncertainties: Optional uncertainties
        times: Time array
        output_dir: Output directory
        object_name: Name for the trajectory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz = TrajectoryVisualizer()
    
    print(f"🎨 Creating visualization report for {object_name}...")
    
    # 3D trajectory comparison
    fig1 = viz.plot_3d_trajectory_comparison(
        predicted_states, true_states, times,
        title=f"{object_name} - 3D Trajectory Comparison",
        save_path=output_dir / "3d_trajectory_comparison.png"
    )
    plt.close(fig1)
    
    # Ground track
    fig2 = viz.plot_ground_track(
        predicted_states, true_states, times,
        title=f"{object_name} - Ground Track Comparison", 
        save_path=output_dir / "ground_track_comparison.png"
    )
    plt.close(fig2)
    
    # Physics violations
    fig3 = viz.plot_physics_violations(
        predicted_states, true_states, times,
        title=f"{object_name} - Physics Conservation Analysis",
        save_path=output_dir / "physics_violations.png"
    )
    plt.close(fig3)
    
    # Residual analysis
    fig4 = viz.plot_residual_analysis(
        predicted_states, true_states, times,
        title=f"{object_name} - Residual Analysis",
        save_path=output_dir / "residual_analysis.png"
    )
    plt.close(fig4)
    
    # Uncertainty evolution (if available)
    if uncertainties is not None:
        fig5 = viz.plot_uncertainty_evolution(
            predicted_states, uncertainties, true_states, times,
            title=f"{object_name} - Uncertainty Evolution",
            save_path=output_dir / "uncertainty_evolution.png"
        )
        plt.close(fig5)
    
    print(f"✅ Visualization report saved to: {output_dir}")


def plot_calibration_diagram(predictions: np.ndarray,
                           uncertainties: np.ndarray,
                           targets: np.ndarray,
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create uncertainty calibration (reliability) diagram.
    
    Args:
        predictions: Model predictions
        uncertainties: Predicted uncertainties
        targets: True targets
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
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
        ax1.text(0.5, 0.5, 'No valid data for calibration', 
                ha='center', va='center', transform=ax1.transAxes)
        return fig
    
    # Compute errors
    errors = np.abs(pred_flat - target_flat)
    
    # Reliability diagram
    n_bins = 10
    bin_boundaries = np.percentile(std_flat, np.linspace(0, 100, n_bins + 1))
    
    bin_centers = []
    observed_frequencies = []
    expected_frequencies = []
    
    for i in range(n_bins):
        mask = (std_flat >= bin_boundaries[i]) & (std_flat < bin_boundaries[i+1])
        
        if mask.sum() > 0:
            bin_errors = errors[mask]
            bin_stds = std_flat[mask]
            
            # Expected frequency (mean predicted std)
            expected_freq = np.mean(bin_stds)
            
            # Observed frequency (mean actual error)
            observed_freq = np.mean(bin_errors)
            
            bin_centers.append(expected_freq)
            expected_frequencies.append(expected_freq)
            observed_frequencies.append(observed_freq)
    
    # Plot reliability diagram
    ax1.plot([0, max(expected_frequencies)], [0, max(expected_frequencies)], 
             'k--', alpha=0.5, label='Perfect Calibration')
    ax1.scatter(expected_frequencies, observed_frequencies, 
               s=60, alpha=0.8, color='red', label='Binned Data')
    ax1.set_xlabel('Expected Error (Predicted σ)')
    ax1.set_ylabel('Observed Error (Actual |Error|)')
    ax1.set_title('Uncertainty Calibration Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sharpness vs calibration scatter
    ax2.scatter(std_flat[::100], errors[::100], alpha=0.3, s=10)  # Subsample for speed
    ax2.plot([0, np.max(std_flat)], [0, np.max(std_flat)], 'k--', alpha=0.5)
    ax2.set_xlabel('Predicted Uncertainty (σ)')
    ax2.set_ylabel('Actual Error |ŷ - y|')
    ax2.set_title('Sharpness vs Calibration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig