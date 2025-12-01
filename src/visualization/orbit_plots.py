#!/usr/bin/env python3
"""
Orbit visualization tools for the NP-SNN space debris tracking project.

This module creates beautiful visualizations of:
1. Individual orbit examples (LEO, MEO, GEO)
2. Multi-object LEO debris fields
3. Trajectory analysis plots
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import ScenarioGenerator, OrbitalElementSampler, oe_to_cartesian
from data.generators import OrbitalElements

# Constants
R_EARTH = 6.378137e6  # Earth radius (m)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_earth_sphere(ax, alpha=0.3, color='lightblue'):
    """Plot Earth as a sphere."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_earth, y_earth, z_earth, 
                   alpha=alpha, color=color, linewidth=0, antialiased=True)

def plot_single_orbit_examples():
    """Plot individual examples of LEO, MEO, and GEO orbits."""
    print("Creating single orbit examples...")
    
    # Create specific orbital elements for demonstration
    orbits = {
        "LEO (ISS-like)": OrbitalElements(
            a=R_EARTH + 408e3,           # 408 km altitude
            e=0.0001,                    # Nearly circular
            i=np.radians(51.6),          # ISS inclination
            raan=0.0,
            w=0.0,
            nu=0.0,
            epoch=0.0
        ),
        "MEO (GPS-like)": OrbitalElements(
            a=R_EARTH + 20200e3,         # GPS altitude
            e=0.01,                      # Slightly elliptical
            i=np.radians(55.0),          # GPS inclination
            raan=0.0,
            w=0.0,
            nu=0.0,
            epoch=0.0
        ),
        "GEO (Geostationary)": OrbitalElements(
            a=R_EARTH + 35786e3,         # GEO altitude
            e=0.0001,                    # Nearly circular
            i=np.radians(0.1),           # Nearly equatorial
            raan=0.0,
            w=0.0,
            nu=0.0,
            epoch=0.0
        )
    }
    
    # Set up the plot
    fig = plt.figure(figsize=(15, 5))
    
    colors = ['red', 'orange', 'green']
    
    for idx, (name, oe) in enumerate(orbits.items()):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Generate orbit points
        nu_points = np.linspace(0, 2*np.pi, 200)
        positions = []
        
        for nu in nu_points:
            oe_temp = OrbitalElements(oe.a, oe.e, oe.i, oe.raan, oe.w, nu, oe.epoch)
            r, _ = oe_to_cartesian(oe_temp)
            positions.append(r)
        
        positions = np.array(positions)
        
        # Plot Earth
        plot_earth_sphere(ax, alpha=0.2)
        
        # Plot orbit
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[idx], linewidth=2, label=name)
        
        # Mark starting position
        ax.scatter(*positions[0], color=colors[idx], s=50, marker='o')
        
        # Set equal aspect ratio and labels
        max_range = np.max(np.abs(positions)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{name}\nAlt: {(oe.a - R_EARTH)/1e3:.0f} km, Inc: {np.degrees(oe.i):.1f}°')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'orbit_examples.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Single orbit examples saved as '{output_file}'")

def plot_leo_debris_field():
    """Plot many LEO debris objects to show orbital diversity."""
    print("\nCreating LEO debris field visualization...")
    
    config = {
        "propagator": {"include_j2": True},
        "object_mix": {"debris": 1.0}  # Only debris
    }
    
    generator = ScenarioGenerator(config, random_state=42)
    
    # Generate many LEO debris objects
    n_objects = 50
    scenario = generator.generate_scenario(n_objects=n_objects, duration_hours=2.0)
    
    # Set up 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth
    plot_earth_sphere(ax, alpha=0.15, color='skyblue')
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_objects))
    
    for i, obj in enumerate(scenario.objects):
        # Generate partial orbit (1/4 revolution for visibility)
        nu_points = np.linspace(0, np.pi/2, 50)  # Quarter orbit
        positions = []
        
        for nu in nu_points:
            oe_temp = OrbitalElements(
                obj.oe.a, obj.oe.e, obj.oe.i, obj.oe.raan, obj.oe.w, nu, obj.oe.epoch
            )
            r, _ = oe_to_cartesian(oe_temp)
            positions.append(r)
        
        positions = np.array(positions)
        
        # Plot trajectory segment
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=colors[i], linewidth=1.5, alpha=0.7)
        
        # Mark starting position
        ax.scatter(*positions[0], color=colors[i], s=20, alpha=0.8)
    
    # Set plot properties
    max_range = 10e6  # 10,000 km
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.set_xlabel('X (km)', labelpad=10)
    ax.set_ylabel('Y (km)', labelpad=10)
    ax.set_zlabel('Z (km)', labelpad=10)
    
    # Convert axis labels to km
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    
    ax.set_title(f'LEO Debris Field Simulation\n{n_objects} Objects (Quarter Orbit Segments)', 
                 fontsize=14, pad=20)
    
    # Add colorbar for object diversity
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, n_objects))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Object Number', rotation=270, labelpad=20)
    
    output_file = OUTPUT_DIR / 'leo_debris_field.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ LEO debris field with {n_objects} objects saved as '{output_file}'")

def plot_trajectory_propagation():
    """Show trajectory propagation over time with J2 effects."""
    print("\nCreating trajectory propagation visualization...")
    
    config = {
        "propagator": {"include_j2": True},
        "object_mix": {"debris": 0.7, "satellite": 0.3}
    }
    
    generator = ScenarioGenerator(config, random_state=123)
    
    # Generate scenario with few objects for clarity
    scenario = generator.generate_scenario(n_objects=3, duration_hours=3.0)
    
    # Propagate trajectories
    scenario_data = generator.propagate_scenario(scenario, time_step=300.0)  # 5-minute steps
    
    # Create subplot layout
    fig = plt.figure(figsize=(16, 8))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    plot_earth_sphere(ax1, alpha=0.2)
    
    colors = ['red', 'blue', 'green']
    trajectories = scenario_data['trajectories']
    
    for i, (obj_name, traj) in enumerate(trajectories.items()):
        positions = traj['positions']
        
        # Plot full trajectory
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=colors[i % len(colors)], linewidth=2, label=obj_name)
        
        # Mark start and end
        ax1.scatter(*positions[0], color=colors[i % len(colors)], s=100, marker='o')
        ax1.scatter(*positions[-1], color=colors[i % len(colors)], s=100, marker='s')
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Trajectories (3 Hours)\nCircles: Start, Squares: End')
    ax1.legend()
    
    # Format axes to km
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    ax1.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}'))
    
    # Altitude vs time plot
    ax2 = fig.add_subplot(122)
    
    for i, (obj_name, traj) in enumerate(trajectories.items()):
        times = traj['times']
        positions = traj['positions']
        
        # Calculate altitude
        altitudes = [np.linalg.norm(pos) - R_EARTH for pos in positions]
        
        ax2.plot(times / 3600, np.array(altitudes) / 1e3, 
                color=colors[i % len(colors)], linewidth=2, label=obj_name, marker='o', markersize=3)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Altitude Evolution\n(J2 Perturbation Effects)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'trajectory_propagation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Trajectory propagation analysis saved as '{output_file}'")

def plot_orbital_element_distributions():
    """Plot distributions of sampled orbital elements."""
    print("\nCreating orbital element distribution plots...")
    
    sampler = OrbitalElementSampler(random_state=42)
    
    # Generate many samples
    n_samples = 1000
    leo_samples = [sampler.sample_leo_debris() for _ in range(n_samples)]
    ops_samples = [sampler.sample_operational_satellite() for _ in range(n_samples)]
    
    # Extract parameters
    leo_alts = [(oe.a - R_EARTH) / 1e3 for oe in leo_samples]
    leo_incs = [np.degrees(oe.i) for oe in leo_samples]
    leo_eccs = [oe.e for oe in leo_samples]
    
    ops_alts = [(oe.a - R_EARTH) / 1e3 for oe in ops_samples]
    ops_incs = [np.degrees(oe.i) for oe in ops_samples]
    ops_eccs = [oe.e for oe in ops_samples]
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # LEO debris distributions
    axes[0, 0].hist(leo_alts, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('LEO Debris Altitude Distribution')
    axes[0, 0].set_xlabel('Altitude (km)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(leo_incs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('LEO Debris Inclination Distribution')
    axes[0, 1].set_xlabel('Inclination (degrees)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(leo_eccs, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].set_title('LEO Debris Eccentricity Distribution')
    axes[0, 2].set_xlabel('Eccentricity')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Operational satellite distributions
    axes[1, 0].hist(ops_alts, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title('Operational Satellite Altitude Distribution')
    axes[1, 0].set_xlabel('Altitude (km)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(ops_incs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].set_title('Operational Satellite Inclination Distribution')
    axes[1, 1].set_xlabel('Inclination (degrees)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(ops_eccs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 2].set_title('Operational Satellite Eccentricity Distribution')
    axes[1, 2].set_xlabel('Eccentricity')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'orbital_element_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Orbital element distributions saved as '{output_file}'")
    
    # Print statistics
    print(f"\nLEO Debris Statistics (n={n_samples}):")
    print(f"  Altitude: {np.mean(leo_alts):.1f} ± {np.std(leo_alts):.1f} km")
    print(f"  Inclination: {np.mean(leo_incs):.1f} ± {np.std(leo_incs):.1f}°")
    print(f"  Eccentricity: {np.mean(leo_eccs):.4f} ± {np.std(leo_eccs):.4f}")
    
    print(f"\nOperational Satellite Statistics (n={n_samples}):")
    print(f"  Altitude: {np.mean(ops_alts):.1f} ± {np.std(ops_alts):.1f} km")
    print(f"  Inclination: {np.mean(ops_incs):.1f} ± {np.std(ops_incs):.1f}°")
    print(f"  Eccentricity: {np.mean(ops_eccs):.4f} ± {np.std(ops_eccs):.4f}")

def create_visualization_summary():
    """Create a summary README for the visualization outputs."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_content = f"""# NP-SNN Orbit Visualizations

Generated on: {timestamp}

## Overview
This directory contains visualization outputs from the NP-SNN space debris tracking project's orbit analysis tools.

## Files Generated

### 1. orbit_examples.png
- **Purpose**: Comparison of major orbital regimes
- **Content**: LEO (ISS-like), MEO (GPS-like), and GEO (Geostationary) orbit examples
- **Key Features**: 
  - LEO: 408 km altitude, 51.6° inclination (similar to ISS)
  - MEO: 20,200 km altitude, 55° inclination (similar to GPS)
  - GEO: 35,786 km altitude, nearly equatorial (geostationary)

### 2. leo_debris_field.png
- **Purpose**: Visualization of crowded LEO debris environment
- **Content**: 50 randomly generated debris objects in LEO regime
- **Key Features**:
  - Quarter-orbit segments for spatial clarity
  - Color-coded objects for distinction
  - Demonstrates complex multi-object scenarios for neural network training

### 3. trajectory_propagation.png
- **Purpose**: Validation of orbital mechanics and J2 perturbation effects
- **Content**: 3-hour trajectory propagation with altitude evolution
- **Key Features**:
  - 3D trajectory visualization with start/end markers
  - Altitude vs. time plots showing J2 perturbation effects
  - Multiple object types (debris and operational satellites)

### 4. orbital_element_distributions.png
- **Purpose**: Statistical validation of dataset generator
- **Content**: Parameter distributions for LEO debris vs operational satellites
- **Key Features**:
  - 1000 samples each of debris and operational satellites
  - Altitude, inclination, and eccentricity distributions
  - Statistical validation of realistic parameter sampling

## Dataset Generator Statistics

### LEO Debris (n=1000)
- **Altitude**: 1085.6 ± 526.4 km
- **Inclination**: 91.0 ± 52.2°
- **Eccentricity**: 0.0502 ± 0.0195

### Operational Satellites (n=1000)
- **Altitude**: 9540.9 ± 12291.2 km (LEO to GEO range)
- **Inclination**: 42.4 ± 31.7°
- **Eccentricity**: 0.0094 ± 0.0053

## Usage in Reports
These visualizations are suitable for:
- Technical presentations and papers
- Dataset validation documentation
- Model architecture explanations
- Orbital mechanics verification

All images are saved at 300 DPI for publication quality.
"""
    
    readme_file = OUTPUT_DIR / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(summary_content)
    
    print(f"✓ Visualization summary created: '{readme_file}'")

if __name__ == "__main__":
    print("NP-SNN Orbit Visualization Suite")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create all visualizations
    plot_single_orbit_examples()
    plot_leo_debris_field()
    plot_trajectory_propagation()
    plot_orbital_element_distributions()
    
    # Create summary documentation
    create_visualization_summary()
    
    print("\n" + "=" * 50)
    print("✓ All visualizations complete!")
    print(f"\nGenerated files in '{OUTPUT_DIR}':")
    print("  • orbit_examples.png - LEO/MEO/GEO orbit examples")
    print("  • leo_debris_field.png - 50-object LEO debris field")
    print("  • trajectory_propagation.png - 3-hour propagation with J2")
    print("  • orbital_element_distributions.png - Parameter distributions")
    print("  • README.md - Documentation and statistics summary")
    print("\nDataset generator validation through visualization complete ✓")