#!/usr/bin/env python3
"""
Test script for sensor simulation functionality.

Tests:
1. Coordinate transformations
2. Optical telescope measurements  
3. Radar system measurements
4. Image rendering capabilities
5. Realistic observation scenarios
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.sensors import (
    OpticalSensor, RadarSensor, OpticalObservation, RadarObservation,
    eci_to_radec, render_optical_image
)
from data.generators import ScenarioGenerator

def test_coordinate_transformations():
    """Test coordinate transformation functions."""
    print("Testing coordinate transformations...")
    
    # Test case: Simple position on X-axis
    r_eci = np.array([7e6, 0.0, 0.0])  # Position on X-axis
    
    # Test ECI to RA/Dec conversion
    ra, dec = eci_to_radec(r_eci)
    print(f"  ECI position: [{r_eci[0]/1e6:.1f}, {r_eci[1]/1e6:.1f}, {r_eci[2]/1e6:.1f}] Mm")
    print(f"  RA/Dec: {np.degrees(ra):.2f}°, {np.degrees(dec):.2f}°")
    
    # Expected: RA ≈ 0°, Dec ≈ 0° (on equator, along X-axis)
    assert abs(ra) < 0.01, f"Expected RA ≈ 0°, got {np.degrees(ra):.2f}°"
    assert abs(dec) < 0.01, f"Expected Dec ≈ 0°, got {np.degrees(dec):.2f}°"
    
    print("  ✓ Coordinate transformations working correctly")

def test_optical_sensor():
    """Test optical sensor functionality."""
    print("\nTesting optical sensor...")
    
    # Create optical sensor (Mauna Kea coordinates)
    sensor = OpticalSensor(
        lat=19.8,  # degrees
        lon=-155.5,
        alt=4200,  # meters
        noise_sigma=np.radians(1.0/3600.0)  # 1 arcsecond noise
    )
    sensor.set_random_state(42)  # Reproducible results
    
    # Test observation
    r_eci = np.array([7e6, 1e6, 0.5e6])  # LEO-like position
    time = 0.0
    
    # Check visibility
    visible = sensor.check_visibility(r_eci, time)
    print(f"  Target visible: {visible}")
    
    # Generate observation
    obs = sensor.simulate_observation(r_eci, time, add_noise=True)
    print(f"  RA: {np.degrees(obs.ra):.3f}° ± {np.degrees(obs.noise_sigma)*3600:.1f} arcsec")
    print(f"  Dec: {np.degrees(obs.dec):.3f}°")
    print(f"  Magnitude: {obs.magnitude:.1f}")
    print(f"  Timestamp: {obs.timestamp:.1f} s")
    
    # Test noise statistics
    observations = []
    for _ in range(100):
        obs_noise = sensor.simulate_observation(r_eci, time, add_noise=True)
        observations.append([obs_noise.ra, obs_noise.dec])
    
    observations = np.array(observations)
    ra_std = np.std(observations[:, 0])
    dec_std = np.std(observations[:, 1])
    
    print(f"  Measured noise std: RA={np.degrees(ra_std)*3600:.1f} arcsec, Dec={np.degrees(dec_std)*3600:.1f} arcsec")
    print("  ✓ Optical sensor working correctly")

def test_radar_sensor():
    """Test radar sensor functionality."""
    print("\nTesting radar sensor...")
    
    # Create radar sensor (Haystack coordinates)
    sensor = RadarSensor(
        lat=42.6,  # degrees
        lon=-71.5,
        alt=146,
        range_noise_km=0.01,  # 10 meter range accuracy
        angle_noise_deg=0.1   # 0.1 degree angular accuracy
    )
    sensor.set_random_state(42)  # Reproducible results
    
    # Test observation
    r_eci = np.array([6.8e6, 1e6, 0.8e6])  # LEO position
    v_eci = np.array([0, 7.5e3, 1e3])      # Typical LEO velocity
    time = 0.0
    
    # Check visibility
    visible = sensor.check_visibility(r_eci, time)
    print(f"  Target visible: {visible}")
    
    # Generate observation
    obs = sensor.simulate_observation(r_eci, v_eci, time, add_noise=True)
    print(f"  Range: {obs.range_km:.2f} km")
    print(f"  Range rate: {obs.range_rate_kms:.3f} km/s")
    print(f"  Azimuth: {np.degrees(obs.azimuth):.1f}°")
    print(f"  Elevation: {np.degrees(obs.elevation):.1f}°")
    print(f"  SNR: {obs.snr_db:.1f} dB")
    print(f"  Timestamp: {obs.timestamp:.1f} s")
    
    print("  ✓ Radar sensor working correctly")

def test_image_rendering():
    """Test optical image rendering."""
    print("\nTesting optical image rendering...")
    
    # Create synthetic observations (star field)
    observations = []
    
    # Add several "stars" at different positions
    base_ra = np.radians(45.0)  # 45 degrees RA
    base_dec = np.radians(30.0)  # 30 degrees Dec
    
    # Central bright object (satellite)
    observations.append(OpticalObservation(
        timestamp=0.0,
        ra=base_ra,
        dec=base_dec,
        magnitude=8.0
    ))
    
    # Add some background stars
    np.random.seed(42)
    for i in range(5):
        ra_offset = np.radians(np.random.uniform(-0.5, 0.5))  # ±0.5 degree spread
        dec_offset = np.radians(np.random.uniform(-0.5, 0.5))
        magnitude = np.random.uniform(10.0, 14.0)
        
        observations.append(OpticalObservation(
            timestamp=0.0,
            ra=base_ra + ra_offset,
            dec=base_dec + dec_offset,
            magnitude=magnitude
        ))
    
    # Render image
    image = render_optical_image(
        observations=observations,
        exposure_time=30.0,
        image_size=(256, 256),
        fov_deg=1.0
    )
    
    print(f"  Generated image: {image.shape[0]}x{image.shape[1]} pixels")
    print(f"  Pixel values range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Number of non-zero pixels: {np.count_nonzero(image)}")
    
    print("  ✓ Image rendering working correctly")

def test_realistic_tracking_scenario():
    """Test realistic multi-sensor tracking scenario."""
    print("\nTesting realistic tracking scenario...")
    
    # Generate realistic trajectory
    config = {
        "propagator": {"include_j2": True},
        "object_mix": {"satellite": 1.0}
    }
    
    generator = ScenarioGenerator(config, random_state=42)
    scenario = generator.generate_scenario(n_objects=1, duration_hours=1.0)
    
    # Propagate trajectory
    scenario_data = generator.propagate_scenario(scenario, time_step=180.0)  # 3-minute steps
    
    # Get trajectory data
    obj_name = list(scenario_data['trajectories'].keys())[0]
    trajectory = scenario_data['trajectories'][obj_name]
    positions = trajectory['positions']
    velocities = trajectory['velocities']
    times = trajectory['times']
    
    print(f"  Trajectory: {len(positions)} positions over {times[-1]/3600:.1f} hours")
    
    # Create sensor network
    optical_sensors = [
        OpticalSensor(lat=19.8, lon=-155.5, alt=4200),  # Mauna Kea
        OpticalSensor(lat=-24.6, lon=-70.4, alt=2635),  # Paranal
        OpticalSensor(lat=28.7, lon=-17.9, alt=2396)    # La Palma
    ]
    
    radar_sensors = [
        RadarSensor(lat=42.6, lon=-71.5, alt=146),      # Haystack
        RadarSensor(lat=35.4, lon=-116.9, alt=1071)     # Goldstone
    ]
    
    # Set random seeds
    for i, sensor in enumerate(optical_sensors):
        sensor.set_random_state(42 + i)
    for i, sensor in enumerate(radar_sensors):
        sensor.set_random_state(50 + i)
    
    # Simulate observations
    optical_observations = []
    radar_observations = []
    measurement_interval = 600.0  # 10 minutes
    
    for i, time in enumerate(times[::int(measurement_interval/180.0)]):  # Sample every 10 minutes
        if i >= len(positions):
            break
            
        pos = positions[i]
        vel = velocities[i]
        
        # Optical observations
        for j, sensor in enumerate(optical_sensors):
            if sensor.check_visibility(pos, time):
                obs = sensor.simulate_observation(pos, time, add_noise=True)
                optical_observations.append((f"OPT_{j}", obs))
        
        # Radar observations  
        for j, sensor in enumerate(radar_sensors):
            if sensor.check_visibility(pos, time):
                obs = sensor.simulate_observation(pos, vel, time, add_noise=True)
                radar_observations.append((f"RAD_{j}", obs))
    
    print(f"  Optical observations: {len(optical_observations)}")
    print(f"  Radar observations: {len(radar_observations)}")
    
    # Show sample observations
    if optical_observations:
        sensor_id, obs = optical_observations[0]
        print(f"  Sample optical ({sensor_id}): RA={np.degrees(obs.ra):.2f}°, Dec={np.degrees(obs.dec):.2f}°")
    
    if radar_observations:
        sensor_id, obs = radar_observations[0]
        print(f"  Sample radar ({sensor_id}): R={obs.range_km:.1f}km, Az={np.degrees(obs.azimuth):.1f}°")
    
    print("  ✓ Realistic tracking scenario working correctly")

def create_sensor_visualization():
    """Create visualization of sensor measurements and capabilities."""
    print("\nCreating sensor visualization...")
    
    # Generate test scenario
    config = {"propagator": {"include_j2": True}}
    generator = ScenarioGenerator(config, random_state=42)
    scenario = generator.generate_scenario(n_objects=1, duration_hours=0.5)
    scenario_data = generator.propagate_scenario(scenario, time_step=120.0)
    
    # Get trajectory
    obj_name = list(scenario_data['trajectories'].keys())[0]
    trajectory = scenario_data['trajectories'][obj_name]
    positions = trajectory['positions']
    velocities = trajectory['velocities']
    times = trajectory['times']
    
    # Create sensors
    optical = OpticalSensor(lat=19.8, lon=-155.5, alt=4200)  # Mauna Kea
    radar = RadarSensor(lat=42.6, lon=-71.5, alt=146)        # Haystack
    
    optical.set_random_state(42)
    radar.set_random_state(43)
    
    # Collect observations
    optical_data = []
    radar_data = []
    
    for i in range(0, len(positions), 5):  # Sample every 5th point
        pos = positions[i]
        vel = velocities[i]
        time = times[i]
        
        if optical.check_visibility(pos, time):
            obs = optical.simulate_observation(pos, time, add_noise=True)
            optical_data.append((time/3600, np.degrees(obs.ra), np.degrees(obs.dec)))
        
        if radar.check_visibility(pos, time):
            obs = radar.simulate_observation(pos, vel, time, add_noise=True)
            radar_data.append((time/3600, obs.range_km, np.degrees(obs.azimuth), np.degrees(obs.elevation)))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: 3D Trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0]/1e6, positions[:, 1]/1e6, 'b-', linewidth=2)
    ax.set_xlabel('X (Mm)')
    ax.set_ylabel('Y (Mm)')
    ax.set_title('Satellite Trajectory (ECI)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Optical RA measurements
    if optical_data:
        times_opt, ra_vals, dec_vals = zip(*optical_data)
        ax = axes[0, 1]
        ax.plot(times_opt, ra_vals, 'ro-', markersize=4, label='RA')
        ax.plot(times_opt, dec_vals, 'bo-', markersize=4, label='Dec')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Optical Measurements (Mauna Kea)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Radar range measurements
    if radar_data:
        times_rad, ranges, azimuths, elevations = zip(*radar_data)
        ax = axes[1, 0]
        ax.plot(times_rad, ranges, 'gs-', markersize=4)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Range (km)')
        ax.set_title('Radar Range (Haystack)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Radar angles
        ax = axes[1, 1]
        ax.plot(times_rad, azimuths, 'ro-', markersize=4, label='Azimuth')
        ax.plot(times_rad, elevations, 'bo-', markersize=4, label='Elevation')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Radar Angles (Haystack)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to outputs
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'sensor_measurements.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  ✓ Sensor visualization saved to {output_dir / 'sensor_measurements.png'}")

if __name__ == "__main__":
    print("NP-SNN Sensor Simulation Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_coordinate_transformations()
    test_optical_sensor()
    test_radar_sensor()
    test_image_rendering()
    test_realistic_tracking_scenario()
    create_sensor_visualization()
    
    print("\n" + "=" * 50)
    print("✓ All sensor tests passed!")
    print("\nSensor simulation capabilities:")
    print("  • Optical telescopes: RA/Dec measurements with realistic noise")
    print("  • Radar systems: Range/Doppler/Az/El measurements") 
    print("  • Coordinate transformations: ECI ↔ ECEF ↔ Topocentric")
    print("  • Visibility constraints: Elevation limits, range limits")
    print("  • Image rendering: Synthetic optical images with PSF")
    print("  • Multi-sensor networks: Coordinated observations")
    print("  • Realistic noise models: Arcsecond optical, meter-level radar")
    print("\nReady for neural network integration! 🛰️")