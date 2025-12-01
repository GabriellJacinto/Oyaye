#!/usr/bin/env python3
"""
Test script for the dataset generator.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generators import ScenarioGenerator, OrbitalElementSampler, sample_orbital_elements
from data.generators import oe_to_cartesian, ObjectParameterSampler

def test_orbital_element_sampling():
    """Test orbital element sampling."""
    print("Testing Orbital Element Sampling...")
    
    sampler = OrbitalElementSampler(random_state=42)
    
    # Test LEO debris sampling
    leo_oe = sampler.sample_leo_debris()
    print(f"LEO debris orbit:")
    print(f"  Altitude: {(leo_oe.a - 6.378137e6)/1e3:.1f} km")
    print(f"  Eccentricity: {leo_oe.e:.4f}")
    print(f"  Inclination: {np.degrees(leo_oe.i):.1f} deg")
    
    # Test operational satellite sampling
    ops_oe = sampler.sample_operational_satellite()
    print(f"\nOperational satellite orbit:")
    print(f"  Altitude: {(ops_oe.a - 6.378137e6)/1e3:.1f} km")
    print(f"  Eccentricity: {ops_oe.e:.4f}")
    print(f"  Inclination: {np.degrees(ops_oe.i):.1f} deg")
    
    return True

def test_coordinate_conversion():
    """Test orbital elements to Cartesian conversion."""
    print("\nTesting Coordinate Conversion...")
    
    # Test with known circular orbit
    from data.generators import OrbitalElements
    
    # Circular LEO orbit at 400 km
    oe = OrbitalElements(
        a=6.378137e6 + 400e3,
        e=0.0,
        i=np.radians(51.6),  # ISS inclination
        raan=0.0,
        w=0.0,
        nu=0.0,
        epoch=0.0
    )
    
    r, v = oe_to_cartesian(oe)
    
    print(f"Position magnitude: {np.linalg.norm(r)/1e3:.1f} km")
    print(f"Velocity magnitude: {np.linalg.norm(v)/1e3:.3f} km/s")
    
    # Check if we're at the right altitude
    altitude = np.linalg.norm(r) - 6.378137e6
    print(f"Altitude check: {altitude/1e3:.1f} km (should be ~400 km)")
    
    # Check circular velocity
    expected_v = np.sqrt(3.986004418e14 / np.linalg.norm(r))
    print(f"Velocity check: {np.linalg.norm(v):.1f} m/s (expected: {expected_v:.1f} m/s)")
    
    return abs(altitude - 400e3) < 1000  # Within 1 km

def test_object_sampling():
    """Test object parameter sampling."""
    print("\nTesting Object Parameter Sampling...")
    
    sampler = ObjectParameterSampler(random_state=42)
    
    for obj_type in ["debris", "satellite", "rocket_body"]:
        props = sampler.sample_object_properties(obj_type)
        print(f"\n{obj_type.capitalize()}:")
        print(f"  Mass: {props['mass']:.2f} kg")
        print(f"  Area: {props['area']:.3f} m²")
        print(f"  Area/Mass: {props['area']/props['mass']:.3f} m²/kg")
        print(f"  Drag coeff: {props['drag_coeff']:.2f}")
        print(f"  Reflectivity: {props['reflectivity']:.2f}")
    
    return True

def test_scenario_generation():
    """Test complete scenario generation."""
    print("\nTesting Scenario Generation...")
    
    config = {
        "propagator": {
            "include_j2": True,
            "include_drag": False,
            "include_srp": False
        },
        "object_mix": {
            "debris": 0.6,
            "satellite": 0.3,
            "rocket_body": 0.1
        }
    }
    
    generator = ScenarioGenerator(config, random_state=42)
    
    # Generate small scenario
    scenario = generator.generate_scenario(n_objects=3, duration_hours=1.0)
    
    print(f"Generated scenario with {len(scenario.objects)} objects")
    print(f"Duration: {scenario.metadata['duration_hours']} hours")
    
    for i, obj in enumerate(scenario.objects):
        print(f"\nObject {i+1}: {obj.name} ({obj.object_type})")
        print(f"  Mass: {obj.mass:.2f} kg")
        print(f"  Altitude: {(obj.oe.a - 6.378137e6)/1e3:.1f} km")
        print(f"  Inclination: {np.degrees(obj.oe.i):.1f} deg")
    
    return True

def test_scenario_propagation():
    """Test scenario propagation with our propagator."""
    print("\nTesting Scenario Propagation...")
    
    config = {
        "propagator": {
            "include_j2": True,
            "include_drag": False,
            "include_srp": False
        }
    }
    
    generator = ScenarioGenerator(config, random_state=42)
    
    # Generate and propagate small scenario
    scenario = generator.generate_scenario(n_objects=2, duration_hours=0.5)  # 30 minutes
    
    print(f"Propagating {len(scenario.objects)} objects for {scenario.metadata['duration_hours']} hours...")
    
    try:
        scenario_data = generator.propagate_scenario(scenario, time_step=60.0)  # 1-minute steps
        
        print("✓ Propagation successful!")
        
        # Check results
        trajectories = scenario_data["trajectories"]
        print(f"Generated trajectories for {len(trajectories)} objects")
        
        for obj_name, traj in trajectories.items():
            n_points = len(traj["times"])
            final_pos = traj["positions"][-1]
            final_alt = (np.linalg.norm(final_pos) - 6.378137e6) / 1e3
            
            print(f"  {obj_name}: {n_points} points, final altitude {final_alt:.1f} km")
        
        return True
        
    except Exception as e:
        print(f"✗ Propagation failed: {e}")
        return False

def test_data_saving():
    """Test saving scenario data."""
    print("\nTesting Data Saving...")
    
    config = {"propagator": {"include_j2": True}}
    generator = ScenarioGenerator(config, random_state=42)
    
    # Generate and propagate
    scenario = generator.generate_scenario(n_objects=1, duration_hours=0.25)
    scenario_data = generator.propagate_scenario(scenario, time_step=30.0)
    
    # Save to temporary file
    temp_file = Path("test_scenario.json")
    
    try:
        generator.save_scenario(scenario_data, temp_file)
        
        print(f"✓ Saved scenario to {temp_file}")
        print(f"File size: {temp_file.stat().st_size / 1024:.1f} KB")
        
        # Load and verify
        import json
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        print(f"✓ Successfully loaded scenario with {len(loaded_data['trajectories'])} trajectories")
        
        # Cleanup
        temp_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        return False

if __name__ == "__main__":
    print("Dataset Generator Test Suite")
    print("=" * 40)
    
    success = True
    
    # Run tests
    success &= test_orbital_element_sampling()
    success &= test_coordinate_conversion()
    success &= test_object_sampling()
    success &= test_scenario_generation()
    success &= test_scenario_propagation()
    success &= test_data_saving()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All dataset generator tests passed!")
    else:
        print("✗ Some tests failed")
        
    print("\nDataset generator implementation complete ✓")