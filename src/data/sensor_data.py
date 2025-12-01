"""
Sensor data management for the NP-SNN project.

This module provides:
- Structured measurement collection and storage
- Data persistence for neural network training
- Quality control and validation
- Export utilities for training pipelines
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from .sensors import OpticalObservation, RadarObservation

@dataclass
class TrackingPass:
    """Container for a complete tracking pass of an object."""
    object_id: str
    start_time: float
    end_time: float
    optical_measurements: List[Dict]
    radar_measurements: List[Dict]
    true_trajectory: Optional[Dict] = None
    
    def duration_hours(self) -> float:
        """Get duration of tracking pass in hours."""
        return (self.end_time - self.start_time) / 3600.0
    
    def total_measurements(self) -> int:
        """Get total number of measurements."""
        return len(self.optical_measurements) + len(self.radar_measurements)

class SensorDataManager:
    """
    Manages sensor measurement collection and persistence for neural network training.
    
    Key features:
    - Structured data collection from multiple sensor types
    - Quality control and validation
    - Persistent storage in JSON format
    - Export utilities for training pipelines
    - Statistical analysis and reporting
    """
    
    def __init__(self, output_dir: str = "outputs/sensor_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.tracking_passes: List[TrackingPass] = []
        self.sensor_configs: Dict = {}
        self.metadata = {
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'NP-SNN sensor measurement dataset'
        }
    
    def add_sensor_config(self, sensor_id: str, sensor_type: str, config: Dict):
        """Register sensor configuration."""
        self.sensor_configs[sensor_id] = {
            'type': sensor_type,
            'config': config,
            'added': datetime.now().isoformat()
        }
    
    def start_tracking_pass(self, object_id: str, start_time: float) -> str:
        """Start a new tracking pass for an object."""
        pass_id = f"{object_id}_{int(start_time)}"
        return pass_id
    
    def add_optical_measurement(self, object_id: str, sensor_id: str, 
                              observation: OpticalObservation,
                              true_state: Optional[Dict] = None):
        """Add optical measurement to current tracking pass."""
        measurement = {
            'sensor_id': sensor_id,
            'timestamp': observation.timestamp,
            'ra': observation.ra,
            'dec': observation.dec,
            'magnitude': observation.magnitude,
            'noise_sigma': observation.noise_sigma,
            'measurement_type': 'optical'
        }
        
        if true_state:
            measurement['true_state'] = true_state
        
        # Find or create tracking pass
        tracking_pass = self._get_or_create_pass(object_id, observation.timestamp)
        tracking_pass.optical_measurements.append(measurement)
    
    def add_radar_measurement(self, object_id: str, sensor_id: str,
                            observation: RadarObservation, 
                            true_state: Optional[Dict] = None):
        """Add radar measurement to current tracking pass."""
        measurement = {
            'sensor_id': sensor_id,
            'timestamp': observation.timestamp,
            'range_km': observation.range_km,
            'range_rate_kms': observation.range_rate_kms,
            'azimuth': observation.azimuth,
            'elevation': observation.elevation,
            'snr_db': observation.snr_db,
            'measurement_type': 'radar'
        }
        
        if true_state:
            measurement['true_state'] = true_state
        
        # Find or create tracking pass
        tracking_pass = self._get_or_create_pass(object_id, observation.timestamp)
        tracking_pass.radar_measurements.append(measurement)
    
    def _get_or_create_pass(self, object_id: str, timestamp: float) -> TrackingPass:
        """Get existing tracking pass or create new one."""
        # Look for existing pass
        for pass_obj in self.tracking_passes:
            if (pass_obj.object_id == object_id and 
                pass_obj.start_time <= timestamp <= pass_obj.end_time + 3600):  # 1 hour tolerance
                # Update end time
                pass_obj.end_time = max(pass_obj.end_time, timestamp)
                return pass_obj
        
        # Create new pass
        new_pass = TrackingPass(
            object_id=object_id,
            start_time=timestamp,
            end_time=timestamp,
            optical_measurements=[],
            radar_measurements=[]
        )
        self.tracking_passes.append(new_pass)
        return new_pass
    
    def add_true_trajectory(self, object_id: str, trajectory_data: Dict):
        """Add ground truth trajectory data for validation."""
        for pass_obj in self.tracking_passes:
            if pass_obj.object_id == object_id:
                pass_obj.true_trajectory = trajectory_data
                break
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {
            'tracking_passes': len(self.tracking_passes),
            'total_objects': len(set(p.object_id for p in self.tracking_passes)),
            'total_optical': sum(len(p.optical_measurements) for p in self.tracking_passes),
            'total_radar': sum(len(p.radar_measurements) for p in self.tracking_passes),
            'total_measurements': 0,
            'sensors': {
                'total': len(self.sensor_configs),
                'optical': len([s for s in self.sensor_configs.values() if s['type'] == 'optical']),
                'radar': len([s for s in self.sensor_configs.values() if s['type'] == 'radar'])
            },
            'time_coverage': {
                'min_time': 0,
                'max_time': 0,
                'total_hours': 0
            },
            'measurement_density': {}
        }
        
        stats['total_measurements'] = stats['total_optical'] + stats['total_radar']
        
        # Time coverage analysis
        if self.tracking_passes:
            all_times = []
            for pass_obj in self.tracking_passes:
                all_times.extend([m['timestamp'] for m in pass_obj.optical_measurements])
                all_times.extend([m['timestamp'] for m in pass_obj.radar_measurements])
            
            if all_times:
                stats['time_coverage']['min_time'] = min(all_times)
                stats['time_coverage']['max_time'] = max(all_times)
                stats['time_coverage']['total_hours'] = (max(all_times) - min(all_times)) / 3600.0
        
        # Measurement density by sensor
        sensor_counts = {}
        for pass_obj in self.tracking_passes:
            for measurement in pass_obj.optical_measurements + pass_obj.radar_measurements:
                sensor_id = measurement['sensor_id']
                sensor_counts[sensor_id] = sensor_counts.get(sensor_id, 0) + 1
        
        stats['measurement_density'] = sensor_counts
        
        return stats
    
    def save_dataset(self, filename: Optional[str] = None) -> Path:
        """Save complete dataset to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_dataset_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare data for serialization
        dataset = {
            'metadata': {
                **self.metadata,
                'statistics': self.get_statistics(),
                'saved': datetime.now().isoformat()
            },
            'sensors': self.sensor_configs,
            'tracking_passes': [asdict(pass_obj) for pass_obj in self.tracking_passes]
        }
        
        # Custom JSON encoder for numpy arrays
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, cls=NumpyEncoder)
        
        return filepath
    
    def load_dataset(self, filepath: Union[str, Path]) -> None:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.sensor_configs = data['sensors']
        
        # Reconstruct tracking passes
        self.tracking_passes = []
        for pass_data in data['tracking_passes']:
            tracking_pass = TrackingPass(**pass_data)
            self.tracking_passes.append(tracking_pass)
    
    def export_for_training(self, output_format: str = 'numpy') -> Dict:
        """
        Export dataset in format suitable for neural network training.
        
        Args:
            output_format: 'numpy', 'pytorch', or 'tensorflow'
            
        Returns:
            Dictionary containing training-ready data
        """
        # Collect all measurements with timestamps
        sequences = []
        
        for pass_obj in self.tracking_passes:
            # Combine all measurements for this object
            all_measurements = []
            
            # Add optical measurements
            for meas in pass_obj.optical_measurements:
                measurement_vector = [
                    meas['timestamp'],
                    meas['ra'],
                    meas['dec'],
                    1.0,  # measurement type indicator (1=optical)
                    meas.get('magnitude', 0.0),
                    meas['noise_sigma']
                ]
                all_measurements.append((meas['timestamp'], measurement_vector, meas.get('true_state')))
            
            # Add radar measurements
            for meas in pass_obj.radar_measurements:
                measurement_vector = [
                    meas['timestamp'],
                    meas['range_km'] * 1000.0,  # convert to meters
                    meas['azimuth'],
                    2.0,  # measurement type indicator (2=radar)
                    meas['elevation'],
                    meas['range_rate_kms'] * 1000.0  # convert to m/s
                ]
                all_measurements.append((meas['timestamp'], measurement_vector, meas.get('true_state')))
            
            # Sort by timestamp
            all_measurements.sort(key=lambda x: x[0])
            
            if all_measurements:
                sequences.append({
                    'object_id': pass_obj.object_id,
                    'measurements': [m[1] for m in all_measurements],
                    'times': [m[0] for m in all_measurements],
                    'true_states': [m[2] for m in all_measurements if m[2] is not None]
                })
        
        training_data = {
            'sequences': sequences,
            'metadata': self.get_statistics(),
            'format': output_format
        }
        
        if output_format == 'numpy':
            # Convert to numpy arrays
            for seq in training_data['sequences']:
                seq['measurements'] = np.array(seq['measurements'])
                seq['times'] = np.array(seq['times'])
        
        return training_data
    
    def print_summary(self):
        """Print human-readable dataset summary."""
        stats = self.get_statistics()
        
        print("Dataset Summary:")
        print("=" * 40)
        print(f"Objects tracked: {stats['total_objects']}")
        print(f"Tracking passes: {stats['tracking_passes']}")
        print(f"Total measurements: {stats['total_measurements']}")
        print(f"  - Optical: {stats['total_optical']}")
        print(f"  - Radar: {stats['total_radar']}")
        print(f"Time coverage: {stats['time_coverage']['total_hours']:.1f} hours")
        print(f"Sensors: {stats['sensors']['total']} ({stats['sensors']['optical']} optical, {stats['sensors']['radar']} radar)")
        
        print("\nMeasurement density by sensor:")
        for sensor_id, count in stats['measurement_density'].items():
            print(f"  {sensor_id}: {count}")
        
        print("\nTracking pass details:")
        for pass_obj in self.tracking_passes:
            duration = pass_obj.duration_hours()
            total = pass_obj.total_measurements()
            print(f"  {pass_obj.object_id}: {total} measurements over {duration:.1f}h")

# Convenience function for quick dataset creation
def create_sensor_dataset(scenario_data: Dict, sensors: Dict, 
                         measurement_interval: float = 600.0,
                         output_dir: str = "outputs/sensor_data") -> SensorDataManager:
    """
    Create sensor dataset from orbital scenario and sensor network.
    
    Args:
        scenario_data: Output from ScenarioGenerator.propagate_scenario()
        sensors: Dictionary of sensor objects {sensor_id: sensor}
        measurement_interval: Time between measurements (seconds)
        output_dir: Output directory for dataset files
    
    Returns:
        SensorDataManager with collected measurements
    """
    manager = SensorDataManager(output_dir)
    
    # Register sensor configurations
    for sensor_id, sensor in sensors.items():
        sensor_type = 'optical' if hasattr(sensor, 'noise_sigma') else 'radar'
        config = {
            'lat': np.degrees(sensor.lat),
            'lon': np.degrees(sensor.lon), 
            'alt': sensor.alt,
            'min_elevation': np.degrees(sensor.min_elevation)
        }
        manager.add_sensor_config(sensor_id, sensor_type, config)
    
    # Process each object trajectory
    for obj_name, trajectory in scenario_data['trajectories'].items():
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        times = trajectory['times']
        
        # Sample measurements at specified intervals
        measurement_indices = range(0, len(times), int(measurement_interval / (times[1] - times[0])))
        
        for idx in measurement_indices:
            if idx >= len(positions):
                break
            
            pos = positions[idx]
            vel = velocities[idx]
            time = times[idx]
            
            # True state for validation
            true_state = {
                'position': pos.tolist(),
                'velocity': vel.tolist(),
                'time': time
            }
            
            # Collect measurements from all visible sensors
            for sensor_id, sensor in sensors.items():
                if sensor.check_visibility(pos, time):
                    try:
                        if hasattr(sensor, 'noise_sigma'):  # Optical sensor
                            obs = sensor.simulate_observation(pos, time, add_noise=True)
                            manager.add_optical_measurement(obj_name, sensor_id, obs, true_state)
                        else:  # Radar sensor
                            obs = sensor.simulate_observation(pos, vel, time, add_noise=True)
                            manager.add_radar_measurement(obj_name, sensor_id, obs, true_state)
                    except Exception as e:
                        print(f"Warning: Measurement failed for {sensor_id}: {e}")
        
        # Add true trajectory
        manager.add_true_trajectory(obj_name, {
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'times': times.tolist()
        })
    
    return manager