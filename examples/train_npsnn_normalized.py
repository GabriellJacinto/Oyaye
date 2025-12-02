"""
Training script for NP-SNN with proper data normalization.

Fixed training pipeline that normalizes orbital data to reasonable scales
to prevent massive loss values and numerical instability.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import yaml
from torch.utils.data import random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.npsnn import NPSNN
from src.data.orbital_dataset import OrbitalDataset, TrajectoryTransforms
from src.data.orbital_generator import OrbitalDataGenerator
from src.train.config import TrainingConfig
from src.train.trainer import NPSNNTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def setup_device() -> torch.device:
    """Setup training device with CUDA optimization if available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU: {torch.get_num_threads()} threads")
    
    return device

def normalize_trajectory_batch(trajectories, position_scale=1e7, velocity_scale=1e4):
    """Apply normalization to a batch of trajectories."""
    normalized_trajectories = []
    for traj in trajectories:
        normalized_traj = TrajectoryTransforms.normalize_states(
            traj, 
            position_scale=position_scale,
            velocity_scale=velocity_scale
        )
        normalized_trajectories.append(normalized_traj)
    return normalized_trajectories

def create_datasets(config: TrainingConfig, device: torch.device) -> tuple:
    """Create training and validation datasets with proper normalization."""
    print("📊 Generating normalized orbital datasets...")
    
    # Initialize data generator
    data_generator = OrbitalDataGenerator(
        n_objects=config.dataset_config.n_objects_train,
        time_horizon_hours=24.0,  # Will be adapted by curriculum
        dt_minutes=config.dataset_config.dt_minutes,
        observation_noise_std=0.001,  # Will be adapted by curriculum
        device=device
    )
    
    # Generate training trajectories
    print(f"   Generating {config.dataset_config.n_trajectories_train:,} training trajectories...")
    train_trajectories = []
    for i in range(0, config.dataset_config.n_trajectories_train, 50):  # Generate in smaller batches
        batch_size = min(50, config.dataset_config.n_trajectories_train - i)
        batch_trajectories = data_generator.generate_batch(
            batch_size=batch_size,
            add_observation_gaps=True,
            gap_probability=0.3
        )
        
        # Apply normalization to this batch
        normalized_batch = normalize_trajectory_batch(batch_trajectories)
        train_trajectories.extend(normalized_batch)
        
        if (i + batch_size) % 100 == 0:
            print(f"      Generated and normalized {i + batch_size:,}/{config.dataset_config.n_trajectories_train:,} training trajectories")
    
    # Generate validation trajectories
    print(f"   Generating {config.dataset_config.n_trajectories_val:,} validation trajectories...")
    val_trajectories = []
    for i in range(0, config.dataset_config.n_trajectories_val, 20):
        batch_size = min(20, config.dataset_config.n_trajectories_val - i)
        batch_trajectories = data_generator.generate_batch(
            batch_size=batch_size,
            add_observation_gaps=True,
            gap_probability=0.2  # Less challenging gaps for validation
        )
        
        # Apply normalization to validation batch
        normalized_batch = normalize_trajectory_batch(batch_trajectories)
        val_trajectories.extend(normalized_batch)
    
    # Create datasets
    train_dataset = OrbitalDataset(train_trajectories)
    val_dataset = OrbitalDataset(val_trajectories)
    
    print(f"✅ Normalized datasets created:")
    print(f"   Training: {len(train_dataset):,} trajectories")
    print(f"   Validation: {len(val_dataset):,} trajectories")
    
    # Print normalization statistics
    sample_traj = train_trajectories[0]
    print(f"   Data scale: positions ±{sample_traj['states'][:, :3].abs().max().item():.3f}, velocities ±{sample_traj['states'][:, 3:6].abs().max().item():.3f}")
    
    return train_dataset, val_dataset

def create_model(config: TrainingConfig, device: torch.device) -> NPSNN:
    """Create and initialize NP-SNN model."""
    print("🧠 Creating NP-SNN model...")
    
    # Import the NPSNN config classes
    from src.models.npsnn import NPSNNConfig, TimeEncodingConfig, SNNConfig, DecoderConfig
    
    # Create NPSNNConfig from training config
    npsnn_config = NPSNNConfig()
    
    # Time encoding configuration
    npsnn_config.time_encoding = TimeEncodingConfig()
    npsnn_config.time_encoding.d_model = config.model_config.hidden_size
    
    # Input/output dimensions
    npsnn_config.obs_input_size = 6  # [x, y, z, vx, vy, vz]
    npsnn_config.obs_encoding_dim = config.model_config.hidden_size // 2
    
    # SNN core configuration
    npsnn_config.snn = SNNConfig()
    npsnn_config.snn.num_layers = config.model_config.num_layers
    npsnn_config.snn.hidden_size = config.model_config.hidden_size
    npsnn_config.snn.dropout = config.model_config.dropout
    
    # Decoder configuration  
    npsnn_config.decoder = DecoderConfig()
    npsnn_config.decoder.output_size = 6  # Same as input
    
    # Create model
    model = NPSNN(npsnn_config)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Hidden size: {config.model_config.hidden_size}")
    print(f"   Layers: {config.model_config.num_layers}")
    
    return model

def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Convert YAML config to TrainingConfig
    # This is a simplified conversion - in practice you'd want more robust parsing
    training_config = TrainingConfig.from_dict(yaml_config)
    
    return training_config

def main():
    parser = argparse.ArgumentParser(description='Train NP-SNN with normalized data')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Override experiment name from config')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with smaller datasets')
    
    args = parser.parse_args()
    
    print("🚀 NP-SNN Normalized Training Pipeline")
    print("=" * 60)
    
    # Load configuration
    print(f"📋 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config.experiment_config.experiment_name = args.experiment_name
    
    # Debug mode adjustments
    if args.debug:
        print("🐛 Debug mode enabled - using smaller datasets")
        config.dataset_config.n_trajectories_train = 100
        config.dataset_config.n_trajectories_val = 20
        config.curriculum_config.total_epochs = 10
    
    # Setup device
    device = setup_device()
    
    # Set random seeds for reproducibility
    seed = config.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"🎲 Random seed set to: {seed}")
    
    try:
        # Create datasets
        train_dataset, val_dataset = create_datasets(config, device)
        
        # Create model
        model = create_model(config, device)
        
        # Create trainer
        print("🏋️  Initializing trainer...")
        trainer = NPSNNTrainer(
            config=config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device
        )
        
        # Start training
        print("\n" + "=" * 60)
        print("🎯 Starting normalized training...")
        print("=" * 60)
        
        results = trainer.train(resume_from_checkpoint=args.resume)
        
        # Print results
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"Status: {results['status']}")
        print(f"Total epochs: {results.get('total_epochs', 'N/A')}")
        print(f"Best validation loss: {results.get('best_val_loss', 'N/A'):.6f}")
        print(f"Final stage: {results.get('final_stage', 'N/A')}")
        print(f"MLflow run ID: {results.get('run_id', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()