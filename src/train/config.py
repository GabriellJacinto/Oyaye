"""
Training configuration system for NP-SNN orbital mechanics training.

This module provides comprehensive configuration classes for all aspects of
training: model architecture, curriculum stages, loss weighting, optimization,
and experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import yaml
import os

class TrainingStage(Enum):
    """Training curriculum stages as defined in implementation plan."""
    SANITY = "sanity"           # Stage 0: Overfit tiny dataset
    SUPERVISED = "supervised"    # Stage 1: Supervised pretraining  
    MIXED = "mixed"             # Stage 2: Mixed supervised + physics
    PHYSICS = "physics"         # Stage 3: Physics-dominant
    FINE_TUNE = "fine_tune"     # Stage 4: Fine-tuning & adaptation

class LossSchedule(Enum):
    """Loss weight scheduling strategies."""
    STATIC = "static"           # Fixed weights
    EXPONENTIAL = "exponential" # Exponential decay/growth
    LINEAR = "linear"           # Linear interpolation
    COSINE = "cosine"          # Cosine annealing
    STEP = "step"              # Step-wise changes

@dataclass
class CurriculumConfig:
    """Curriculum learning configuration per implementation plan section 4.1."""
    
    # Total training epochs 
    total_epochs: int = 200
    
    # Stage durations (in epochs)
    sanity_epochs: int = 5
    supervised_epochs: int = 50
    mixed_epochs: int = 80
    physics_epochs: int = 50
    fine_tune_epochs: int = 15
    
    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-6
    
    # Progressive schedules within stages
    supervised_decay_schedule: str = "exponential"
    supervised_decay_rate: float = 2.0
    dynamics_growth_schedule: str = "linear"
    dynamics_growth_rate: float = 1.0
    
    # Data complexity progression
    start_horizon_hours: float = 2.0
    end_horizon_hours: float = 24.0
    start_noise_level: float = 0.0001
    end_noise_level: float = 0.01
    
    # Collocation strategy
    collocation_points_per_gap: int = 3
    max_collocation_points: int = 50
    
    # Loss weight schedules for each stage - simplified
    stage_configs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "sanity": {"w_state_supervised": 1.0, "w_dynamics": 0.0},
        "supervised": {"w_state_supervised": 1.0, "w_dynamics": 0.0},
        "mixed": {"w_state_supervised": 0.5, "w_dynamics": 0.5},
        "physics": {"w_state_supervised": 0.1, "w_dynamics": 5.0},
        "fine_tune": {"w_state_supervised": 0.1, "w_dynamics": 2.0}
    })

@dataclass 
class OptimizationConfig:
    """Optimization configuration per implementation plan section 4.3."""
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "exponential", "plateau"
    lr_warmup_epochs: int = 10
    lr_min: float = 1e-6
    lr_decay_rate: float = 0.95
    lr_step_size: int = 100
    
    # Gradient handling
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Advanced optimization
    use_mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation

@dataclass
class DataConfig:
    """Data loading and batching configuration."""
    
    # Batching strategy per implementation plan section 4.4
    batch_size: int = 32
    val_batch_size: int = 64
    
    # Loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Sequence handling
    max_sequence_length: int = 100
    
    # Data augmentation
    noise_augmentation: bool = True
    temporal_jitter: float = 0.01     # Fraction of time step
    dropout_augmentation: float = 0.1  # Observation dropout rate

@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration."""
    
    # Experiment identification
    experiment_name: str = "npsnn_orbital_mechanics"
    run_name: Optional[str] = None    # Auto-generated if None
    
    # MLflow settings
    tracking_uri: Optional[str] = None
    
    # Logging and checkpoints
    log_frequency: int = 1
    checkpoint_frequency: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # What to log
    log_gradients: bool = True
    log_model_parameters: bool = True
    log_predictions: bool = True
    log_loss_components: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Output paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    visualization_dir: str = "visualizations"

@dataclass
class HyperparameterSearchConfig:
    """Hyperparameter search configuration per implementation plan section 4.3."""
    
    # Search strategy
    search_strategy: str = "random"  # "random", "grid", "bayesian"
    max_trials: int = 50
    timeout_hours: float = 48.0
    
    # Search spaces (for random/bayesian search)
    search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "model": {
            "time_feat_dim": [16, 32, 64, 128],
            "snn_hidden": [64, 128, 256], 
            "snn_layers": [1, 2, 3],
            "surrogate_slope": [5, 25, 50]
        },
        "optimization": {
            "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
            "batch_size": [8, 16, 32],
            "weight_decay": [0, 1e-5, 1e-4]
        },
        "loss": {
            "w_measurement": [0.1, 1.0, 3.0],
            "w_dynamics": [0.1, 1.0, 5.0, 10.0],
            "w_energy": [0.0, 0.01, 0.05, 0.2]
        }
    })
    
    # Pruning (for bayesian optimization)
    use_pruning: bool = True
    pruning_patience: int = 10

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    n_objects_train: int = 1000
    n_objects_val: int = 200
    n_trajectories_train: int = 10000
    n_trajectories_val: int = 2000
    dt_minutes: float = 5.0

@dataclass
class PhysicsConfig:
    """Physics loss configuration."""
    adaptive_weights: bool = True
    weight_update_frequency: int = 10
    reference_loss_window: int = 50
    
    # Physics constants
    earth_mu: float = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
    earth_radius: float = 6.371e6     # Earth's mean radius (m)
    
    # Loss weights
    energy_weight: float = 1.0
    momentum_weight: float = 1.0
    dynamics_weight: float = 10.0
    temporal_weight: float = 0.1
    measurement_weight: float = 1.0
    optical_weight: float = 1.0
    radar_weight: float = 1.0
    
    # Scaling factors
    energy_scale: float = 1e8         # Energy scaling for numerical stability
    angular_momentum_scale: float = 1e10  # Angular momentum scaling
    momentum_scale: float = 1e10      # Momentum scaling alias
    
    # Adaptive weighting
    use_adaptive_weights: bool = True  # Enable learned log-variance balancing
    use_gradnorm: bool = False        # Enable GradNorm balancing

@dataclass 
class TrainingConfig:
    """Master training configuration combining all components."""
    
    # Sub-configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    physics_config: PhysicsConfig = field(default_factory=PhysicsConfig)
    hyperparameter_search_config: Optional[HyperparameterSearchConfig] = None
    
    # Training control
    random_seed: int = 42
    mixed_precision: bool = False
    compile_model: bool = False
    
    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu"
    distributed_training: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary (e.g., loaded from YAML)."""
        
        # Model config
        model_config = ModelConfig(**(config_dict.get('model', {})))
        
        # Dataset config  
        dataset_config = DatasetConfig(**(config_dict.get('dataset', {})))
        
        # Curriculum config
        curriculum_dict = config_dict.get('curriculum', {})
        # Ensure stage_configs exists
        if 'stage_configs' not in curriculum_dict:
            curriculum_dict['stage_configs'] = {
                "sanity": {"w_state_supervised": 1.0, "w_dynamics": 0.0},
                "supervised": {"w_state_supervised": 1.0, "w_dynamics": 0.0},
                "mixed": {"w_state_supervised": 0.5, "w_dynamics": 0.5},
                "physics": {"w_state_supervised": 0.1, "w_dynamics": 5.0},
                "fine_tune": {"w_state_supervised": 0.1, "w_dynamics": 2.0}
            }
        curriculum_config = CurriculumConfig(**curriculum_dict)
        
        # Optimization config
        optimization_config = OptimizationConfig(**(config_dict.get('optimization', {})))
        
        # Data config
        data_config = DataConfig(**(config_dict.get('data', {})))
        
        # Experiment config
        experiment_config = ExperimentConfig(**(config_dict.get('experiment', {})))
        
        # Physics config
        physics_config = PhysicsConfig(**(config_dict.get('physics', {})))
        
        # Hyperparameter search if present
        hyperparameter_search_config = None
        if 'hyperparameter_search' in config_dict:
            hyperparameter_search_config = HyperparameterSearchConfig(**config_dict['hyperparameter_search'])
        
        return cls(
            model_config=model_config,
            dataset_config=dataset_config,
            curriculum_config=curriculum_config,
            optimization_config=optimization_config,
            data_config=data_config,
            experiment_config=experiment_config,
            physics_config=physics_config,
            hyperparameter_search_config=hyperparameter_search_config,
            random_seed=config_dict.get('random_seed', 42),
            mixed_precision=config_dict.get('mixed_precision', False),
            compile_model=config_dict.get('compile_model', False),
            device=config_dict.get('device', 'auto'),
            distributed_training=config_dict.get('distributed_training', False)
        )
    
    def save_to_yaml(self, path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary, handling enums
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def convert_value(value):
            if isinstance(value, Enum):
                return value.value
            elif hasattr(value, 'to_dict'):
                return value.to_dict()
            elif isinstance(value, (list, tuple)):
                return [convert_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value
        
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            result[field_name] = convert_value(value)
        
        return result
    


# Factory functions for common configurations
def create_development_config() -> TrainingConfig:
    """Create configuration for development/debugging."""
    config = TrainingConfig()
    
    # Shorter epochs for quick iteration
    config.curriculum_config.sanity_epochs = 10
    config.curriculum_config.supervised_epochs = 50
    config.curriculum_config.mixed_epochs = 100
    config.curriculum_config.physics_epochs = 200
    config.curriculum_config.fine_tune_epochs = 50
    
    # Smaller batches and data
    config.data_config.batch_size = 8
    
    # More frequent logging
    config.experiment_config.log_frequency = 5
    config.experiment_config.checkpoint_frequency = 25
    
    return config

def create_production_config() -> TrainingConfig:
    """Create configuration for full production training."""
    config = TrainingConfig()
    
    # Full curriculum as specified in implementation plan
    config.curriculum_config.sanity_epochs = 100
    config.curriculum_config.supervised_epochs = 500
    config.curriculum_config.mixed_epochs = 1000
    config.curriculum_config.physics_epochs = 2000
    config.curriculum_config.fine_tune_epochs = 500
    
    # Production batch sizes
    config.data_config.batch_size = 32
    config.optimization_config.learning_rate = 1e-3
    
    # Standard logging
    config.experiment_config.log_frequency = 50
    config.experiment_config.checkpoint_frequency = 100
    
    return config

def create_hyperparameter_search_config() -> TrainingConfig:
    """Create configuration for hyperparameter search."""
    config = create_development_config()  # Use shorter epochs
    
    # Enable hyperparameter search
    config.hyperparameter_search = HyperparameterSearchConfig()
    config.hyperparameter_search.max_trials = 20
    
    return config