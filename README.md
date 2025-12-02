# OYAYE: A Hybrid PINN–SNN Framework for Energy-Efficient Space Situational Awareness

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/mlflow-tracking-blue)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

**Advanced neural physics-informed spiking neural network system for space debris detection, tracking, and orbital trajectory prediction combining neuromorphic computing with physics-informed machine learning.**

## 🚀 Project Overview

This project implements a novel **NP-SNN (Neural Physics-Informed Spiking Neural Network)** architecture for space debris detection and tracking. The system combines:

- **Spiking Neural Networks (SNN)** for neuromorphic, event-driven sensor processing
- **Physics-Informed Neural Networks (PINN)** with orbital mechanics constraints  
- **High-fidelity orbital propagation** with J2, atmospheric drag, and solar radiation pressure
- **Multi-sensor fusion** (optical angles, radar range/Doppler, imaging)
- **Uncertainty quantification** with aleatoric and epistemic uncertainty estimation
- **Hybrid filtering** integration with EKF/UKF/Particle filters

## ✨ Key Features

### 🧠 Neural Architecture
- **Time-encoding layers** with Fourier features and learned temporal representations
- **Multi-layer SNN core** using LIF neurons with surrogate gradient training
- **Physics-constrained decoders** outputting continuous orbital states (r, v)
- **Uncertainty quantification** with probabilistic and multi-head decoders

### 🛰️ Orbital Mechanics
- **High-fidelity propagation** with numerical integration (RK45)
- **Realistic perturbations**: J2/J3/J4 harmonics, atmospheric drag (NRLMSISE-00), SRP
- **Energy and angular momentum conservation** constraints in loss functions
- **Multi-object scaling** with shared models and object-specific embeddings

### 📡 Sensor Modeling  
- **Optical telescopes**: RA/Dec angles with realistic noise and visibility constraints
- **Radar systems**: Range/Doppler with beam patterns and detection thresholds
- **Event-based cameras**: Future integration for neuromorphic sensing
- **Domain randomization**: Sensor biases, noise correlation, missed detections

### 🎯 Training & Evaluation
- **Curriculum learning**: Supervised pretraining → mixed physics → physics-dominant
- **Dynamic loss balancing** with learnable uncertainty weighting
- **MLflow experiment tracking** with full reproducibility (configs, seeds, artifacts)
- **Comprehensive benchmarking** against SGP4, EKF, UKF baselines

## 📦 Project Structure

```
Oyaye/
├── configs/
│   └── space_debris_simulation.yaml    # Configuration parameters
├── src/
│   ├── data/
│   │   ├── generators.py              # Scenario generation & sampling
│   │   ├── sensors.py                 # Optical/radar/imaging simulation
│   │   └── io.py                      # Dataset I/O utilities
│   ├── models/
│   │   ├── time_encoding.py           # Fourier/learned time features
│   │   ├── snn_core.py                # LIF/RLIF neuron layers
│   │   ├── decoder.py                 # MLP decoders + uncertainty
│   │   └── npsnn.py                   # Full NP-SNN model
│   ├── physics/
│   │   ├── propagators.py             # Numerical orbital propagation
│   │   └── accel_models.py            # Force models (J2, drag, SRP)
│   ├── train/
│   │   ├── train_loop.py              # Curriculum training pipeline
│   │   ├── losses.py                  # Physics-informed loss functions
│   │   └── schedule.py                # Learning rate & loss scheduling
│   ├── eval/
│   │   ├── metrics.py                 # Evaluation metrics (RMSE, energy drift)
│   │   └── benchmarks.py              # Baseline comparisons
│   └── infra/
│       ├── mlflow_logger.py           # Experiment tracking
│       └── utils.py                   # Utilities & configuration
├── tests/
│   └── unit/
│       └── test_propagators.py        # Unit tests
├── docs/
│   ├── implementation_plan.md         # Detailed implementation roadmap
│   └── project_proposal.md            # Project overview & motivation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Project setup script
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/GabriellJacinto/Oyaye.git
cd Oyaye

# Set up Python environment
conda create -n npsnn-env python=3.10
conda activate npsnn-env

# Install dependencies
pip install -r requirements.txt
```

## 🔬 Reproducing Experimental Results

This section provides step-by-step instructions to reproduce the exact experimental results documented in our research report (see `docs/paper.pdf`).

### 📋 Prerequisites

Ensure you have:
- **NVIDIA GPU** with CUDA support (tested on GeForce MX550 2GB)
- **Python 3.10+** with conda/mamba environment manager
- **8GB+ RAM** for training (16GB+ recommended for full curriculum)
- **5GB+ disk space** for datasets, models, and MLflow artifacts

### 🎯 Reproducing Core Results (6,877km RMSE)

#### Step 1: Environment Setup
```bash
# Activate environment and configure Python paths
conda activate npsnn-env
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Step 2: Generate Training Data with Exact Parameters
```bash
# Create the exact dataset used in experiments
python -c "
import sys
sys.path.append('.')
from src.data.generators import ScenarioGenerator
from src.data.trajectory_transforms import TrajectoryTransforms

# Generate LEO scenarios with documented parameters
config = {
    'n_scenarios_train': 100,  # Debug mode (full: 1000+)
    'n_scenarios_val': 20,    # Debug mode (full: 200+)
    'altitude_range': [300e3, 800e3],  # 300-800 km LEO
    'time_horizon_hours': 8,           # 8-hour prediction horizon
    'dt_minutes': 5,                   # 5-minute time resolution
    'observation_noise': 0.001,        # Standard noise level
    'missing_data_rate': 0.2,          # 20% observation gaps
    'seed': 42                         # Reproducibility seed
}

generator = ScenarioGenerator(config)
train_data, val_data = generator.generate_training_set()
print('Dataset generation complete - matches experimental parameters')
"
```

#### Step 3: Train NP-SNN with Documented Configuration
```bash
# Train with exact experimental settings (reproduces 799,142 parameters)
python examples/train_npsnn_normalized.py \
    --config configs/npsnn_training.yaml \
    --experiment-name "reproduction_run_$(date +%Y%m%d)" \
    --debug \
    --seed 42

# This reproduces the documented training:
# - 799,142 total parameters
# - Normalized data pipeline (positions ÷ 1e7, velocities ÷ 1e4)  
# - 5-stage curriculum learning
# - Final validation loss: ~0.531
```

#### Step 4: Evaluate Trained Model
```bash
# Test model performance (reproduces 6,877km RMSE result)
python tests/test_npsnn_simple.py

# Expected output should show:
# Position RMSE: ~6,877 km across prediction horizons
# Validation Loss: ~0.531 
# Model Status: Stable, finite outputs
```

#### Step 5: Generate Architecture Visualizations
```bash
# Create publication-quality architecture diagrams
python scripts/generate_architecture_diagrams.py

# Generates 4 PNG files in docs/:
# - npsnn_architecture_overview.png (system architecture)
# - snn_layer_details.png (LIF neuron dynamics)  
# - training_flow_curriculum.png (curriculum learning)
# - data_normalization_flow.png (data pipeline)
```

### 📊 Baseline Comparison Reproduction

#### Run Classical Baselines
```bash
# Evaluate all baseline methods with documented parameters
python scripts/evaluate_baselines.py \
    --n-scenarios 15 \
    --horizon-hours 8 \
    --baselines SGP4 EKF_J2 UKF_J2 MLP

# Expected baseline performance (literature estimates):
# SGP4: ~100-200 km RMSE
# EKF with J2: ~80-150 km RMSE  
# UKF with J2: ~70-120 km RMSE
# MLP Neural Net: ~200-500 km RMSE
```

### 🔍 Key Experimental Parameters

The following parameters are critical for exact reproduction:

#### Model Architecture (799,142 parameters)
```python
model_config = {
    'time_encoding': {
        'fourier_features': 64,
        'model_dim': 256
    },
    'snn_core': {
        'n_layers': 3,
        'neurons_per_layer': 256, 
        'neuron_type': 'LIF',
        'threshold': 0.5,
        'decay': 0.9,
        'dropout': 0.1
    },
    'decoder': {
        'ensemble_size': 5,
        'hidden_dim': 128,
        'output_dim': 6  # [x, y, z, vx, vy, vz]
    }
}
```

#### Training Configuration
```python
training_config = {
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 4,          # GPU memory optimized
    'max_epochs': 50,         # Debug mode (full: 200)
    'curriculum_stages': 5,   # Sanity→Supervised→Mixed→Physics→Finetune
    'scheduler': 'CosineAnnealingLR'
}
```

#### Data Normalization (Critical for Stability)
```python
# ESSENTIAL: These transforms prevent training divergence
normalization = {
    'position_scale': 1e7,    # Reduces ~6.7e6m to ~0.67
    'velocity_scale': 1e4,    # Reduces ~7.5e3m/s to ~0.75
    'impact': '10^13x loss reduction (4.5e13 → 0.531)'
}
```


## 🧪 Example Usage

### Basic Training Example

```python
from src.models.npsnn import NPSNN
from src.train.train_loop import NPSNNTrainer, TrainingConfig
from src.data.generators import ScenarioGenerator

# Load configuration
config = {
    'time_encoding': {'type': 'fourier', 'dim': 64},
    'snn': {'hidden_sizes': [128, 64], 'beta': 0.9},
    'decoder': {'type': 'probabilistic', 'output_size': 6}
}

# Create model
model = NPSNN(config)

# Generate training data
generator = ScenarioGenerator(config)
train_data = generator.generate_scenarios(n_objects=100)

# Set up training
training_config = TrainingConfig(
    model_config=config,
    num_epochs=1000,
    batch_size=32,
    learning_rate=1e-3
)

trainer = NPSNNTrainer(training_config, train_data, val_data)
trainer.train()
```

### Physics-Informed Loss Example

```python
from src.train.losses import CompositeLoss

# Configure loss function
loss_config = {
    'w_measurement': 1.0,
    'w_dynamics': 3.0,
    'w_conservation': 0.1,
    'include_j2': True
}

criterion = CompositeLoss(loss_config)

# Compute loss with automatic differentiation
losses = criterion(model_outputs, batch_data)
print(f"Total loss: {losses['total_loss']:.6f}")
print(f"Dynamics residual: {losses['dynamics_loss']:.6f}")
print(f"Energy conservation: {losses['conservation_loss']:.6f}")
```

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@software{oyaye2025,
  title={OYAYE: A Hybrid PINN–SNN Framework for Energy-Efficient Space Situational Awareness},
  year={2025},
  url={https://github.com/GabriellJacinto/Oyaye}
}
```
