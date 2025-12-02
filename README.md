# NP-SNN: Neural Physics-Informed Spiking Neural Networks for Space Debris Tracking

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

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dataset       │    │   Time Encoding  │    │   SNN Core      │
│   Generation    │───▶│   (Fourier/MLP)  │───▶│   (LIF Layers)  │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Physics       │    │   Decoder        │    │   Hybrid        │
│   Propagation   │◀───│   (State + σ)    │───▶│   Filtering     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

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

### 2. Generate Synthetic Data

```bash
# Create synthetic orbital scenarios
python -m src.data.generators --config configs/space_debris_simulation.yaml

# Generate sensor observations  
python -m src.data.sensors --scenario baseline_leo
```

### 3. Train NP-SNN Model

```bash
# Stage 1: Supervised pretraining
python -m src.train.train_loop --stage supervised --epochs 200

# Stage 2: Mixed physics training  
python -m src.train.train_loop --stage mixed --epochs 300

# Stage 3: Physics-dominant training
python -m src.train.train_loop --stage physics --epochs 500
```

### 4. Evaluate Performance

```bash
# Run comprehensive evaluation
python -m src.eval.metrics --model-path checkpoints/best_model.pt

# Compare against baselines
python -m src.eval.benchmarks --models npsnn sgp4 ekf ukf
```

### 5. View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open browser to http://localhost:5000
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

## 📊 Current Implementation Status

### ✅ Completed Components
- **Project structure and build system** - Complete modular architecture
- **Configuration management** - YAML-based config with validation
- **Orbital mechanics simulation** - Numerical propagation with J2 perturbations
- **NP-SNN model architecture** - Time encoding, SNN core, probabilistic decoders
- **Physics-informed loss functions** - Dynamics residual, conservation constraints
- **Training pipeline** - Curriculum learning with MLflow tracking
- **Evaluation metrics** - Comprehensive trajectory and physics validation
- **Testing framework** - Unit tests with pytest

### 🚧 In Progress
- **Advanced force models** - Atmospheric drag (NRLMSISE-00), solar radiation pressure
- **Multi-sensor fusion** - Optical + radar data integration
- **Uncertainty quantification** - Calibration and propagation validation
- **Domain randomization** - Robust training under sensor variations

### 🔮 Future Work
- **Real-time processing** - Optimization for operational deployment
- **Hardware acceleration** - CUDA kernels and neuromorphic chip integration
- **Multi-object scaling** - Demonstrated performance on 100+ objects
- **Production API** - RESTful service with containerization

## 🔬 Technical Details

### Physics-Informed Training
- **Automatic differentiation** for computing dr/dt and dv/dt from neural network outputs
- **Collocation points** between observations for physics residual evaluation
- **Conservation constraints** with soft penalty terms for energy and angular momentum
- **Dynamic loss balancing** using learnable uncertainty parameters

### Neuromorphic Integration
- **Surrogate gradients** for SNN backpropagation (fast sigmoid, piecewise linear)
- **Membrane potential normalization** for training stability
- **Temporal dynamics** preserved through recurrent connections
- **Event-driven processing** for future integration with neuromorphic cameras

## 🧪 Testing & Validation

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/unit/test_propagators.py -v
pytest tests/unit/test_time_encoding.py -v
```

## 📚 Documentation

- **[Project Proposal](docs/project_proposal.md)**: Scientific motivation and overview
- **Configuration Reference**: Complete parameter documentation in YAML files
- **API Documentation**: Generated from comprehensive docstrings

## 🤝 Contributing

Contributions welcome! Please:

1. **Fork the repository** and create a feature branch
2. **Follow code style** (black, flake8, mypy)
3. **Add tests** for new functionality  
4. **Update documentation** including docstrings
5. **Submit pull request** with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@software{oyaye2025,
  title={OYAYE: A Hybrid PINN–SNN Framework for Energy-Efficient Space Situational Awareness},
  year={2025},
  url={https://github.com/GabriellJacinto/Oyaye}
}
```
