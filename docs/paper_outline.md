# **2. Background & Literature Review — 2.5 pages**

## 2.4 Machine Learning for SSA

* RNN/Transformers (limitations: require dense data)
* PINNs (pros/cons)
* Neural ODEs for motion prediction

## 2.5 Spiking Neural Networks

* LIF/ALIF neurons
* Surrogate gradients
* Strong inductive bias for temporal dynamics
* Energy-efficient inference (relevance for spacecraft)

## 2.6 NP-SNN

* Hybrid SNN + physics-informed loss
* Why it is suitable for orbital mechanics
* Summary of the original paper and your adaptation

# **3. Methodology — 4 pages**

## **3.1 Dataset Design & Simulation Framework**

### 3.1.1 Motivation for Simulation

* Lack of open real datasets
* Need for controlled multi-scenario coverage

### 3.1.2 Orbital Dynamics Models

* Two-body + J2
* Drag model
* SRP for GEO
* Atmospheric model (simplified or exponential)

### 3.1.3 Scenario Space

* LEO circular
* LEO elliptical
* GEO
* Molniya
* Fragmentation cloud generation
* Domain randomization strategy

### 3.1.4 Observation Models

* RA/Dec optical
* Range/Doppler radar
* Temporal sampling
* Noise model (Gaussian + outliers)

### 3.1.5 Dataset Statistics

* Train/val/test split
* OOD subsets
* Per-scenario sample counts
* Visualization of trajectory diversity

---

## **3.2 NP-SNN Model Architecture**

### 3.2.1 Overview

* Encoder → SNN core → Decoder
* Diagram (ASCII or figure)

### 3.2.2 Time Encoding

* sin/cos encoding
* Temporal embeddings
* Optional scenario conditioning

### 3.2.3 SNN Core

* LIF/ALIF equation
* Membrane dynamics
* Surrogate gradients
* Stability considerations

### 3.2.4 Physics-Informed Module

Define all losses mathematically:

* State loss
* Dynamics residual
* Conservation loss (if used)
* Regularization terms
* Weighting strategy

### 3.2.5 Decoder

* MLP for r(t), v(t)
* Optional uncertainty head
* Optional multi-horizon output

### 3.2.6 Training Strategy

* Curriculum
* Domain randomization
* Balanced scenario sampling
* MLflow experiment tracking

---

## **3.3 Experimental Setup**

### 3.3.1 Baselines

* EKF
* SGP4
* RNN/MLP baseline (optional)

### 3.3.2 Metrics

* RMSE position/velocity
* Long-term drift
* Stability of invariants
* Uncertainty calibration
* Computational footprint

### 3.3.3 Implementation Details

* Hardware
* Training time
* Learning rate schedule
* Hyperparameters
* Data preprocessing pipeline

---

# **4. Results — 3.5 pages**

## 4.1 Quantitative Results

Tables comparing:

* NP-SNN vs EKF vs baselines
* Per-scenario performance (LEO/GEO/FRAG/etc.)
* OOD generalization

## 4.2 Qualitative Results

Plots:

* Trajectory overlays
* Residual errors vs time
* Energy drift
* Fragmentation cloud behavior
* Prediction vs ground truth timelines

## 4.3 Discussion

* Where NP-SNN excels
* Failure modes
* Effect of physical constraints
* Effect of conditioning

---

## **4.4 Ablation Studies**

At least 3 ablations:

1. No physics loss
2. No scenario conditioning
3. No domain randomization
4. No surrogate gradients (optional)
5. SNN → replaced by MLP (optional)

Show tables/plots.

Explain what each ablation reveals about the model.

---

# **5. Conclusion — 1 page**

## 5.1 Summary

* Recap contributions
* Key findings
* Significance of using NP-SNN for space debris prediction

## 5.2 Limitations

* Dependence on synthetic data
* Simplified physics
* No multi-object tracking
* Fragmentation model may be simplistic

## 5.3 Future Work (Phase 2 Roadmap)

* Hungarian/JPDAF integration
* Multi-object tracking
* Real sensor fine-tuning (optical/radar/event cameras)
* Uncertainty-aware collision-risk estimation
* Mixture-of-experts for scenario specialization
* Onboard neuromorphic deployment (Loihi/Dynap-SE)
