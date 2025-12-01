# NP-SNN Development — Advanced TODOs and Implementation Plan

**Purpose:** provide a single, detailed, actionable roadmap and checklist to implement a production-grade physics-informed Spiking Neural Network (NP-SNN) system for space-debris detection & tracking, including advanced model architecture, rigorous dataset generation, training, evaluation, and deployment tasks.

## 0. High-level objectives

1. Implement a physics-informed SNN model that outputs continuous-time state trajectories (r, v) and enforces orbital dynamics via loss functions.
2. Build a high-fidelity, configurable dataset generator that samples real-distribution priors and creates synthetic sensor streams (optical, radar, images) with domain randomization.
3. Integrate uncertainty quantification, hybrid filtering (EKF/UKF/PF) for online updates, and experiment tracking (MLflow).
4. Scale to multi-object scenarios and support data-association experiments (track initiation, fragmentation events).

## 1. Project layout (files & modules)

```
project/
├── configs/
│   └── space_debris_simulation.yaml
├── src/
│   ├── data/
│   │   ├── generators.py        # scenario generator + samplers
│   │   ├── sensors.py           # optical/radar/image emulators
│   │   └── io.py                # dataset writer/reader (.npz, parquet)
│   ├── models/
│   │   ├── time_encoding.py     # Fourier / learned encoders
│   │   ├── snn_core.py          # LIF/RLIF stacks, surrogate grads setup
│   │   ├── decoder.py           # MLP decoders + uncertainty heads
│   │   └── npsnn.py             # NP-SNN full model class + forward
│   ├── physics/
│   │   ├── propagators.py      # numerical integrator, J2/drag/SRP
│   │   └── accel_models.py     # modular acceleration functions
│   ├── train/
│   │   ├── train_loop.py       # full training loop + curriculum
│   │   ├── losses.py           # measurement, dynamics, invariants, balancing
│   │   └── schedule.py         # curriculum & LR/weight schedulers
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── viz.py
│   │   └── benchmarks.py       # baselines: SGP4, EKF, UKF, PF
│   ├── infra/
│   │   ├── mlflow_logger.py
│   │   └── utils.py
│   └── experiments/
│       └── notebooks/
└── tests/
    └── unit/
```


## 2. Dataset generation — advanced tasks (priority: HIGH)

### 2.1. Real-distribution priors & catalog anchoring

* Implement `generators.sample_from_catalog()` that pulls (or reads cached) TLE/SATCAT snapshots and extracts empirical distributions for: altitude, inclination, eccentricity, object size classes, AMR.
* Save PDF/visualization of sampled marginals for validation.
* **Deliverable:** `configs/catalog_priors.json` and `notebooks/catalog_analysis.ipynb`.

### 2.2. High-fidelity propagator

* Implement `propagators.NumericalPropagator` using an adaptive integrator (Dormand-Prince RK45 or similar) with modular forces: 2-body, J2/J3/J4, drag (NRLMSISE-00), SRP, 3rd-body (Sun/Moon).
* Expose toggle flags for cheaper modes (SGP4 mode for TLE-consistency) vs. full numeric.
* Validate energy conservation for no-drag runs and J2 secular trends.
* **Deliverable:** `propagators.py` with unit tests comparing to poliastro and/or Orekit outputs (if available).

### 2.3. Drag & atmosphere

* Wrap NRLMSISE-00 model (via an existing Python lib or local implementation). Provide sampling of F10.7 & Ap and optionally random time-varying indices.
* Allow per-object ballistic coefficient uncertain parameterization (learnable or sampled).

### 2.4. Sensor emulators (optical, radar, imaging)

* Optical (angles): RA/Dec conversion with Earth rotation, topocentric conversion (observer lat/lon/alt), star background occlusion, moon illumination phase, magnitude-to-detection model, PSF, motion blur (streak rendering), pixel-level image generator (fast approximate) using randomized PSF convolve.
* Radar: range, az/el, range-rate; radar site models with beam patterns, detection SNR thresholding, false alarm rates, scan scheduling simulation (beam dwell + revisit cadence), RCS model. Include optional chirp-based range precision modeling.
* Camera images: lightweight image generator that composes a starfield + point/streak; support saving FITS or PNG. Use OpenCV for PSF and noise.
* **Deliverable:** `sensors.py` with `simulate_optical()`, `simulate_radar()`, `render_image()`.

### 2.5. Observation scheduling & visibility

* Implement an observation scheduler that simulates realistic sensor ops: night-only optical windows, cloud-induced dropout probability, radar scan windows and angular coverage, pointing constraints, Moon/Sun occlusion.
* Allow injection of prioritized follow-ups (targeted tracking after detection) vs. survey-mode sampling.

### 2.6. Metadata & provenance

* Every generated file must include metadata: random seed, scenario name, sample priors, versioned YAML snapshot, environment indexes (F10.7, Ap), catalog seeds. Use Parquet + JSON sidecars.

### 2.7. Domain randomization & adversarial scenarii

* Implement randomized perturbations: additive sensor biases, temporally correlated noise (AR(1)), mis-calibrations, false positive injection, missed detections with clustering.
* Implement fragmentation event generator with N-fragments at event time, size/velocity delta sampling based on explosion models (e.g., NASA explosion cloud model references).

## 3. Model architecture — advanced tasks (priority: VERY HIGH)

> Goal: implement a production-grade NP-SNN architecture that supports multi-mode inputs, continuous-time output, and advanced training techniques from the paper and modern PINN literature.

### 3.1. Time encoding & conditioning

* Implement multiple time encoders (switchable): Fourier features, positional-encoding MLP, and learnable temporal basis (sinc/bump). Provide normalization strategies (minutes/hours scaling) and multi-scale frequencies.
* Implement object-conditioning vector (embedding): per-object learnable parameter vector to encode ballistic coefficient, size class, known priors (if multi-object shared model is used).

### 3.2. SNN core (Encode → Main → Decode)

* Implement an `Encode` block: maps time embeddings (and optional obs embeddings) into input currents for SNN. Support spike-encoding alternatives: rate coding, temporal coding, and direct analogue current injection.
* Implement multi-layer SNN core with configurable topology:

  * Stacked LIF / Leaky neurons with optional residual connections (SNN residual blocks).
  * Layer normalization for membrane potentials.
  * Support for both feedforward and recurrent spiking layers (stateful across timesteps for sequence training).
* Implement decoder head(s):

  * Deterministic decoder (MLP) that maps membrane potentials → state (r,v).
  * Probabilistic decoder: outputs mean + log-variance for each state dimension (for aleatoric uncertainty).
  * Auxiliary heads: energy estimate, angular momentum, ballistic coefficient estimator (optional, per-object latent).

### 3.3. Surrogate gradients & training stability

* Implement modular surrogate gradient functions (fast sigmoid, piecewise-linear), allow switching in config.
* Add gradient clipping, loss scaling, and batch-norm-like population stats for membrane variables.
* Implement mixed-precision training safe wrappers if using GPU (AMP) — ensure surrogate gradient numeric stability.

### 3.4. Loss design (composite) — implementation details

* Measurement loss: support both angle-only and range-only measurement models; implement geometric projection functions with proper topocentric transforms; support per-modality weighting and heteroscedastic noise models.
* Dynamics residual loss:

  * Compute dr/dt and dv/dt via autograd of model outputs w.r.t time input (as in earlier script), but scale up efficiency:

    * Implement per-sample automatic differentiation carefully to avoid exploding memory; consider chunked processing or Jacobian-vector product trick.
  * Alternative: learn a neural ODE style residual by parameterizing time-derivative as a small NN that depends on membrane state — include physics-informed regularizer to pull it toward analytic accel_model(r,v).
* Energy / invariants loss: compute specific energy and angular momentum invariants and penalize drift; implement options for soft constraints vs. Lagrange-multiplier style enforcement.
* Regularization: parameter & membrane L2, temporal smoothness (penalize second derivative of predicted r), spectral norm constraints (if needed).

### 3.5. Dynamic loss-balancing & meta-weighting

* Implement dynamic loss balancing mechanism (learnable log-variance parameters per loss term ala Kendall et al.) *and* a scheduling approach (e.g., gradually increase L_dyn weight).
* Experiment with gradient-normalization (GradNorm) to keep different losses balanced.

### 3.6. Uncertainty quantification

* Aleatoric: via probabilistic decoder head (Gaussian), heteroscedastic per-dimension.
* Epistemic: use Monte Carlo dropout in decoder or deep ensembles (train K models) — integrate with MLflow to store ensembles.
* Propagate uncertainties into downstream filter (EKF/UKF/PF) as observation priors.

### 3.7. Hybrid filtering & online update API

* Implement an API to consume NP-SNN continuous predictions as a motion prior in a filter:

  * Build `filters.EKF_from_model` that linearizes NP-SNN outputs (compute Jacobian w.r.t state/time) for EKF update.
  * Implement `ParticleFilterWithNPrior`: sample particles around NP-SNN prediction and weight by measurement likelihood.
* Implement a sequential update routine: model predicts continuous trajectory for a horizon; when new obs arrive, filter updates state and re-conditions NP-SNN (fine-tune or update latent per-object embedding).

### 3.8. Scalability & multi-object handling

* Design patterns to handle many objects efficiently:

  * Single shared NP-SNN conditioned by object embedding (scales better) vs. per-object model (simpler but costly).
  * Batch processing across objects & times with efficient autograd tricks.
* Consider training on mixed batches: many short arcs from many objects rather than single long arcs.

### 3.9. Model IO & serialization

* Implement deterministic save/load using PyTorch state_dict plus separate JSON for config & latent embeddings. Version models with git hash + config snapshot.

## 4. Training & experiments — advanced tasks (priority: HIGH)

### 4.1 Curriculum & staged training (recap + actionable schedule)

* **Stage 0 — Sanity / warm-start (very short):**

  * Purpose: verify data pipeline, loss, autograd, and that model can overfit tiny dataset.
  * Data: 2–5 objects, horizon 10–30 min, dense observations.
  * Loss: strong `w_state_supervised=1.0`, `w_dyn=0`.
  * Train for 100–300 steps until near-zero supervised error.

* **Stage 1 — Supervised short-horizon pretraining (2–7 days of runs):**

  * Purpose: learn basic mapping from time→state and decoder conditioning.
  * Data: many short arcs (1–2 orbits) from many objects (200–2000 short arcs).
  * Loss weights: `w_state_supervised` high (0.5–1.0), `w_meas` normal (1.0), `w_dyn` small (0.1).
  * Horizon: 1–4 hours.
  * Early stopping on val supervised RMSE.

* **Stage 2 — Mixed supervised + physics (2–4 weeks of runs):**

  * Purpose: shift emphasis from data to physics regularization.
  * Reduce `w_state_supervised` progressively to 0 over 10–50 epochs; increase `w_dyn` to 1.0–10.0 and `w_energy` to 0.01–0.2.
  * Add collocation samples between observation times (10–60 s density).
  * Add small noise/domain randomization.

* **Stage 3 — Physics-dominant training + robustness (long):**

  * Purpose: remove reliance on perfect state labels; train with measurement + dynamics only.
  * `w_state_supervised = 0`; `w_meas = 1.0`; `w_dyn = 1.0–10.0`; optionally `w_energy = 0.05–0.2`.
  * Add challenging scenarios: sparse ops, small-telescope noise, fragmentation, HAMR objects.

* **Stage 4 — Fine-tuning & domain adaptation:**

  * Use TLEs and real catalog arcs as *weak supervision* (treat them as measured trajectories, not ground truth).
  * Fine-tune only measurement-model portions (decoder + observation head) with small LR (1e-5–1e-4) and early stopping.

### 4.2 Loss weight schedules & dynamic balancing (practical recipes)

* **Static baseline:** set `w_meas=1.0`, `w_dyn=1.0`, `w_energy=0.05`; tune from there.
* **Gradual schedule:** `w_state_supervised` decays exponentially: `w0 * exp(-k * epoch)` where `k` chosen to reach ~0 after 20–50 epochs.
* **Learned log-variance (Kendall):** implement `s_i` trainable scalar per loss term; final combined loss `Σ (exp(-s_i) L_i + s_i)`. Good to stabilize tradeoffs.
* **GradNorm:** if multiple tasks, optionally implement GradNorm to normalize gradient magnitudes across losses (harder, but effective).

### 4.3 Hyperparameter ranges & search plan (concrete)

Run sweeps (random or Bayesian) over these ranges:

Model architecture:

* time_feat_dim: [16, 32, 64, 128]
* snn_hidden: [64, 128, 256]
* snn_layers: [1, 2, 3]
* surrogate slope: [5, 25, 50]

Training:

* learning_rate: [1e-4, 3e-4, 1e-3, 3e-3]
* optimizer: Adam / AdamW
* batch_size: [8, 16, 32]
* weight_decay: [0, 1e-5, 1e-4]

Loss:

* w_meas: [0.1, 1.0, 3.0]
* w_dyn: [0.1, 1.0, 5.0, 10.0]
* w_energy: [0.0, 0.01, 0.05, 0.2]
* supervised weight initial: [0.0, 0.1, 0.5, 1.0] (for pretraining)

Search plan:

* Stage A: small random search (20–50 trials) on time_feat_dim, snn_hidden, learning_rate.
* Stage B: fix architecture best candidate; sweep loss weights (30 trials).
* Stage C: final fine-tuning sweep with longer runs (10–20 trials).

Use MLflow sweeps; save seeds + best-run artifacts.

### 4.4 Batch design and collocation strategy (detailed)

* **Mini-batch composition:** a batch contains `B_objects x N_times` where each object contributes multiple (t, obs) points. Practical shape: batch size = number of (t,obs) pairs; ensure diversity across objects per batch.
* **Collocation augmentation:** for each object in the batch, add `k_coll` extra times sampled uniformly between that object's measurement times (k_coll in [1, 10]); include these in compute of `L_dyn` but do not add measurement loss.
* **Temporal chunking:** for recurrent training (if unrolling SNN), use short sequence chunks e.g., 8–32 time steps per unroll to control memory.
* **Data balancing:** ensure balanced representation of sensor modalities (optical vs radar) per batch to prevent model bias.


### 4.5 Validation & evaluation protocols (concrete)

Define three validation axes:

1. **Static hold-out (in-distribution):**

   * Hold out 10–20% of objects (object-level split).
   * Metrics: position RMSE (m), velocity RMSE (m/s), energy drift (relative).
   * Evaluate for horizons: 1 min, 10 min, 1 hr, 6 hr, 24 hr.

2. **Time extrapolation (temporal hold-out):**

   * Train on early arc, test on later arc for same objects (extrapolation).
   * Measure error growth vs time from last observation (plot error vs forecast horizon).

3. **Out-of-distribution (O.O.D.) scenarios:**

   * Test on scenarios not in training set: HAMR objects, fragmentation cloud, different F10.7/Ap conditions, sparse small-telescope datasets.
   * Metrics: same as above + track-loss rate (fraction of tracks that diverge > X km before Y hours).

Additional evaluation:

* **NEES** (Normalized Estimation Error Squared) if comparing with filter covariances.
* **Conjunction recall**: for predicted close approaches within threshold (e.g., <1 km) count detection recall/precision.
* **Uncertainty calibration:** use reliability diagrams and compute NLL if probabilistic outputs used.

Reporting:

* Provide tables with median and 90th percentile errors for each horizon.
* Plot error growth curves and sample trajectory reconstructions with residual plots (position & energy).

### 4.6 Baselines & ablation experiments (explicit list)

Run and report these:

Baselines:

* **SGP4 propagation** initialized from the same initial state (TLE) — measures how much ML improves over standard propagation.
* **EKF**: classical EKF with 2-body + J2 (tuned process noise).
* **UKF**: Unscented Kalman Filter for nonlinearity check.
* **Particle Filter**: small PF to provide non-Gaussian baseline.
* **Pure MLP (no SNN)**: same decode capacity but no spiking dynamics.

Ablations:

* NP-SNN without dynamics loss (`w_dyn=0`)
* NP-SNN without energy constraints (`w_energy=0`)
* NP-SNN with no collocation points
* NP-SNN with alternative time encoding (Fourier vs learned MLP)
* Shared model conditioned vs per-object models

Measure effect sizes (relative improvements) and include paired statistical tests where appropriate.

### 4.7 Uncertainty evaluation & propagation

* For **aleatoric uncertainty**, record predictive variance; compute negative log-likelihood (NLL) on validation set.
* For **epistemic**, use ensembles (K=3–5) or MC-dropout (n=16 samples). Compute predictive entropy and compare calibration (expected calibration error).
* Use uncertainties as inputs to downstream EKF/PF (observation covariance); evaluate NEES consistency.

### 4.8 Compute & resource planning (practical)

Storage & archiving:

* Each generated .npz for a 1-hour track per object ~ few MB; plan accordingly for thousands of samples.
* Log all runs in MLflow; keep top-K runs and archive rest.

Estimate runtimes:

* Small training run (epochs=300, batch_size=32) on dataset of ~10k samples: few hours on single GPU.
* Physics-dominant long runs (staged) may take days—schedule accordingly.

### 4.9 Reproducibility & experiment hygiene

* Always log: random seed (numpy, torch, python), YAML config snapshot, dataset fingerprint (hash), commit hash.
* Use deterministic seeds in DataLoader where possible; record non-deterministic ops (CUDA CuDNN may require disabling benchmarking for exact determinism).
* Provide a `reproduce.sh` that downloads artifacts from MLflow and re-runs best configuration.
* Containerize environment (Dockerfile) and attach conda env YAML.

### 4.10 Monitoring & continuous evaluation

* During training log:

  * Per-epoch metrics: loss breakdowns (L_meas, L_dyn, L_energy), val RMSEs, LR, gradient norms.
  * Periodic artifact saves: sample predictions vs truth (plot & short animation).
* Build an evaluation pipeline (script) that, given a model checkpoint, runs a standardized battery of tests (baseline comparison, O.O.D. tests, uncertainty calibration) and produces a PDF report.

### 4.11 Experiment recipes — concrete experiments to run (priority order)

1. **Dev sanity:** tiny dataset (n=5), supervised pretrain, ensure overfit. (1–2 runs)
2. **Architecture search:** random search over `time_feat_dim`, `snn_hidden`, `snn_layers`. (20–50 runs)
3. **Loss weight sweep:** fix architecture; grid search over `w_meas`, `w_dyn`, `w_energy`. (30 runs)
4. **Physics-only test:** train with `w_state_supervised=0`, `w_dyn>0`, measure extrapolation error growth. (5 runs)
5. **Uncertainty baseline:** compare ensembles vs MC-dropout on calibration. (5–10 runs)
6. **Hybrid filter integration:** use best NP-SNN prior inside EKF; measure NEES and 24 h prediction RMSE. (3–5 runs)
7. **Stress tests:** fragmentation cloud (vary fragment counts), small-telescope high noise; evaluate track-loss and association performance. (5 runs)

Record results and rank experiments by validation metrics.

### 4.12 Reporting & visualization suggestions

* For each experiment produce:

  * Table: hyperparams + final metrics.
  * Plots: training loss curves, validation RMSE vs epoch, error growth vs forecast horizon.
  * Example figure: predicted vs true orbit for 3 representative objects (show 3D, and projected RA/Dec + residual RMSE).
  * Uncertainty calibration plot: predicted sigma vs empirical error.
  * Ablation bar-chart (relative % change vs baseline).
* Include a one-page “experiment summary” sheet for each major run to include in the report.

### 4.13 Failure modes & debugging checklist

If model fails to converge or extrapolates poorly:

* Check data pipeline: alignment of times, units (km vs m), time scaling in encoders.
* Check autograd derivatives: validate with finite differences for dr/dt and dv/dt.
* Reduce learning rate; enable gradient clipping.
* Temporarily re-enable supervised state loss to stabilize.
* Verify physics model correctness (J2 sign, units).
* Check batch composition — ensure not single-object batches causing poor generalization.

## 5. Evaluation, testing & ablation (priority: MEDIUM)

* Ablations:

  * Remove dynamics loss — measure degradation.
  * Remove invariants loss.
  * change time-encoding type.
  * SNN vs non-spiking baseline (MLP or continuous PINN).
* Unit tests:

  * Propagator comparisons to poliastro/SGP4 for fixed orbits.
  * Sensor sim: known geometry produces expected RA/Dec.
  * Autograd derivative test: finite-diff check for dr/dt and dv/dt computations.

## 6. Infrastructure, experiments & reproducibility (priority: HIGH)

* MLflow integration: log config snapshot, scenario metadata, dataset hashes, random seeds, model checkpoints, metrics, and sample visualizations.
* Dockerfile & conda env YAML for reproducible compute stack (torch, snnTorch, poliastro, sgp4, mlflow).
* CI: lightweight tests for propagate_orbit, sensor sim, and end-to-end tiny training run (1–2 batches) to detect regression.


## 8. Deliverables checklist (ready-to-check)

* [ ] YAML config + scenario presets (versioned)
* [ ] Catalog prior extractor + distribution visualizations
* [ ] Numerical propagator with J2/drag/SRP + tests
* [ ] Optical/radar/image sensor emulators + tests
* [ ] Domain randomization module
* [ ] Dataset writer (npz/parquet) with metadata
* [ ] NP-SNN core (Encode/Main/Decode) + surrogate gradients
* [ ] Composite loss implementations + dynamic balancing
* [ ] Training loop with curriculum + MLflow logging
* [ ] Uncertainty quantification & hybrid filter interfaces
* [ ] Multi-object scaling experiments + baselines
* [ ] CI tests & reproducible env (Docker/conda)

## 9. Useful references & pointers (implementer notes)

* Use `snnTorch` for surrogate gradients and SNN primitives; study the NP-SNN paper's training strategies for loss-balancing.
* For atmospheric density, prefer wrapping `NRLMSISE-00` (via `pymsis` or other) and make F10.7 & Ap inputs configurable.
* For high-fidelity orbit reference comparisons, use `orekit` — Orekit Java bindings if you need absolute precision.
* For sensor image rendering, start with OpenCV for speed; migrate to a HAMR renderer only if needed.