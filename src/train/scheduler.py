"""
Curriculum learning scheduler for NP-SNN training.

Implements staged curriculum learning as specified in implementation plan section 4.1:
- Stage 0: Sanity check (overfit tiny dataset)
- Stage 1: Supervised pretraining  
- Stage 2: Mixed supervised + physics
- Stage 3: Physics-dominant training
- Stage 4: Fine-tuning & adaptation

Includes dynamic loss weight scheduling, data complexity progression,
and collocation point management.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np
from enum import Enum

from .config import TrainingConfig, CurriculumConfig, TrainingStage, LossSchedule

@dataclass
class StageInfo:
    """Information about current training stage."""
    stage: TrainingStage
    stage_epoch: int        # Epoch within current stage
    total_epoch: int        # Total epoch across all stages
    stage_progress: float   # Progress through current stage [0, 1]
    total_progress: float   # Progress through entire curriculum [0, 1]
    
class CurriculumScheduler:
    """
    Manages curriculum learning progression through training stages.
    
    Handles:
    - Stage transitions based on epoch counts
    - Dynamic loss weight scheduling within stages  
    - Data complexity progression (horizon, noise, observations)
    - Collocation point scheduling
    - Early stopping and validation logic
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        
        # Calculate stage boundaries
        self.stage_boundaries = self._calculate_stage_boundaries()
        
        # Track current state
        self.current_stage = TrainingStage.SANITY
        self.current_epoch = 0
        self.stage_start_epoch = 0
        
        # Track best validation performance for early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Cache for loss weights to avoid recalculation
        self._loss_weights_cache = {}
        
    def _calculate_stage_boundaries(self) -> Dict[TrainingStage, Tuple[int, int]]:
        """Calculate epoch boundaries for each stage."""
        boundaries = {}
        current_epoch = 0
        
        stages_and_durations = [
            (TrainingStage.SANITY, self.config.sanity_epochs),
            (TrainingStage.SUPERVISED, self.config.supervised_epochs),
            (TrainingStage.MIXED, self.config.mixed_epochs),
            (TrainingStage.PHYSICS, self.config.physics_epochs),
            (TrainingStage.FINE_TUNE, self.config.fine_tune_epochs)
        ]
        
        for stage, duration in stages_and_durations:
            start_epoch = current_epoch
            end_epoch = current_epoch + duration
            boundaries[stage] = (start_epoch, end_epoch)
            current_epoch = end_epoch
            
        return boundaries
    
    def update(self, epoch: int, val_loss: Optional[float] = None) -> StageInfo:
        """
        Update curriculum state for given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            val_loss: Validation loss for early stopping
            
        Returns:
            Current stage information
        """
        self.current_epoch = epoch
        
        # Determine current stage
        previous_stage = self.current_stage
        self.current_stage = self._get_stage_for_epoch(epoch)
        
        # Update stage start if we transitioned
        if self.current_stage != previous_stage:
            self.stage_start_epoch = epoch
            print(f"🚀 Curriculum transition: {previous_stage.value} → {self.current_stage.value} at epoch {epoch}")
            
            # Reset early stopping when entering new stage
            if self.current_stage != TrainingStage.SANITY:
                self.best_val_loss = float('inf')
                self.epochs_without_improvement = 0
        
        # Update early stopping tracking
        if val_loss is not None:
            self._update_early_stopping(val_loss)
        
        # Clear cache when stage changes
        if self.current_stage != previous_stage:
            self._loss_weights_cache.clear()
            
        return self.get_stage_info()
    
    def _get_stage_for_epoch(self, epoch: int) -> TrainingStage:
        """Determine which stage the given epoch belongs to."""
        for stage, (start, end) in self.stage_boundaries.items():
            if start <= epoch < end:
                return stage
        
        # If past all stages, stay in fine-tuning
        return TrainingStage.FINE_TUNE
    
    def _update_early_stopping(self, val_loss: float):
        """Update early stopping state."""
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        # Only apply early stopping after sanity stage
        if self.current_stage == TrainingStage.SANITY:
            return False
            
        return self.epochs_without_improvement >= self.config.early_stopping_patience
    
    def get_stage_info(self) -> StageInfo:
        """Get information about current training stage."""
        stage_start, stage_end = self.stage_boundaries[self.current_stage]
        stage_epoch = self.current_epoch - stage_start
        stage_duration = stage_end - stage_start
        
        # Calculate progress
        stage_progress = min(1.0, stage_epoch / max(1, stage_duration))
        
        total_epochs = max(self.stage_boundaries.values(), key=lambda x: x[1])[1]
        total_progress = min(1.0, self.current_epoch / max(1, total_epochs))
        
        return StageInfo(
            stage=self.current_stage,
            stage_epoch=stage_epoch,
            total_epoch=self.current_epoch,
            stage_progress=stage_progress,
            total_progress=total_progress
        )
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current loss weights based on curriculum stage and progress.
        
        Implements dynamic scheduling within stages per implementation plan.
        """
        cache_key = (self.current_stage, self.current_epoch)
        if cache_key in self._loss_weights_cache:
            return self._loss_weights_cache[cache_key]
        
        stage_info = self.get_stage_info()
        base_weights = self.config.stage_configs[self.current_stage.value].copy()
        
        # Apply dynamic scheduling for mixed stage
        if self.current_stage == TrainingStage.MIXED:
            base_weights = self._apply_mixed_stage_scheduling(base_weights, stage_info.stage_progress)
        
        self._loss_weights_cache[cache_key] = base_weights
        return base_weights
    
    def _apply_mixed_stage_scheduling(self, weights: Dict[str, float], progress: float) -> Dict[str, float]:
        """Apply dynamic weight scheduling during mixed stage."""
        # Supervised loss decreases exponentially
        if self.config.supervised_decay_schedule == LossSchedule.EXPONENTIAL:
            decay_factor = math.exp(-self.config.supervised_decay_rate * progress * 5)  # 5x for faster decay
            weights["w_state_supervised"] *= decay_factor
        elif self.config.supervised_decay_schedule == LossSchedule.LINEAR:
            weights["w_state_supervised"] *= (1.0 - progress)
        
        # Dynamics loss increases
        if self.config.dynamics_growth_schedule == LossSchedule.EXPONENTIAL:
            growth_factor = 1.0 - math.exp(-self.config.dynamics_growth_rate * progress * 3)
            weights["w_dynamics"] = 0.1 + (5.0 - 0.1) * growth_factor  # 0.1 → 5.0
        elif self.config.dynamics_growth_schedule == LossSchedule.LINEAR:
            weights["w_dynamics"] = 0.1 + (5.0 - 0.1) * progress
        
        return weights
    
    def get_data_complexity(self) -> Dict[str, float]:
        """
        Get current data complexity parameters based on curriculum progress.
        
        Returns trajectory horizons, noise levels, and observation densities.
        """
        stage_info = self.get_stage_info()
        
        # Progressive complexity based on overall training progress
        progress = stage_info.total_progress
        
        # Trajectory horizon increases over time
        horizon_hours = (
            self.config.start_horizon_hours + 
            (self.config.end_horizon_hours - self.config.start_horizon_hours) * progress
        )
        
        # Noise level increases gradually
        noise_level = (
            self.config.start_noise_level + 
            (self.config.end_noise_level - self.config.start_noise_level) * progress
        )
        
        # Observation density decreases (more challenging)  
        obs_density = 1.0 - 0.3 * progress  # Start dense, end with 70% observations
        
        return {
            "horizon_hours": horizon_hours,
            "noise_level": noise_level,
            "observation_density": obs_density
        }
    
    def get_collocation_points(self, trajectory_length: int, observation_times: torch.Tensor) -> torch.Tensor:
        """
        Generate collocation points between observations for physics loss.
        
        Args:
            trajectory_length: Total number of time points in trajectory
            observation_times: Tensor of observation time indices
            
        Returns:
            Collocation time indices for physics loss computation
        """
        if len(observation_times) < 2:
            return torch.tensor([], dtype=torch.long)
        
        collocation_points = []
        
        # Add points between consecutive observations
        for i in range(len(observation_times) - 1):
            start_time = observation_times[i].item()
            end_time = observation_times[i + 1].item()
            
            # Skip if too close
            if end_time - start_time <= 1:
                continue
            
            # Generate points in gap
            n_points = min(
                self.config.collocation_points_per_gap,
                max(1, int((end_time - start_time) / 2))  # At least every 2 time steps
            )
            
            gap_points = torch.linspace(
                start_time + 1, end_time - 1, n_points, dtype=torch.float32
            ).long()
            
            collocation_points.extend(gap_points.tolist())
        
        # Limit total number of collocation points
        if len(collocation_points) > self.config.max_collocation_points:
            # Sample uniformly from available points
            indices = np.random.choice(
                len(collocation_points), 
                self.config.max_collocation_points, 
                replace=False
            )
            collocation_points = [collocation_points[i] for i in sorted(indices)]
        
        return torch.tensor(collocation_points, dtype=torch.long)
    
    def should_use_supervised_loss(self) -> bool:
        """Check if supervised loss should be used in current stage."""
        return self.current_stage in [TrainingStage.SANITY, TrainingStage.SUPERVISED, TrainingStage.MIXED]
    
    def should_use_physics_loss(self) -> bool:
        """Check if physics loss should be used in current stage."""
        return self.current_stage in [TrainingStage.MIXED, TrainingStage.PHYSICS, TrainingStage.FINE_TUNE]
    
    def get_learning_rate_scale(self) -> float:
        """Get learning rate scaling factor for current stage."""
        # Reduce learning rate in later stages for stability
        stage_scales = {
            TrainingStage.SANITY: 1.0,
            TrainingStage.SUPERVISED: 1.0,
            TrainingStage.MIXED: 0.8,
            TrainingStage.PHYSICS: 0.5,
            TrainingStage.FINE_TUNE: 0.1
        }
        
        return stage_scales.get(self.current_stage, 1.0)
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current curriculum state."""
        stage_info = self.get_stage_info()
        loss_weights = self.get_loss_weights()
        data_complexity = self.get_data_complexity()
        
        return {
            "stage": self.current_stage.value,
            "stage_epoch": stage_info.stage_epoch,
            "total_epoch": stage_info.total_epoch,
            "stage_progress": f"{stage_info.stage_progress:.3f}",
            "total_progress": f"{stage_info.total_progress:.3f}",
            "loss_weights": {k: f"{v:.4f}" for k, v in loss_weights.items()},
            "data_complexity": {k: f"{v:.3f}" for k, v in data_complexity.items()},
            "use_supervised": self.should_use_supervised_loss(),
            "use_physics": self.should_use_physics_loss(),
            "lr_scale": self.get_learning_rate_scale(),
            "early_stop_patience_left": max(0, self.config.early_stopping_patience - self.epochs_without_improvement)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save scheduler state for checkpointing."""
        return {
            "current_stage": self.current_stage.value,
            "current_epoch": self.current_epoch,
            "stage_start_epoch": self.stage_start_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_stage = TrainingStage(state["current_stage"])
        self.current_epoch = state["current_epoch"]
        self.stage_start_epoch = state["stage_start_epoch"]
        self.best_val_loss = state["best_val_loss"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        
        # Clear cache after loading
        self._loss_weights_cache.clear()


class CollocationSampler:
    """
    Efficient collocation point sampling for physics loss computation.
    
    Handles different sampling strategies and manages memory usage
    for large trajectory datasets.
    """
    
    def __init__(self, 
                 strategy: str = "uniform",
                 max_points_per_trajectory: int = 100,
                 min_time_separation: float = 0.01):
        """
        Initialize collocation sampler.
        
        Args:
            strategy: Sampling strategy ("uniform", "adaptive", "random")
            max_points_per_trajectory: Maximum collocation points per trajectory
            min_time_separation: Minimum time separation between points (hours)
        """
        self.strategy = strategy
        self.max_points_per_trajectory = max_points_per_trajectory
        self.min_time_separation = min_time_separation
    
    def sample_points(self, 
                     times: torch.Tensor, 
                     observation_mask: torch.Tensor) -> torch.Tensor:
        """
        Sample collocation points for a trajectory.
        
        Args:
            times: Time points for trajectory (N,)
            observation_mask: Boolean mask indicating observation times (N,)
            
        Returns:
            Collocation point indices
        """
        if self.strategy == "uniform":
            return self._uniform_sampling(times, observation_mask)
        elif self.strategy == "adaptive":
            return self._adaptive_sampling(times, observation_mask)
        elif self.strategy == "random":
            return self._random_sampling(times, observation_mask)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def _uniform_sampling(self, times: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
        """Sample points uniformly between observations."""
        # Find observation indices
        obs_indices = torch.where(observation_mask)[0]
        
        if len(obs_indices) < 2:
            return torch.tensor([], dtype=torch.long)
        
        collocation_indices = []
        
        # Sample between each pair of observations
        for i in range(len(obs_indices) - 1):
            start_idx = obs_indices[i]
            end_idx = obs_indices[i + 1]
            
            # Skip if too close
            if end_idx - start_idx <= 2:
                continue
            
            # Determine number of points based on gap size
            gap_size = end_idx - start_idx - 1
            n_points = min(gap_size, self.max_points_per_trajectory // len(obs_indices))
            
            if n_points > 0:
                # Sample uniformly in gap
                gap_points = torch.linspace(
                    start_idx + 1, end_idx - 1, n_points, dtype=torch.float32
                ).long()
                collocation_indices.extend(gap_points.tolist())
        
        return torch.tensor(collocation_indices, dtype=torch.long)
    
    def _adaptive_sampling(self, times: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
        """Sample more points in larger gaps (adaptive to trajectory complexity)."""
        # Similar to uniform but with density proportional to gap size
        obs_indices = torch.where(observation_mask)[0]
        
        if len(obs_indices) < 2:
            return torch.tensor([], dtype=torch.long)
        
        # Calculate gap sizes
        gap_sizes = []
        for i in range(len(obs_indices) - 1):
            gap_sizes.append(obs_indices[i + 1] - obs_indices[i] - 1)
        
        total_gap = sum(gap_sizes)
        if total_gap == 0:
            return torch.tensor([], dtype=torch.long)
        
        collocation_indices = []
        
        for i, gap_size in enumerate(gap_sizes):
            if gap_size <= 1:
                continue
            
            # Allocate points proportional to gap size
            n_points = max(1, int(self.max_points_per_trajectory * gap_size / total_gap))
            n_points = min(n_points, gap_size)
            
            start_idx = obs_indices[i]
            end_idx = obs_indices[i + 1]
            
            gap_points = torch.linspace(
                start_idx + 1, end_idx - 1, n_points, dtype=torch.float32
            ).long()
            collocation_indices.extend(gap_points.tolist())
        
        return torch.tensor(collocation_indices, dtype=torch.long)
    
    def _random_sampling(self, times: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
        """Sample points randomly between observations."""
        # Find all non-observation indices
        non_obs_indices = torch.where(~observation_mask)[0]
        
        if len(non_obs_indices) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Random sample up to max points
        n_points = min(len(non_obs_indices), self.max_points_per_trajectory)
        
        if n_points == 0:
            return torch.tensor([], dtype=torch.long)
        
        selected_indices = torch.randperm(len(non_obs_indices))[:n_points]
        return non_obs_indices[selected_indices]