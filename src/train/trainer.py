"""
Main training loop for NP-SNN orbital mechanics model.

Implements the complete training pipeline with:
- Curriculum learning progression
- Physics-informed and supervised loss combination  
- MLflow experiment tracking
- Dynamic learning rate scheduling
- Checkpoint management
- Validation and early stopping

Follows implementation plan sections 4.1-4.5 for production-grade training.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Iterator
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from ..models.npsnn import NPSNN
from ..data.orbital_dataset import OrbitalDataset
from ..train.losses import PhysicsInformedLossFunction
from .config import TrainingConfig, ExperimentConfig, CurriculumConfig
from .scheduler import CurriculumScheduler, CollocationSampler


class NPSNNTrainer:
    """
    Main trainer class for NP-SNN orbital mechanics model.
    
    Handles:
    - Curriculum learning with stage transitions
    - Combined physics + supervised training
    - Experiment tracking with MLflow
    - Checkpointing and resuming
    - Validation and early stopping
    """
    
    def __init__(self, 
                 config: TrainingConfig,
                 model: NPSNN,
                 train_dataset: OrbitalDataset,
                 val_dataset: OrbitalDataset,
                 device: torch.device):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model: NP-SNN model to train
            train_dataset: Training dataset  
            val_dataset: Validation dataset
            device: Training device (CPU/GPU)
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Datasets and data loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._setup_data_loaders()
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        self.loss_function = PhysicsInformedLossFunction(config.physics_config)
        
        # Curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(config.curriculum_config)
        self.collocation_sampler = CollocationSampler(
            strategy="adaptive",
            max_points_per_trajectory=config.curriculum_config.max_collocation_points
        )
        
        # Tracking
        self.current_epoch = 0
        self.training_history = []
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # MLflow experiment
        self.experiment_id = None
        self.run_id = None
        
    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data_config.batch_size,
            shuffle=True,
            num_workers=self.config.data_config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data_config.val_batch_size,
            shuffle=False,
            num_workers=self.config.data_config.num_workers,
            pin_memory=True
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config.optimization_config
        
        if opt_config.optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif opt_config.optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=0.9,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.optimizer_type}")
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        opt_config = self.config.optimization_config
        
        if opt_config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.curriculum_config.total_epochs,
                eta_min=opt_config.learning_rate * 0.01
            )
        elif opt_config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        elif opt_config.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def setup_mlflow_experiment(self) -> str:
        """Setup MLflow experiment and start run."""
        exp_config = self.config.experiment_config
        
        # Set tracking URI
        if exp_config.tracking_uri:
            mlflow.set_tracking_uri(exp_config.tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.create_experiment(exp_config.experiment_name)
            self.experiment_id = experiment
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(exp_config.experiment_name)
            self.experiment_id = experiment.experiment_id
        
        # Start run
        mlflow.start_run(experiment_id=self.experiment_id)
        self.run_id = mlflow.active_run().info.run_id
        
        # Log configuration
        self._log_config_to_mlflow()
        
        return self.run_id
    
    def _log_config_to_mlflow(self):
        """Log training configuration to MLflow."""
        # Log model parameters
        mlflow.log_param("model_type", "NP-SNN")
        mlflow.log_param("model_parameters", sum(p.numel() for p in self.model.parameters()))
        
        # Log training config
        mlflow.log_param("batch_size", self.config.data_config.batch_size)
        mlflow.log_param("learning_rate", self.config.optimization_config.learning_rate)
        mlflow.log_param("optimizer", self.config.optimization_config.optimizer_type)
        mlflow.log_param("lr_scheduler", self.config.optimization_config.lr_scheduler)
        
        # Log curriculum config
        mlflow.log_param("total_epochs", self.config.curriculum_config.total_epochs)
        mlflow.log_param("curriculum_stages", 5)
        mlflow.log_param("early_stopping_patience", self.config.curriculum_config.early_stopping_patience)
        
        # Log physics config
        mlflow.log_param("physics_weight_max", max(self.config.curriculum_config.stage_configs["physics"].values()))
        mlflow.log_param("adaptive_weights", self.config.physics_config.adaptive_weights)
        
        # Log dataset info
        mlflow.log_param("train_size", len(self.train_dataset))
        mlflow.log_param("val_size", len(self.val_dataset))
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop with curriculum learning.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results summary
        """
        print(f"🚀 Starting NP-SNN training with curriculum learning")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 Device: {self.device}")
        print(f"📚 Training samples: {len(self.train_dataset):,}")
        print(f"🔍 Validation samples: {len(self.val_dataset):,}")
        
        # Setup MLflow tracking
        run_id = self.setup_mlflow_experiment()
        print(f"📈 MLflow run: {run_id}")
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
            print(f"🔄 Resumed from epoch {start_epoch}")
        
        try:
            # Main training loop
            for epoch in range(start_epoch, self.config.curriculum_config.total_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Update curriculum scheduler
                stage_info = self.curriculum_scheduler.update(epoch)
                
                # Log curriculum state
                curriculum_summary = self.curriculum_scheduler.get_stage_summary()
                self._log_curriculum_state(curriculum_summary)
                
                # Adjust learning rate based on curriculum
                lr_scale = self.curriculum_scheduler.get_learning_rate_scale()
                self._adjust_learning_rate(lr_scale)
                
                # Training step
                train_metrics = self._train_epoch()
                
                # Validation step
                val_metrics = self._validate_epoch()
                
                # Update curriculum with validation loss
                stage_info = self.curriculum_scheduler.update(epoch, val_metrics['total_loss'])
                
                # Learning rate scheduler step
                if self.lr_scheduler:
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics['total_loss'])
                    else:
                        self.lr_scheduler.step()
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time, curriculum_summary)
                
                # Save checkpoint
                if epoch % self.config.experiment_config.checkpoint_frequency == 0:
                    self.save_checkpoint(epoch)
                
                # Early stopping check
                if self.curriculum_scheduler.should_stop_early():
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break
                
                # Update best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # Save best model to MLflow
                    self._save_best_model_to_mlflow()
            
            # Final model save
            self.save_checkpoint(self.current_epoch, is_final=True)
            
            # Training complete
            training_summary = self._create_training_summary()
            print(f"✅ Training completed! Best validation loss: {self.best_val_loss:.6f}")
            
            return training_summary
            
        except KeyboardInterrupt:
            print(f"⏹️  Training interrupted at epoch {self.current_epoch}")
            self.save_checkpoint(self.current_epoch, is_interrupted=True)
            return {"status": "interrupted", "epoch": self.current_epoch}
            
        finally:
            # End MLflow run
            if mlflow.active_run():
                mlflow.end_run()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Get current curriculum state
        loss_weights = self.curriculum_scheduler.get_loss_weights()
        data_complexity = self.curriculum_scheduler.get_data_complexity()
        
        # Metrics tracking
        epoch_metrics = {
            'total_loss': 0.0,
            'supervised_loss': 0.0,
            'physics_loss': 0.0,
            'n_batches': 0
        }
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            times = batch['times']  # (B, T)
            observations = batch['observations']  # (B, T, 6)
            observation_mask = batch['observation_mask']  # (B, T)
            
            model_output = self.model(times, observations, observation_mask)
            
            # Compute losses
            loss_components = self._compute_batch_losses(
                batch, model_output, loss_weights
            )
            
            total_loss = loss_components['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.optimization_config.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization_config.gradient_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            batch_size = times.shape[0]
            for key in epoch_metrics:
                if key == 'n_batches':
                    epoch_metrics[key] += 1
                else:
                    epoch_metrics[key] += loss_components.get(key, 0.0) * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'stage': self.curriculum_scheduler.current_stage.value
            })
        
        # Average metrics over batches
        n_samples = len(self.train_dataset)
        for key in epoch_metrics:
            if key != 'n_batches':
                epoch_metrics[key] /= n_samples
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        loss_weights = self.curriculum_scheduler.get_loss_weights()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'supervised_loss': 0.0,
            'physics_loss': 0.0,
            'n_batches': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                times = batch['times']
                observations = batch['observations']
                observation_mask = batch['observation_mask']
                
                model_output = self.model(times, observations, observation_mask)
                
                # Compute losses
                loss_components = self._compute_batch_losses(
                    batch, model_output, loss_weights
                )
                
                # Update metrics
                batch_size = times.shape[0]
                for key in epoch_metrics:
                    if key == 'n_batches':
                        epoch_metrics[key] += 1
                    else:
                        epoch_metrics[key] += loss_components.get(key, 0.0) * batch_size
        
        # Average metrics
        n_samples = len(self.val_dataset)
        for key in epoch_metrics:
            if key != 'n_batches':
                epoch_metrics[key] /= n_samples
        
        return epoch_metrics
    
    def _compute_batch_losses(self, 
                            batch: Dict[str, torch.Tensor],
                            model_output: Dict[str, torch.Tensor],
                            loss_weights: Dict[str, float]) -> Dict[str, float]:
        """Compute all loss components for a batch."""
        times = batch['times']
        observations = batch['observations']
        observation_mask = batch['observation_mask']
        
        # Extract ground truth
        gt_states = batch['states']  # (B, T, 6)
        
        # Get predicted states
        pred_states = model_output['states']  # (B, T, 6)
        pred_uncertainties = model_output.get('uncertainties')  # (B, T, 6)
        
        loss_components = {'total_loss': 0.0}
        
        # Supervised loss (on observed states)
        if self.curriculum_scheduler.should_use_supervised_loss():
            supervised_loss = self.loss_function.compute_supervised_loss(
                pred_states, gt_states, observation_mask, pred_uncertainties
            )
            
            weighted_supervised = loss_weights['w_state_supervised'] * supervised_loss
            loss_components['supervised_loss'] = supervised_loss.item()
            loss_components['total_loss'] += weighted_supervised
        
        # Physics loss (on collocation points)
        if self.curriculum_scheduler.should_use_physics_loss():
            # Sample collocation points for each trajectory in batch
            collocation_losses = []
            
            for b in range(times.shape[0]):
                # Get collocation points for this trajectory
                traj_times = times[b]  # (T,)
                traj_obs_mask = observation_mask[b]  # (T,)
                
                collocation_indices = self.collocation_sampler.sample_points(
                    traj_times, traj_obs_mask
                )
                
                if len(collocation_indices) > 0:
                    # Extract states at collocation points
                    colloc_times = traj_times[collocation_indices]  # (C,)
                    colloc_states = pred_states[b, collocation_indices]  # (C, 6)
                    
                    # Compute physics loss
                    physics_loss = self.loss_function.compute_physics_loss(
                        colloc_times.unsqueeze(0),  # (1, C)
                        colloc_states.unsqueeze(0)  # (1, C, 6)
                    )
                    
                    collocation_losses.append(physics_loss)
            
            if collocation_losses:
                avg_physics_loss = torch.stack(collocation_losses).mean()
                weighted_physics = loss_weights['w_dynamics'] * avg_physics_loss
                loss_components['physics_loss'] = avg_physics_loss.item()
                loss_components['total_loss'] += weighted_physics
        
        return loss_components
    
    def _adjust_learning_rate(self, scale: float):
        """Adjust learning rate based on curriculum stage."""
        base_lr = self.config.optimization_config.learning_rate
        new_lr = base_lr * scale
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def _log_curriculum_state(self, curriculum_summary: Dict[str, Any]):
        """Log curriculum state to console."""
        if self.current_epoch % 10 == 0:  # Log every 10 epochs
            print(f"\n📚 Curriculum State (Epoch {self.current_epoch}):")
            print(f"   Stage: {curriculum_summary['stage']}")
            print(f"   Progress: {curriculum_summary['stage_progress']} (stage), {curriculum_summary['total_progress']} (total)")
            print(f"   Loss weights: {curriculum_summary['loss_weights']}")
            print(f"   LR scale: {curriculum_summary['lr_scale']}")
    
    def _log_epoch_metrics(self, 
                          epoch: int,
                          train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float],
                          epoch_time: float,
                          curriculum_summary: Dict[str, Any]):
        """Log metrics to MLflow and console."""
        # MLflow logging
        mlflow.log_metrics({
            "epoch": epoch,
            "train_loss": train_metrics['total_loss'],
            "val_loss": val_metrics['total_loss'],
            "train_supervised_loss": train_metrics.get('supervised_loss', 0),
            "val_supervised_loss": val_metrics.get('supervised_loss', 0),
            "train_physics_loss": train_metrics.get('physics_loss', 0),
            "val_physics_loss": val_metrics.get('physics_loss', 0),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time
        }, step=epoch)
        
        # Log curriculum state
        for key, value in curriculum_summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"curriculum_{key}", value, step=epoch)
        
        # Console logging (every 5 epochs)
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch:3d} | "
                  f"Train: {train_metrics['total_loss']:.4f} | "
                  f"Val: {val_metrics['total_loss']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Stage: {curriculum_summary['stage']}")
    
    def _save_best_model_to_mlflow(self):
        """Save current best model to MLflow."""
        # Save model state dict
        model_path = "best_model"
        mlflow.pytorch.log_state_dict(self.best_model_state, model_path)
        
        # Log validation loss as metric
        mlflow.log_metric("best_val_loss", self.best_val_loss)
    
    def save_checkpoint(self, epoch: int, is_final: bool = False, is_interrupted: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_state': self.curriculum_scheduler.save_state(),
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Determine filename
        if is_final:
            filename = 'final_checkpoint.pt'
        elif is_interrupted:
            filename = 'interrupted_checkpoint.pt'
        else:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        # Save locally
        checkpoint_dir = Path(self.config.experiment_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save to MLflow
        mlflow.log_artifact(str(checkpoint_path))
        
        print(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint and return starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Load curriculum state
        self.curriculum_scheduler.load_state(checkpoint['curriculum_state'])
        
        # Load training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_model_state = checkpoint.get('best_model_state')
        self.training_history = checkpoint.get('training_history', [])
        
        return checkpoint['epoch'] + 1  # Start from next epoch
    
    def _create_training_summary(self) -> Dict[str, Any]:
        """Create final training summary."""
        return {
            'status': 'completed',
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'final_stage': self.curriculum_scheduler.current_stage.value,
            'model_parameters': self.model.count_parameters(),
            'run_id': self.run_id,
            'training_history': self.training_history
        }