"""
PyTorch dataset for orbital trajectory data.

This module provides:
- OrbitalDataset class for loading and batching trajectory data
- Collate functions for variable-length sequences
- Data transforms and augmentations
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np


class OrbitalDataset(Dataset):
    """
    PyTorch dataset for orbital trajectory data.
    
    Handles variable-length trajectories with observation masks
    and provides efficient batching for training.
    """
    
    def __init__(self, 
                 trajectories: List[Dict[str, torch.Tensor]],
                 transform: Optional[Callable] = None,
                 max_sequence_length: Optional[int] = None):
        """
        Initialize orbital dataset.
        
        Args:
            trajectories: List of trajectory dictionaries from OrbitalDataGenerator
            transform: Optional transform function to apply to samples
            max_sequence_length: Maximum sequence length (for truncation/padding)
        """
        self.trajectories = trajectories
        self.transform = transform
        self.max_sequence_length = max_sequence_length
        
        # Pre-process trajectories
        self._preprocess_trajectories()
    
    def _preprocess_trajectories(self):
        """Pre-process trajectories for efficient loading."""
        processed_trajectories = []
        
        for i, traj in enumerate(self.trajectories):
            try:
                # Ensure all tensors are on CPU for storage efficiency
                processed_traj = {}
                for key, value in traj.items():
                    if isinstance(value, torch.Tensor):
                        processed_traj[key] = value.cpu()
                    else:
                        processed_traj[key] = value
                
                # Apply sequence length limits if specified
                if self.max_sequence_length is not None:
                    seq_len = processed_traj['times'].shape[0]
                    if seq_len > self.max_sequence_length:
                        # Truncate to max length
                        for key in ['times', 'states', 'observations', 'observation_mask']:
                            if key in processed_traj:
                                processed_traj[key] = processed_traj[key][:self.max_sequence_length]
                
                processed_trajectories.append(processed_traj)
                
            except Exception as e:
                print(f"Warning: Skipping trajectory {i} due to preprocessing error: {e}")
                continue
        
        self.trajectories = processed_trajectories
        print(f"Dataset initialized with {len(self.trajectories)} trajectories")
    
    def __len__(self) -> int:
        """Return number of trajectories in dataset."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get trajectory sample by index.
        
        Args:
            idx: Trajectory index
            
        Returns:
            Dictionary containing trajectory data
        """
        if idx >= len(self.trajectories):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.trajectories)}")
        
        # Get trajectory
        traj = self.trajectories[idx].copy()
        
        # Apply transform if provided
        if self.transform is not None:
            traj = self.transform(traj)
        
        return traj
    
    def get_sequence_lengths(self) -> List[int]:
        """Get list of sequence lengths for all trajectories."""
        return [traj['times'].shape[0] for traj in self.trajectories]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        seq_lengths = self.get_sequence_lengths()
        
        # Compute state statistics
        all_states = torch.cat([traj['states'] for traj in self.trajectories], dim=0)
        all_observations = torch.cat([traj['observations'] for traj in self.trajectories], dim=0)
        
        # Count total observations vs gaps
        all_masks = torch.cat([traj['observation_mask'] for traj in self.trajectories], dim=0)
        total_points = len(all_masks)
        observed_points = all_masks.sum().item()
        gap_rate = 1.0 - (observed_points / total_points)
        
        return {
            'n_trajectories': len(self.trajectories),
            'sequence_lengths': {
                'min': min(seq_lengths),
                'max': max(seq_lengths), 
                'mean': np.mean(seq_lengths),
                'std': np.std(seq_lengths)
            },
            'state_statistics': {
                'mean': all_states.mean(dim=0).tolist(),
                'std': all_states.std(dim=0).tolist(),
                'min': all_states.min(dim=0)[0].tolist(),
                'max': all_states.max(dim=0)[0].tolist()
            },
            'observation_statistics': {
                'total_points': total_points,
                'observed_points': observed_points,
                'gap_rate': gap_rate
            }
        }


def orbital_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length orbital trajectories.
    
    Args:
        batch: List of trajectory dictionaries
        
    Returns:
        Batched dictionary with padded sequences
    """
    if not batch:
        return {}
    
    # Get maximum sequence length in batch
    seq_lengths = [sample['times'].shape[0] for sample in batch]
    max_seq_len = max(seq_lengths)
    batch_size = len(batch)
    
    # Get dimensions
    state_dim = batch[0]['states'].shape[-1] if 'states' in batch[0] else 6
    
    # Initialize batched tensors
    batched_data = {
        'times': torch.zeros(batch_size, max_seq_len),
        'states': torch.zeros(batch_size, max_seq_len, state_dim),
        'observations': torch.zeros(batch_size, max_seq_len, state_dim),
        'observation_mask': torch.zeros(batch_size, max_seq_len, dtype=torch.bool),
        'sequence_lengths': torch.tensor(seq_lengths, dtype=torch.long)
    }
    
    # Fill batched tensors with padding
    for i, sample in enumerate(batch):
        seq_len = seq_lengths[i]
        
        # Copy data up to sequence length
        batched_data['times'][i, :seq_len] = sample['times']
        batched_data['states'][i, :seq_len] = sample['states']
        batched_data['observations'][i, :seq_len] = sample['observations']
        batched_data['observation_mask'][i, :seq_len] = sample['observation_mask']
        
        # Padding is automatically zeros/False for the rest
    
    return batched_data


class TrajectoryTransforms:
    """Common transforms for trajectory data."""
    
    @staticmethod
    def normalize_states(trajectory: Dict[str, torch.Tensor], 
                        position_scale: float = 1e7,
                        velocity_scale: float = 1e4) -> Dict[str, torch.Tensor]:
        """
        Normalize position and velocity scales.
        
        Args:
            trajectory: Trajectory dictionary
            position_scale: Scale factor for positions (default: 10 Mm)
            velocity_scale: Scale factor for velocities (default: 10 km/s)
            
        Returns:
            Trajectory with normalized states
        """
        traj = trajectory.copy()
        
        # Normalize states
        if 'states' in traj:
            states = traj['states'].clone()
            states[:, :3] = states[:, :3] / position_scale  # Positions
            states[:, 3:] = states[:, 3:] / velocity_scale  # Velocities
            traj['states'] = states
        
        # Normalize observations
        if 'observations' in traj:
            observations = traj['observations'].clone()
            observations[:, :3] = observations[:, :3] / position_scale
            observations[:, 3:] = observations[:, 3:] / velocity_scale
            traj['observations'] = observations
        
        return traj
    
    @staticmethod
    def add_noise_augmentation(trajectory: Dict[str, torch.Tensor],
                              noise_std: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Add noise augmentation to observations.
        
        Args:
            trajectory: Trajectory dictionary  
            noise_std: Standard deviation of additive noise
            
        Returns:
            Trajectory with augmented observations
        """
        traj = trajectory.copy()
        
        if 'observations' in traj:
            observations = traj['observations'].clone()
            mask = traj.get('observation_mask', torch.ones_like(observations[:, 0], dtype=torch.bool))
            
            # Add noise only to observed points
            noise = torch.randn_like(observations) * noise_std
            observations[mask] = observations[mask] + noise[mask]
            
            traj['observations'] = observations
        
        return traj
    
    @staticmethod
    def temporal_subsample(trajectory: Dict[str, torch.Tensor],
                          subsample_rate: int = 2) -> Dict[str, torch.Tensor]:
        """
        Temporally subsample trajectory.
        
        Args:
            trajectory: Trajectory dictionary
            subsample_rate: Take every nth time step
            
        Returns:
            Subsampled trajectory
        """
        traj = trajectory.copy()
        
        # Subsample all time-series data
        for key in ['times', 'states', 'observations', 'observation_mask']:
            if key in traj:
                traj[key] = traj[key][::subsample_rate]
        
        return traj


def create_orbital_dataloader(trajectories: List[Dict[str, torch.Tensor]],
                             batch_size: int = 32,
                             shuffle: bool = True,
                             num_workers: int = 4,
                             transform: Optional[Callable] = None,
                             max_sequence_length: Optional[int] = None) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for orbital trajectory data.
    
    Args:
        trajectories: List of trajectory dictionaries
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transform function
        max_sequence_length: Maximum sequence length
        
    Returns:
        Configured DataLoader
    """
    dataset = OrbitalDataset(
        trajectories=trajectories,
        transform=transform,
        max_sequence_length=max_sequence_length
    )
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=orbital_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if shuffle else False
    )