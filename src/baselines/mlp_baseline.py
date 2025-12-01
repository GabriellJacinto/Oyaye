"""
MLP Baseline Implementation.

Pure Multi-Layer Perceptron without spiking dynamics for comparison
with NP-SNN. This tests whether the spiking mechanism provides benefits
over standard neural networks for orbital prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from .sgp4_baseline import BaselineModel


class TimeEncoder(nn.Module):
    """Time encoding module for MLP baseline."""
    
    def __init__(self, 
                 output_dim: int = 64,
                 max_time: float = 24.0,
                 encoding_type: str = 'fourier'):
        """
        Initialize time encoder.
        
        Args:
            output_dim: Output dimension
            max_time: Maximum time in hours for normalization
            encoding_type: 'fourier' or 'learned'
        """
        super().__init__()
        self.output_dim = output_dim
        self.max_time = max_time
        self.encoding_type = encoding_type
        
        if encoding_type == 'fourier':
            # Fourier feature encoding
            self.freq_bands = nn.Parameter(
                torch.logspace(-1, 2, output_dim // 2), 
                requires_grad=False
            )
        elif encoding_type == 'learned':
            # Learned time encoding
            self.time_mlp = nn.Sequential(
                nn.Linear(1, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, output_dim),
                nn.ReLU()
            )
    
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Encode time features.
        
        Args:
            times: Time tensor (..., 1) in hours
            
        Returns:
            Time encoding (..., output_dim)
        """
        # Normalize time
        normalized_times = times / self.max_time
        
        if self.encoding_type == 'fourier':
            # Fourier features
            angles = 2 * np.pi * normalized_times * self.freq_bands
            cos_features = torch.cos(angles)
            sin_features = torch.sin(angles)
            encoding = torch.cat([cos_features, sin_features], dim=-1)
        
        elif self.encoding_type == 'learned':
            # Learned encoding
            encoding = self.time_mlp(normalized_times)
        
        return encoding


class MLPCore(nn.Module):
    """Core MLP network for trajectory prediction."""
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dims: List[int] = [128, 128, 64],
                 output_dim: int = 6,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize MLP core.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (6 for state)
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super().__init__()
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.network(x)


class MLPBaseline(BaselineModel, nn.Module):
    """
    MLP baseline model for orbital trajectory prediction.
    
    Pure neural network approach without spiking dynamics,
    using time encoding and standard MLP architecture.
    """
    
    def __init__(self,
                 time_encoding_dim: int = 64,
                 hidden_dims: List[int] = [128, 128, 64],
                 encoding_type: str = 'fourier',
                 dropout: float = 0.1,
                 max_time: float = 24.0,
                 uncertainty_head: bool = True,
                 device: str = 'cpu'):
        """
        Initialize MLP baseline.
        
        Args:
            time_encoding_dim: Time encoding dimension
            hidden_dims: Hidden layer dimensions
            encoding_type: Time encoding type ('fourier' or 'learned')
            dropout: Dropout probability
            max_time: Maximum time for normalization
            uncertainty_head: Include uncertainty prediction
            device: Device for computation
        """
        super(MLPBaseline, self).__init__()
        
        self.time_encoding_dim = time_encoding_dim
        self.hidden_dims = hidden_dims
        self.encoding_type = encoding_type
        self.dropout = dropout
        self.max_time = max_time
        self.uncertainty_head = uncertainty_head
        self.device = torch.device(device)
        
        # Time encoder
        self.time_encoder = TimeEncoder(
            output_dim=time_encoding_dim,
            max_time=max_time,
            encoding_type=encoding_type
        )
        
        # Main MLP network
        self.mlp_core = MLPCore(
            input_dim=time_encoding_dim,
            hidden_dims=hidden_dims,
            output_dim=6,  # State prediction
            dropout=dropout
        )
        
        # Uncertainty head (if enabled)
        if uncertainty_head:
            self.uncertainty_mlp = MLPCore(
                input_dim=time_encoding_dim,
                hidden_dims=hidden_dims[:-1],  # Smaller for uncertainty
                output_dim=6,  # Log-variance for each state component
                dropout=dropout
            )
        
        # Move to device
        self.to(self.device)
    
    def forward(self, times: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through MLP.
        
        Args:
            times: Time tensor (batch_size, 1) in hours
            
        Returns:
            State predictions or (predictions, uncertainties)
        """
        # Encode time
        time_features = self.time_encoder(times)
        
        # Predict state
        state_pred = self.mlp_core(time_features)
        
        if self.uncertainty_head:
            # Predict log-variance (for numerical stability)
            log_var = self.uncertainty_mlp(time_features)
            uncertainty = torch.exp(0.5 * log_var)  # Convert to std
            
            return state_pred, uncertainty
        else:
            return state_pred
    
    def predict_trajectory(self, 
                          initial_state: np.ndarray,
                          times: np.ndarray,
                          **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict trajectory using trained MLP.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz]
            times: Time array in hours from initial epoch
            
        Returns:
            Tuple of (predicted_states, uncertainties)
        """
        self.eval()
        
        with torch.no_grad():
            # Convert times to tensor
            times_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)
            times_tensor = times_tensor.unsqueeze(-1)  # Add feature dimension
            
            # Forward pass
            if self.uncertainty_head:
                state_pred, uncertainty_pred = self.forward(times_tensor)
                uncertainties = uncertainty_pred.cpu().numpy()
            else:
                state_pred = self.forward(times_tensor)
                uncertainties = None
            
            # Convert predictions to numpy
            predicted_states = state_pred.cpu().numpy()
            
            # Add initial state offset (model predicts relative to initial)
            # This is a simple approach - more sophisticated methods exist
            for i in range(len(times)):
                if times[i] == 0:
                    predicted_states[i] = initial_state
                else:
                    # Simple interpolation/extrapolation from initial state
                    # In practice, this would be learned during training
                    predicted_states[i] = initial_state + predicted_states[i]
        
        return predicted_states, uncertainties
    
    def train_on_data(self,
                     training_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     validation_data: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
                     epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 1e-3,
                     patience: int = 10) -> Dict[str, List[float]]:
        """
        Train MLP on trajectory data.
        
        Args:
            training_data: List of (initial_state, times, true_states) tuples
            validation_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        self.train()
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if validation_data else None
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training loop
            self.train()
            train_losses = []
            
            # Create batches (simplified - would use DataLoader in practice)
            np.random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Prepare batch data
                batch_times = []
                batch_targets = []
                
                for initial_state, times, true_states in batch:
                    # Add all time points from this trajectory
                    batch_times.extend(times.tolist())
                    batch_targets.extend(true_states.tolist())
                
                if len(batch_times) == 0:
                    continue
                
                # Convert to tensors
                times_tensor = torch.tensor(batch_times, dtype=torch.float32, device=self.device)
                targets_tensor = torch.tensor(batch_targets, dtype=torch.float32, device=self.device)
                
                times_tensor = times_tensor.unsqueeze(-1)
                
                # Forward pass
                optimizer.zero_grad()
                
                if self.uncertainty_head:
                    predictions, pred_uncertainty = self.forward(times_tensor)
                    
                    # Loss with uncertainty weighting
                    mse = torch.mean((predictions - targets_tensor)**2)
                    
                    # Uncertainty regularization (simplified)
                    uncertainty_loss = torch.mean(pred_uncertainty)
                    
                    loss = mse + 0.01 * uncertainty_loss
                else:
                    predictions = self.forward(times_tensor)
                    loss = mse_loss(predictions, targets_tensor)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Average training loss
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            history['train_loss'].append(avg_train_loss)
            
            # Validation loop
            if validation_data:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    for initial_state, times, true_states in validation_data:
                        times_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)
                        targets_tensor = torch.tensor(true_states, dtype=torch.float32, device=self.device)
                        
                        times_tensor = times_tensor.unsqueeze(-1)
                        
                        if self.uncertainty_head:
                            predictions, _ = self.forward(times_tensor)
                        else:
                            predictions = self.forward(times_tensor)
                        
                        val_loss = mse_loss(predictions, targets_tensor)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses) if val_losses else 0.0
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                val_str = f", Val: {avg_val_loss:.6f}" if validation_data else ""
                print(f"Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.6f}{val_str}")
        
        return history
    
    def get_name(self) -> str:
        """Return baseline model name."""
        return f"MLP_{self.encoding_type}"


def test_mlp_baseline():
    """Test MLP baseline implementation."""
    
    print("🧪 Testing MLP Baseline...")
    
    # Create test initial state
    initial_state = np.array([
        6.778e6, 0.0, 0.0,        # position (m)
        0.0, 7660.0, 0.0          # velocity (m/s)
    ])
    
    # Test times
    times = np.linspace(0, 2, 25)  # hours
    
    # Initialize baseline
    mlp = MLPBaseline(
        time_encoding_dim=32,
        hidden_dims=[64, 32],
        uncertainty_head=True
    )
    
    print(f"✅ MLP model created: {mlp.get_name()}")
    print(f"   Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    # Test forward pass (without training)
    predicted_states, uncertainties = mlp.predict_trajectory(initial_state, times)
    
    print(f"   Prediction shape: {predicted_states.shape}")
    if uncertainties is not None:
        print(f"   Uncertainty shape: {uncertainties.shape}")
        print(f"   Avg uncertainty: ±{np.mean(uncertainties):.2f}")
    
    # Note: This is just structure testing - actual performance requires training
    print("   Note: Performance requires training on orbital data")
    
    return predicted_states, uncertainties


if __name__ == "__main__":
    test_mlp_baseline()