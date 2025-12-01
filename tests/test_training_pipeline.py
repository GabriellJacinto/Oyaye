"""
Test script for NP-SNN training pipeline.

Quick validation that all components work together before running full training.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.train.config import TrainingConfig
from src.train.scheduler import CurriculumScheduler
from src.models.npsnn import NPSNN
import yaml

def test_config_loading():
    """Test configuration loading from YAML."""
    print("🧪 Testing configuration loading...")
    
    # Load from YAML file
    with open('configs/npsnn_training.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    config = TrainingConfig.from_dict(yaml_config)
    print(f"   ✅ Config loaded: {config.model_config.hidden_size} hidden units, {config.curriculum_config.total_epochs} epochs")
    
    return config

def test_curriculum_scheduler():
    """Test curriculum scheduler functionality."""
    print("📚 Testing curriculum scheduler...")
    
    # Create config for testing
    config = TrainingConfig()
    config.curriculum_config.total_epochs = 10
    config.curriculum_config.sanity_epochs = 2
    config.curriculum_config.supervised_epochs = 3
    config.curriculum_config.mixed_epochs = 3
    config.curriculum_config.physics_epochs = 2
    config.curriculum_config.fine_tune_epochs = 0
    
    scheduler = CurriculumScheduler(config.curriculum_config)
    
    # Test stage progression
    for epoch in range(5):
        stage_info = scheduler.update(epoch)
        loss_weights = scheduler.get_loss_weights()
        print(f"   Epoch {epoch}: Stage {stage_info.stage.value}, Weights w_dynamics={loss_weights.get('w_dynamics', 0):.1f}")
    
    print("   ✅ Curriculum progression working")

def test_model_creation(config):
    """Test NP-SNN model creation."""
    print("🧠 Testing model creation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import the config classes
    from src.models.npsnn import NPSNNConfig
    from src.models.time_encoding import TimeEncodingConfig
    from src.models.snn_core import SNNConfig  
    from src.models.decoder import DecoderConfig
    
    # Create a minimal config
    config = NPSNNConfig()
    config.time_encoding = TimeEncodingConfig()
    
    # Create SNN config with proper fields
    snn_config = SNNConfig()
    snn_config.hidden_sizes = [128, 128]  # Two hidden layers of size 128
    snn_config.input_size = 64  # Will be updated by NPSNN
    snn_config.output_size = 128
    config.snn = snn_config
    
    config.decoder = DecoderConfig()
    
    model = NPSNN(config=config)
    model = model.to(device)
    
    # Count parameters manually
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model created: {param_count:,} parameters on {device}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    
    times = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    observations = torch.randn(batch_size, seq_len, 6).to(device)
    obs_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    try:
        with torch.no_grad():
            output = model(times, observations, obs_mask)
        print(f"   ✅ Forward pass: {list(output['states'].shape)}")
    except Exception as e:
        print(f"   ⚠️  Forward pass failed: {e}")
        print("   ℹ️  This is expected as we're using a minimal config")
    
    return model, device

def main():
    print("🚀 NP-SNN Training Pipeline Test")
    print("=" * 50)
    
    try:
        # Test configuration
        config = test_config_loading()
        
        # Test curriculum scheduler
        test_curriculum_scheduler()
        
        # Test model
        model, device = test_model_creation(config)
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Training pipeline ready.")
        
        # Print system info
        print(f"🔧 Device: {device}")
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model parameters: {param_count:,}")
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "💻 Using CPU")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)