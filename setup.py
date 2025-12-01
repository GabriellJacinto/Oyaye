#!/usr/bin/env python3
"""
Setup script for NP-SNN project.

This script:
1. Validates the project structure
2. Creates necessary directories
3. Sets up Python environment 
4. Downloads initial data (if needed)
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def check_python_version():
    """Check that Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_directories(project_root: Path):
    """Create necessary directories."""
    directories = [
        "data",
        "data/raw", 
        "data/processed",
        "data/experiments",
        "logs",
        "models",
        "results",
        "plots",
        "mlruns"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def install_dependencies(requirements_file: Path, use_conda: bool = False):
    """Install Python dependencies."""
    if not requirements_file.exists():
        print(f"Warning: Requirements file not found: {requirements_file}")
        return
        
    if use_conda:
        cmd = ["conda", "install", "--file", str(requirements_file), "-y"]
    else:
        cmd = ["pip", "install", "-r", str(requirements_file)]
    
    try:
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.run(cmd, check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("You may need to install dependencies manually.")

def setup_environment_file(project_root: Path):
    """Create .env file with default settings."""
    env_file = project_root / ".env"
    
    if env_file.exists():
        print("✓ .env file already exists")
        return
        
    env_content = """# NP-SNN Project Environment Variables

# MLflow tracking
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns

# Data paths
DATA_ROOT=./data
RESULTS_ROOT=./results

# Compute settings
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4

# Logging
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    print("✓ Created .env file with default settings")

def validate_project_structure(project_root: Path):
    """Validate that project structure is correct."""
    required_files = [
        "src/__init__.py",
        "configs/space_debris_simulation.yaml",
        "requirements.txt"
    ]
    
    required_dirs = [
        "src/data",
        "src/models", 
        "src/physics",
        "src/train",
        "src/eval",
        "src/infra"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
            
    for dir_path in required_dirs:
        if not (project_root / dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print("Error: Project structure validation failed!")
        if missing_files:
            print(f"Missing files: {missing_files}")
        if missing_dirs:
            print(f"Missing directories: {missing_dirs}")
        return False
        
    print("✓ Project structure validation passed")
    return True

def check_git_setup(project_root: Path):
    """Check git repository setup."""
    git_dir = project_root / ".git"
    
    if not git_dir.exists():
        print("Warning: Not a git repository. Consider running 'git init'")
        return False
        
    print("✓ Git repository detected")
    
    # Check for initial commit
    try:
        result = subprocess.run(
            ["git", "log", "--oneline"], 
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Note: No commits found. Consider making an initial commit.")
    except subprocess.CalledProcessError:
        pass
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup NP-SNN project")
    parser.add_argument("--conda", action="store_true", 
                       help="Use conda instead of pip for dependencies")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    print(f"Setting up NP-SNN project in: {project_root}")
    
    # Check Python version
    check_python_version()
    
    # Validate project structure
    if not validate_project_structure(project_root):
        print("Please fix project structure issues before continuing.")
        sys.exit(1)
    
    # Create directories
    create_directories(project_root)
    
    # Setup environment
    setup_environment_file(project_root)
    
    # Install dependencies (if not skipped)
    if not args.skip_deps:
        requirements_file = project_root / "requirements.txt"
        install_dependencies(requirements_file, args.conda)
    
    # Check git setup
    check_git_setup(project_root)
    
    print("\n" + "="*50)
    print("✓ NP-SNN project setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate your Python environment")
    print("2. Run tests: python -m pytest tests/")
    print("3. Generate sample data: python -m src.data.generators")
    print("4. Start training: python -m src.train.train_loop")
    print("="*50)

if __name__ == "__main__":
    main()