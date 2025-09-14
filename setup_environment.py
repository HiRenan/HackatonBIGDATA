#!/usr/bin/env python3
"""
Environment Setup Script for Hackathon Forecast Big Data 2025
Automated setup of development environment with optimizations
"""

import subprocess
import sys
import os
import psutil
from pathlib import Path

def check_system_requirements():
    """Check if system meets minimum requirements"""
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 50)
    
    # RAM Check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 8:
        print("WARNING: Less than 8GB RAM detected. Consider using Google Colab Pro.")
    elif ram_gb >= 16:
        print("EXCELLENT: 16GB+ RAM available for optimal processing")
    else:
        print("GOOD: Sufficient RAM for chunked processing")
    
    # Storage Check
    free_space_gb = psutil.disk_usage('.').free / (1024**3)
    print(f"Available storage: {free_space_gb:.1f} GB")
    
    if free_space_gb < 20:
        print("WARNING: Less than 20GB free space. Clean up disk space.")
    else:
        print("GOOD: Sufficient storage available")
    
    # CPU Check
    cpu_count = psutil.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    return {
        'ram_gb': ram_gb,
        'storage_gb': free_space_gb,
        'cpu_count': cpu_count,
        'meets_requirements': ram_gb >= 8 and free_space_gb >= 20
    }

def create_conda_env():
    """Create optimized conda environment"""
    print("\nCREATING CONDA ENVIRONMENT")
    print("=" * 50)
    
    env_name = "hackathon_forecast_2025"
    
    # Check if conda is available
    try:
        subprocess.run(['conda', '--version'], check=True, capture_output=True)
        print("Conda detected. Creating optimized environment...")
        
        # Create environment with Python 3.10 for optimal performance
        create_cmd = [
            'conda', 'create', '-n', env_name, 
            'python=3.10', '-y'
        ]
        subprocess.run(create_cmd, check=True)
        
        # Install conda-forge packages for performance
        conda_packages = [
            'conda', 'install', '-n', env_name, '-c', 'conda-forge',
            'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'lightgbm',
            'matplotlib', 'seaborn', 'jupyter', '-y'
        ]
        subprocess.run(conda_packages, check=True)
        
        print(f"Environment '{env_name}' created successfully!")
        print(f"Activate with: conda activate {env_name}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Conda not found. Will use pip installation instead.")
        return False

def install_requirements():
    """Install Python requirements with pip"""
    print("\nINSTALLING REQUIREMENTS")
    print("=" * 50)
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("ERROR: requirements.txt not found!")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True)
        
        # Install requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        print("All requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing requirements: {e}")
        return False

def setup_memory_optimizations():
    """Setup memory and performance optimizations"""
    print("\nSETTING UP OPTIMIZATIONS")
    print("=" * 50)
    
    # Create config directory
    config_dir = Path("src/config")
    config_dir.mkdir(exist_ok=True)
    
    # Memory optimization config
    memory_config = '''# Memory Optimization Config
import os
import pandas as pd
import numpy as np

# Environment variables for performance
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

# Pandas optimizations
pd.options.mode.copy_on_write = True
pd.options.compute.use_numba = True

# LightGBM config
LIGHTGBM_CONFIG = {
    'max_bin': 255,
    'num_threads': 8,
    'device_type': 'cpu',
    'verbose': -1
}

# Memory thresholds
MEMORY_THRESHOLDS = {
    'chunk_size': 100000,  # Process 100k rows at a time
    'max_memory_gb': 8,    # Maximum memory usage
    'sample_size': 50000   # Sample size for large files
}
'''
    
    config_file = config_dir / "memory_config.py"
    config_file.write_text(memory_config)
    
    print("Memory optimizations configured!")

def verify_installation():
    """Verify critical packages are installed correctly"""
    print("\nVERIFYING INSTALLATION")
    print("=" * 50)
    
    critical_packages = [
        'pandas', 'numpy', 'pyarrow', 'lightgbm', 
        'prophet', 'sklearn', 'mlflow'
    ]
    
    success_count = 0
    
    for package in critical_packages:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✓ {package}: {version}")
            success_count += 1
            
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
    
    print(f"\nVerification: {success_count}/{len(critical_packages)} packages working")
    return success_count == len(critical_packages)

def main():
    """Main setup function"""
    print("HACKATHON FORECAST BIG DATA 2025")
    print("ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check system requirements
    sys_info = check_system_requirements()
    
    if not sys_info['meets_requirements']:
        print("\nWARNING: System may not meet minimum requirements!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Try conda environment creation
    conda_success = create_conda_env()
    
    # Install requirements
    pip_success = install_requirements()
    
    # Setup optimizations
    setup_memory_optimizations()
    
    # Verify installation
    verification_success = verify_installation()
    
    print("\nSETUP SUMMARY")
    print("=" * 50)
    print(f"System Requirements: {'✓' if sys_info['meets_requirements'] else '⚠'}")
    print(f"Conda Environment: {'✓' if conda_success else '✗'}")
    print(f"Pip Installation: {'✓' if pip_success else '✗'}")
    print(f"Package Verification: {'✓' if verification_success else '✗'}")
    
    if pip_success and verification_success:
        print("\n🎉 SETUP COMPLETE! Ready for Phase 2!")
        print("\nNext steps:")
        if conda_success:
            print("1. conda activate hackathon_forecast_2025")
        print("2. jupyter lab (to start development)")
        print("3. Run Phase 2 EDA notebooks")
    else:
        print("\n❌ Setup incomplete. Check errors above.")

if __name__ == "__main__":
    main()