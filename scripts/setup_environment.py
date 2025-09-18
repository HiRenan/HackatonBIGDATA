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

def check_python_version():
    """Check Python version compatibility"""
    print("\nCHECKING PYTHON VERSION")
    print("=" * 50)

    current_version = sys.version_info
    current_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    print(f"Current Python version: {current_str}")

    # Recommended versions
    if current_version >= (3, 13):
        print("GOOD: Python 3.13+ detected - using optimized packages")
        print("   Using pre-compiled wheels for better compatibility")
        return True
    elif current_version >= (3, 10) and current_version < (3, 12):
        print("EXCELLENT: Python version is optimal for this project")
        return True
    elif current_version >= (3, 8):
        print("GOOD: Python version should work")
        return True
    else:
        print("ERROR: Python 3.8+ required")
        return False

def create_conda_env():
    """Create optimized conda environment"""
    print("\nCREATING CONDA ENVIRONMENT")
    print("=" * 50)

    env_name = "hackathon_forecast_2025"

    # Check if conda is available
    try:
        subprocess.run(['conda', '--version'], check=True, capture_output=True)
        print("Conda detected. Creating optimized environment...")

        # Create environment with Python 3.10 for maximum compatibility
        print("Using Python 3.10 for maximum compatibility")
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
        print("Note: This creates a Python 3.10 environment for compatibility")

        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Conda not found. Will use pip installation instead.")
        return False

def install_requirements():
    """Install Python requirements with pip"""
    print("\nINSTALLING REQUIREMENTS")
    print("=" * 50)

    # Check available requirements files
    requirements_files = {
        'core': Path("requirements.txt"),
        'py310': Path("requirements-py310.txt"),
        'py313': Path("requirements-py313.txt"),
        'optional': Path("requirements-optional.txt"),
        'dev': Path("requirements-dev.txt")
    }

    # Determine which requirements to use
    current_version = sys.version_info

    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                      check=True)

        # Choose requirements file based on Python version
        if current_version >= (3, 13) and requirements_files['py313'].exists():
            print("Using Python 3.13 specific requirements with pre-compiled wheels")
            requirements_file = requirements_files['py313']
        elif current_version >= (3, 10) and current_version < (3, 11) and requirements_files['py310'].exists():
            print("Using Python 3.10 specific requirements for maximum compatibility")
            requirements_file = requirements_files['py310']
        elif requirements_files['core'].exists():
            print("Using flexible core requirements")
            requirements_file = requirements_files['core']
        else:
            print("ERROR: No requirements.txt found!")
            return False

        # Install core requirements with optimizations
        print(f"Installing {requirements_file.name}...")
        pip_args = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file),
                   '--prefer-binary']

        # Check if packages are already installed to avoid unnecessary reinstall
        try:
            import pandas, numpy, pyarrow, lightgbm, sklearn, mlflow
            print("Core packages already installed, skipping installation...")
            return True
        except ImportError:
            pass

        subprocess.run(pip_args, check=True)

        # Ask about optional dependencies
        if requirements_files['optional'].exists():
            print("\nOptional dependencies available (PyTorch, TensorFlow, advanced features)")
            install_optional = input("Install optional dependencies? (y/N): ").lower() == 'y'
            if install_optional:
                print("Installing optional requirements...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-optional.txt'],
                              check=True)

        print("✅ Requirements installed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR installing requirements: {e}")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Try using Python 3.10 instead of 3.13")
        print("2. Create a fresh virtual environment")
        print("3. Use conda instead: conda create -n env_name python=3.10")
        print("4. Install core requirements only first")
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
            
            print(f"+ {package}: {version}")
            success_count += 1
            
        except ImportError:
            print(f"- {package}: NOT INSTALLED")
    
    print(f"\nVerification: {success_count}/{len(critical_packages)} packages working")
    return success_count == len(critical_packages)

def main():
    """Main setup function"""
    print("HACKATHON FORECAST BIG DATA 2025")
    print("ENVIRONMENT SETUP")
    print("=" * 60)

    # Check Python version first
    python_ok = check_python_version()

    # Check system requirements
    sys_info = check_system_requirements()

    if not sys_info['meets_requirements']:
        print("\nWARNING: System may not meet minimum requirements!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    if not python_ok:
        print("\nPython version warning detected!")
        print("   Recommendation: Use Python 3.10 for best compatibility")
        response = input("Continue with current Python version? (y/N): ")
        if response.lower() != 'y':
            print("\nTo install Python 3.10:")
            print("   • With pyenv: pyenv install 3.10.12 && pyenv local 3.10.12")
            print("   • With conda: conda create -n hackathon python=3.10")
            print("   • Download from: https://www.python.org/downloads/release/python-31012/")
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
    print(f"Python Version: {'OK' if python_ok else 'WARN'}")
    print(f"System Requirements: {'OK' if sys_info['meets_requirements'] else 'WARN'}")
    print(f"Conda Environment: {'OK' if conda_success else 'FAIL'}")
    print(f"Pip Installation: {'OK' if pip_success else 'FAIL'}")
    print(f"Package Verification: {'OK' if verification_success else 'FAIL'}")

    if pip_success and verification_success:
        print("\nSETUP COMPLETE! Ready for development!")
        print("\nNext steps:")
        if conda_success:
            print("1. conda activate hackathon_forecast_2025")
        print("2. jupyter lab (to start development)")
        print("3. Run notebooks for analysis")
        print("\nAvailable requirements:")
        print("   • requirements.txt - Core dependencies")
        print("   • requirements-py310.txt - Python 3.10 tested versions")
        print("   • requirements-py313.txt - Python 3.13 tested versions")
        print("   • requirements-optional.txt - Advanced features")
        print("   • requirements-dev.txt - Development tools")
    else:
        print("\nSetup incomplete. Check errors above.")
        print("\nNeed help? Check the troubleshooting section in README.md")

if __name__ == "__main__":
    main()