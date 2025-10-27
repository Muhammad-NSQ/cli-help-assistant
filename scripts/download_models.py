#!/usr/bin/env python3
"""
Model download script for CLI Help Assistant
"""

import sys
import subprocess
import shutil
from pathlib import Path

def check_git_lfs():
    """Check if git-lfs is installed"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_deepseek_model():
    """Download DeepSeek Coder model"""
    model_dir = Path("models/deepseek-coder-v2-lite")
    
    if model_dir.exists():
        print(f"Model already exists at {model_dir}")
        return True
    
    print("Downloading DeepSeek Coder model...")
    print("This may take a while (several GB)...")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    try:
        # Clone the model repository
        result = subprocess.run([
            'git', 'clone',
            'https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct',
            str(model_dir)
        ], cwd=Path("models").parent)
        
        if result.returncode == 0:
            print(f"Model downloaded successfully to {model_dir}")
            return True
        else:
            print("Failed to download model")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def download_alternative_model():
    """Download a smaller alternative model"""
    model_dir = Path("models/deepseek-coder-1.3b")
    
    if model_dir.exists():
        print(f"Alternative model already exists at {model_dir}")
        return True
    
    print("Downloading smaller DeepSeek model (1.3B)...")
    
    try:
        result = subprocess.run([
            'git', 'clone',
            'https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct',
            str(model_dir)
        ], cwd=Path("models").parent)
        
        if result.returncode == 0:
            print(f"Alternative model downloaded to {model_dir}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error downloading alternative model: {e}")
        return False

def main():
    """Main download function"""
    print("CLI Help Assistant - Model Download")
    print("=" * 40)
    
    # Check prerequisites
    if not shutil.which('git'):
        print("Error: git is not installed")
        return 1
    
    if not check_git_lfs():
        print("Warning: git-lfs not found. Large files may not download properly.")
        print("Install with: apt-get install git-lfs (Ubuntu) or brew install git-lfs (Mac)")
    
    # Ask user which model to download
    print("\nAvailable models:")
    print("1. DeepSeek Coder 6.7B (recommended, ~13GB)")
    print("2. DeepSeek Coder 1.3B (smaller, ~3GB)")
    print("3. Both models")
    
    try:
        choice = input("\nSelect model (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nDownload cancelled")
        return 0
    
    success = False
    
    if choice == "1":
        success = download_deepseek_model()
    elif choice == "2":
        success = download_alternative_model()
    elif choice == "3":
        success = download_deepseek_model() and download_alternative_model()
    else:
        print("Invalid choice")
        return 1
    
    if success:
        print("\nModel download completed!")
        print("You can now start the services with:")
        print("  python scripts/start_services.py")
        return 0
    else:
        print("\nModel download failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())