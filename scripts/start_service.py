#!/usr/bin/env python3
"""
Development startup script for CLI Help Assistant services
"""

import sys
import time
import subprocess
import signal
import os
from pathlib import Path

def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            return True
        else:
            print("⚠ No NVIDIA GPU detected, using CPU")
            return False
    except FileNotFoundError:
        print("⚠ nvidia-smi not found, assuming CPU mode")
        return False

def check_model_exists():
    """Check if model files exist"""
    model_path = Path("models/deepseek-coder-v2-lite")
    if model_path.exists():
        print(f"✓ Model found at {model_path}")
        return True
    else:
        print(f"✗ Model not found at {model_path}")
        print("Please download the model first:")
        print("  mkdir -p models")
        print("  cd models")
        print("  git clone https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct deepseek-coder-v2-lite")
        return False

def start_model_server():
    """Start the model server"""
    print("\n🚀 Starting model server...")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': str(Path.cwd()),
        'MODEL_PATH': str(Path.cwd() / "models" / "deepseek-coder-v2-lite"),
        'HOST': '0.0.0.0',
        'PORT': '8000'
    })
    
    # Start model server process
    process = subprocess.Popen(
        [sys.executable, '-m', 'model_server.main'],
        env=env,
        cwd=Path.cwd()
    )
    
    print(f"Model server started with PID {process.pid}")
    print("Waiting for model to load...")
    
    # Wait for server to be ready
    max_wait = 120  # 2 minutes
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get('model_loaded'):
                    print("✓ Model server is ready!")
                    break
        except:
            pass
        
        time.sleep(5)
        wait_time += 5
        print(f"  Waiting... ({wait_time}s)")
    
    if wait_time >= max_wait:
        print("✗ Model server failed to start within timeout")
        process.terminate()
        return None
    
    return process

def test_cli():
    """Test CLI functionality"""
    print("\n🧪 Testing CLI...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'cli_help.main', 'ask', 'how do I list files?'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ CLI test successful!")
            print("Response preview:", result.stdout[:100] + "..." if len(result.stdout) > 100 else result.stdout)
        else:
            print("✗ CLI test failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ CLI test timed out")
    except Exception as e:
        print(f"✗ CLI test error: {e}")

def main():
    """Main startup function"""
    print("🔧 CLI Help Assistant - Development Startup")
    print("=" * 50)
    
    # Pre-flight checks
    if not check_model_exists():
        return 1
    
    check_gpu()
    
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    
    model_server_process = None
    
    try:
        # Start model server
        model_server_process = start_model_server()
        if not model_server_process:
            return 1
        
        # Test CLI
        test_cli()
        
        print("\n✅ Services started successfully!")
        print("\nUsage:")
        print("  python -m cli_help.main ask 'how do I use git?'")
        print("  python -m cli_help.main explain 'find'")
        print("\nModel server API:")
        print("  Health: http://localhost:8000/health")
        print("  Info: http://localhost:8000/model/info")
        print("\nPress Ctrl+C to stop all services")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        if model_server_process:
            model_server_process.terminate()
            model_server_process.wait()
        print("✓ Services stopped")
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if model_server_process:
            model_server_process.terminate()
        return 1

if __name__ == "__main__":
    sys.exit(main())