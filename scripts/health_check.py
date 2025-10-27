#!/usr/bin/env python3
"""
Health check script for CLI Help Assistant services
"""

import sys
import requests
import time
from pathlib import Path

def check_model_server():
    """Check model server health"""
    try:
        print("Checking model server...")
        response = requests.get('http://localhost:8000/health', timeout=10)
        
        if response.status_code == 200:
            health = response.json()
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Model loaded: {health.get('model_loaded', False)}")
            print(f"  GPU available: {health.get('gpu_available', False)}")
            
            if health.get('model_loaded'):
                # Get model info
                info_response = requests.get('http://localhost:8000/model/info', timeout=5)
                if info_response.status_code == 200:
                    info = info_response.json()
                    print(f"  Model: {info.get('model_name', 'unknown')}")
                    print(f"  Device: {info.get('device', 'unknown')}")
            
            return health.get('status') == 'healthy' and health.get('model_loaded')
        else:
            print(f"  HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  Connection failed - server not running")
        return False
    except requests.exceptions.Timeout:
        print("  Request timed out")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_generation():
    """Test text generation"""
    try:
        print("\nTesting text generation...")
        payload = {
            "prompt": "How do I list files in Linux?",
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:8000/generate',
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            tokens = result.get('total_tokens', 0)
            duration = end_time - start_time
            
            print(f"  Response time: {duration:.2f}s")
            print(f"  Tokens: {tokens}")
            print(f"  Response preview: {response_text[:100]}...")
            return True
        else:
            print(f"  Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  Generation error: {e}")
        return False

def check_cli_integration():
    """Test CLI integration"""
    try:
        print("\nTesting CLI integration...")
        import subprocess
        
        result = subprocess.run([
            sys.executable, '-m', 'cli_help.main', 
            'ask', 'test question'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  CLI test successful")
            return True
        else:
            print(f"  CLI test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  CLI test timed out")
        return False
    except Exception as e:
        print(f"  CLI test error: {e}")
        return False

def main():
    """Main health check function"""
    print("CLI Help Assistant - Health Check")
    print("=" * 40)
    
    all_healthy = True
    
    # Check model server
    if not check_model_server():
        all_healthy = False
    
    # Test generation if server is healthy
    if all_healthy:
        if not test_generation():
            all_healthy = False
    
    # Test CLI integration
    if all_healthy:
        if not check_cli_integration():
            all_healthy = False
    
    print("\n" + "=" * 40)
    if all_healthy:
        print("✓ All services healthy")
        return 0
    else:
        print("✗ Some services unhealthy")
        return 1

if __name__ == "__main__":
    sys.exit(main())