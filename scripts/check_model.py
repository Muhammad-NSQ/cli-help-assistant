#!/usr/bin/env python3
"""
Check model files for corruption or issues
"""

import sys
from pathlib import Path

def check_model_directory(model_path):
    """Check model files in directory"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model directory does not exist: {model_path}")
        return False
    
    print(f"📁 Checking model directory: {model_path}")
    
    # Check required files
    required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    model_files = []
    
    print("\n📋 Required files:")
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file} ({size} bytes)")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Check model weight files
    print("\n⚖️  Model weight files:")
    weight_files = [
        'pytorch_model.bin',
        'model.safetensors', 
        'pytorch_model-00001-of-00001.bin',
        'model-00001-of-00001.safetensors'
    ]
    
    found_weights = False
    for file in weight_files:
        file_path = model_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.1f} MB)")
            
            # Check if file seems corrupted (too small)
            if size < 1024 * 1024:  # Less than 1MB
                print(f"    ⚠️  File seems too small for a real model")
            elif size < 100:  # Less than 100 bytes
                print(f"    ❌ File is likely corrupted or empty")
                
            found_weights = True
        else:
            print(f"  ⬜ {file} (not found)")
    
    if not found_weights:
        print("  ❌ No model weight files found!")
        return False
    
    # List all files in directory
    print(f"\n📂 All files in {model_path}:")
    for item in sorted(model_path.iterdir()):
        if item.is_file():
            size = item.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    return True

def main():
    """Main function"""
    print("🔍 Model File Checker")
    print("=" * 40)
    
    # Check common model locations
    model_locations = [
        "models/deepseek-coder-v2-lite",
        "models/deepseek-coder-1.3b", 
        "model_server/models/deepseek-coder-1.3b",
        "model_server/models/deepseek-coder-v2-lite"
    ]
    
    for location in model_locations:
        path = Path(location)
        if path.exists():
            print(f"\n🎯 Found model directory: {location}")
            check_model_directory(path)
        else:
            print(f"\n⬜ {location} (not found)")
    
    # Test loading with transformers
    print(f"\n🧪 Testing transformers loading...")
    
    for location in model_locations:
        path = Path(location)
        if path.exists():
            try:
                print(f"\nTesting {path}...")
                from transformers import AutoTokenizer, AutoConfig
                
                # Test config
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                print(f"  ✅ Config loaded: {config.model_type}")
                
                # Test tokenizer
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                print(f"  ✅ Tokenizer loaded: {len(tokenizer)} tokens")
                
                # Try to test model loading (just check if files are readable)
                try:
                    from transformers import AutoModelForCausalLM
                    print("  🔄 Testing model loading (this may take a moment)...")
                    
                    # Try with safetensors disabled first
                    model = AutoModelForCausalLM.from_pretrained(
                        path,
                        trust_remote_code=True,
                        torch_dtype="auto",
                        device_map="cpu",
                        use_safetensors=False
                    )
                    print("  ✅ Model loaded successfully with PyTorch format")
                    del model
                    
                except Exception as e:
                    print(f"  ❌ Model loading failed: {e}")
                    
                    # Try with safetensors
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            path,
                            trust_remote_code=True,
                            torch_dtype="auto", 
                            device_map="cpu",
                            use_safetensors=True
                        )
                        print("  ✅ Model loaded successfully with safetensors format")
                        del model
                    except Exception as e2:
                        print(f"  ❌ Safetensors loading also failed: {e2}")
                
            except Exception as e:
                print(f"  ❌ Basic loading failed: {e}")

if __name__ == "__main__":
    main()

