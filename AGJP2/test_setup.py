"""Test script to verify DeepSeek ODE Tutor setup."""
import sys
import pathlib
import importlib.util

def test_imports():
    """Test that all required packages can be imported."""
    print("🔧 Testing imports...")
    
    required_packages = [
        'torch', 'transformers', 'accelerate', 'bitsandbytes',
        'safetensors', 'huggingface_hub', 'fastapi', 'uvicorn',
        'pydantic', 'rich', 'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        return False
    else:
        print("✅ All packages available")
        return True

def test_cuda():
    """Test CUDA availability."""
    print("\n🔧 Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ⚠️  CUDA not available - will use CPU (slower)")
            return True
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False

def test_config():
    """Test configuration file."""
    print("\n🔧 Testing configuration...")
    config_path = pathlib.Path("config/settings.yaml")
    
    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model', 'generation', 'prompts']
        for key in required_keys:
            if key not in config:
                print(f"  ❌ Missing config section: {key}")
                return False
        
        print("  ✅ Configuration file valid")
        return True
    except Exception as e:
        print(f"  ❌ Error reading config: {e}")
        return False

def test_model_path():
    """Test model directory."""
    print("\n🔧 Testing model directory...")
    model_path = pathlib.Path("models/deepseek_r1_1_5b")
    
    if not model_path.exists():
        print(f"  ❌ Model directory not found: {model_path}")
        print("  💡 Run download_weights.py first")
        return False
    
    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"  ❌ Model config not found: {config_file}")
        print("  💡 Model download may be incomplete")
        return False
    
    print("  ✅ Model directory exists and looks valid")
    return True

def test_chatbot_import():
    """Test chatbot module import."""
    print("\n🔧 Testing chatbot import...")
    try:
        sys.path.insert(0, str(pathlib.Path.cwd()))
        from src.chatbot import DeepSeekODETutor
        print("  ✅ Chatbot module imports successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error importing chatbot: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 DeepSeek ODE Tutor Setup Test\n")
    
    tests = [
        test_imports,
        test_cuda,
        test_config,
        test_model_path,
        test_chatbot_import
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Your setup looks good.")
        print("\n🚀 You can now run:")
        print("  scripts\\run_cli.ps1   # Interactive CLI")
        print("  scripts\\run_api.ps1   # REST API server")
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 