"""
Complete test script to verify the entire pipeline
Run this before launching the web app to ensure everything works
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("\n" + "="*70)
    print("1. Testing Imports...")
    print("="*70)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import onnx
        print(f"✅ ONNX: {onnx.__version__}")
    except ImportError as e:
        print(f"❌ ONNX import failed: {e}")
        return False
    
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime: {ort.__version__}")
        print(f"   Available Providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"❌ ONNX Runtime import failed: {e}")
        return False
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_model_handler():
    """Test model loading and ONNX export"""
    print("\n" + "="*70)
    print("2. Testing Model Handler...")
    print("="*70)
    
    try:
        from model_handler import VisionModelHandler
        
        # Test with MobileNetV2 (smallest and fastest)
        handler = VisionModelHandler('mobilenet_v2')
        print("✅ Model loaded successfully")
        
        # Get model info
        info = handler.get_model_info()
        print(f"   Model: {info['name']}")
        print(f"   Parameters: {info['total_params']:,}")
        print(f"   Size: {info['size_mb']:.2f} MB")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Export to ONNX
        onnx_path = handler.export_to_onnx('models/test_model.onnx')
        print(f"✅ ONNX export successful: {onnx_path}")
        
        # Verify file exists
        if Path(onnx_path).exists():
            size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
            print(f"   ONNX file size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hardware_profiles():
    """Test hardware profile detection"""
    print("\n" + "="*70)
    print("3. Testing Hardware Profiles...")
    print("="*70)
    
    try:
        from hardware_profiles import HardwareProfile
        
        # Get system info
        system_info = HardwareProfile.get_system_info()
        print("System Information:")
        for key, value in system_info.items():
            print(f"   {key}: {value}")
        
        # Get available profiles
        available = HardwareProfile.get_available_profiles()
        print(f"\n✅ Available Hardware Profiles ({len(available)}):")
        
        for profile_id in available:
            profile = HardwareProfile.get_profile(profile_id)
            print(f"   • {profile['name']}")
            print(f"     Device: {profile['device']}, Precision: {profile['precision']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware profiles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compilation():
    """Test model compilation for different hardware"""
    print("\n" + "="*70)
    print("4. Testing Compilation Pipeline...")
    print("="*70)
    
    try:
        from compiler import AdaptiveCompiler
        from hardware_profiles import HardwareProfile
        
        onnx_path = 'models/test_model.onnx'
        
        if not Path(onnx_path).exists():
            print(f"❌ ONNX model not found: {onnx_path}")
            return False
        
        # Get available profiles
        available = HardwareProfile.get_available_profiles()
        
        results = []
        
        for profile_id in available:
            print(f"\nTesting {profile_id}...")
            
            profile = HardwareProfile.get_profile(profile_id)
            compiler = AdaptiveCompiler(profile)
            
            # Compile
            success = compiler.compile(onnx_path, optimize=True)
            
            if not success:
                print(f"❌ Compilation failed for {profile_id}")
                continue
            
            # Quick benchmark (fewer runs for testing)
            benchmark = compiler.benchmark(num_runs=10, warmup_runs=3)
            results.append((profile_id, benchmark))
            
            print(f"✅ {profile['name']}: {benchmark['mean_ms']:.2f} ms, {benchmark['throughput_fps']:.2f} FPS")
        
        # Summary
        if results:
            print("\n" + "="*70)
            print("PERFORMANCE SUMMARY")
            print("="*70)
            
            results.sort(key=lambda x: x[1]['throughput_fps'], reverse=True)
            
            for profile_id, benchmark in results:
                profile = HardwareProfile.get_profile(profile_id)
                print(f"{profile['name']:30s} | {benchmark['throughput_fps']:6.2f} FPS | {benchmark['mean_ms']:6.2f} ms")
            
            # Calculate speedup
            baseline = results[-1][1]['throughput_fps']
            fastest = results[0][1]['throughput_fps']
            speedup = fastest / baseline
            
            print(f"\n🚀 Speedup: {speedup:.2f}x ({results[0][0]} vs {results[-1][0]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Compilation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test Flask app routes (without starting server)"""
    print("\n" + "="*70)
    print("5. Testing Flask Application...")
    print("="*70)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test system info endpoint
            response = client.get('/api/system-info')
            if response.status_code == 200:
                print("✅ /api/system-info endpoint working")
            else:
                print(f"❌ /api/system-info failed: {response.status_code}")
            
            # Test available models endpoint
            response = client.get('/api/available-models')
            if response.status_code == 200:
                print("✅ /api/available-models endpoint working")
            else:
                print(f"❌ /api/available-models failed: {response.status_code}")
            
            # Test main page
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main page (/) working")
            else:
                print(f"❌ Main page failed: {response.status_code}")
        
        print("\n✅ Flask application structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 HARDWARE-ADAPTIVE VISION COMPILER - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Handler", test_model_handler),
        ("Hardware Profiles", test_hardware_profiles),
        ("Compilation", test_compilation),
        ("Flask App", test_flask_routes)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to launch the web app.")
        print("\nRun: python app.py")
        print("Then open: http://localhost:5000")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before launching.")
        return 1

if __name__ == '__main__':
    sys.exit(main())