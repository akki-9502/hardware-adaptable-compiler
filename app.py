from flask import Flask, render_template, request, jsonify
from model_handler import VisionModelHandler
from hardware_profiles import HardwareProfile
from compiler import AdaptiveCompiler
import os
import json
from datetime import datetime

app = Flask(__name__)

# Global storage for results
compilation_results = []
current_model = None
current_onnx_path = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/system-info', methods=['GET'])
def get_system_info():
    """Get system hardware information"""
    try:
        system_info = HardwareProfile.get_system_info()
        available_profiles = HardwareProfile.get_available_profiles()
        
        profiles_info = {}
        for profile_id in available_profiles:
            profiles_info[profile_id] = HardwareProfile.get_profile(profile_id)
        
        return jsonify({
            'status': 'success',
            'system': system_info,
            'available_profiles': profiles_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    try:
        models = list(VisionModelHandler.AVAILABLE_MODELS.keys())
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load a vision model"""
    global current_model, current_onnx_path
    
    try:
        data = request.json
        model_name = data.get('model_name', 'mobilenet_v2')
        
        print(f"\nLoading model: {model_name}")
        
        # Load model
        handler = VisionModelHandler(model_name)
        model_info = handler.get_model_info()
        
        # Export to ONNX
        onnx_path = f"models/{model_name}.onnx"
        os.makedirs('models', exist_ok=True)
        
        handler.export_to_onnx(onnx_path)
        
        current_model = model_name
        current_onnx_path = onnx_path
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'model_info': model_info,
            'onnx_path': onnx_path
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compile', methods=['POST'])
def compile_model():
    """Compile model for target hardware"""
    global current_onnx_path, compilation_results
    
    try:
        if current_onnx_path is None:
            return jsonify({
                'status': 'error',
                'message': 'No model loaded. Please load a model first.'
            }), 400
        
        data = request.json
        profile_id = data.get('profile_id', 'cpu_x86')
        
        print(f"\nCompiling for profile: {profile_id}")
        
        # Get hardware profile
        profile = HardwareProfile.get_profile(profile_id)
        
        # Create compiler
        compiler = AdaptiveCompiler(profile)
        
        # Compile
        success = compiler.compile(current_onnx_path, optimize=True)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Compilation failed'
            }), 500
        
        # Benchmark
        benchmark_results = compiler.benchmark(num_runs=30, warmup_runs=5)
        
        # Store results
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': current_model,
            'profile_id': profile_id,
            'profile_name': profile['name'],
            'results': benchmark_results
        }
        compilation_results.append(result_entry)
        
        return jsonify({
            'status': 'success',
            'profile': profile,
            'benchmark': benchmark_results,
            'model_info': compiler.get_model_info()
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/compile-all', methods=['POST'])
def compile_all_profiles():
    """Compile model for all available hardware profiles"""
    global current_onnx_path, compilation_results
    
    try:
        if current_onnx_path is None:
            return jsonify({
                'status': 'error',
                'message': 'No model loaded. Please load a model first.'
            }), 400
        
        # Get all available profiles
        available_profiles = HardwareProfile.get_available_profiles()
        
        results = []
        
        for profile_id in available_profiles:
            print(f"\n{'='*70}")
            print(f"Compiling for: {profile_id}")
            
            profile = HardwareProfile.get_profile(profile_id)
            compiler = AdaptiveCompiler(profile)
            
            success = compiler.compile(current_onnx_path, optimize=True)
            
            if success:
                benchmark_results = compiler.benchmark(num_runs=30, warmup_runs=5)
                
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'model': current_model,
                    'profile_id': profile_id,
                    'profile_name': profile['name'],
                    'results': benchmark_results
                }
                
                results.append(result_entry)
                compilation_results.append(result_entry)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_profiles': len(results)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get all compilation results"""
    return jsonify({
        'status': 'success',
        'results': compilation_results,
        'count': len(compilation_results)
    })

@app.route('/api/clear-results', methods=['POST'])
def clear_results():
    """Clear all results"""
    global compilation_results
    compilation_results = []
    return jsonify({'status': 'success', 'message': 'Results cleared'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Hardware-Adaptive Vision Model Compiler")
    print("="*70)
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)