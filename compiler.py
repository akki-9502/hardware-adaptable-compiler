import onnxruntime as ort
import numpy as np
import time
import torch
from pathlib import Path

class AdaptiveCompiler:
    """Compiles and optimizes models for different hardware targets"""
    
    def __init__(self, hardware_profile):
        """Initialize compiler with hardware profile"""
        self.profile = hardware_profile
        self.session = None
        self.compiled_model_path = None
    
    def compile(self, onnx_model_path, optimize=True):
        """Compile ONNX model for target hardware"""
        print(f"\n{'='*60}")
        print(f"Compiling for: {self.profile['name']}")
        print(f"Device: {self.profile['device']}")
        print(f"Precision: {self.profile['precision']}")
        print(f"{'='*60}")
        
        # Set session options
        sess_options = ort.SessionOptions()
        
        if optimize:
            # Enable optimizations
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Get execution providers based on hardware profile
        providers = self._get_execution_providers()
        
        print(f"Execution Providers: {providers}")
        
        try:
            # Create inference session
            self.session = ort.InferenceSession(
                onnx_model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            print(f"✓ Model compiled successfully!")
            print(f"✓ Active provider: {self.session.get_providers()[0]}")
            
            self.compiled_model_path = onnx_model_path
            
            return True
            
        except Exception as e:
            print(f"✗ Compilation failed: {str(e)}")
            return False
    
    def _get_execution_providers(self):
        """Get execution providers based on hardware profile"""
        providers = []
        
        # Add hardware-specific provider
        if self.profile['device'] == 'cuda':
            providers.append('CUDAExecutionProvider')
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def benchmark(self, num_runs=50, warmup_runs=10, input_shape=(1, 3, 224, 224)):
        """Benchmark model performance"""
        if self.session is None:
            raise ValueError("Model not compiled. Call compile() first.")
        
        print(f"\nBenchmarking ({num_runs} runs)...")
        
        # Create random input
        input_name = self.session.get_inputs()[0].name
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.session.run(None, {input_name: input_data})
        
        # Actual benchmark
        latencies = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(None, {input_name: input_data})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000 / np.mean(latencies),
            'hardware': self.profile['name'],
            'device': self.profile['device'],
            'precision': self.profile['precision']
        }
        
        self._print_benchmark_results(results)
        
        return results
    
    def _print_benchmark_results(self, results):
        """Pretty print benchmark results"""
        print(f"\n{'='*60}")
        print(f"Benchmark Results - {results['hardware']}")
        print(f"{'='*60}")
        print(f"Mean Latency:    {results['mean_ms']:.2f} ms")
        print(f"Std Dev:         {results['std_ms']:.2f} ms")
        print(f"Min Latency:     {results['min_ms']:.2f} ms")
        print(f"Max Latency:     {results['max_ms']:.2f} ms")
        print(f"Median Latency:  {results['median_ms']:.2f} ms")
        print(f"95th Percentile: {results['p95_ms']:.2f} ms")
        print(f"99th Percentile: {results['p99_ms']:.2f} ms")
        print(f"Throughput:      {results['throughput_fps']:.2f} FPS")
        print(f"{'='*60}\n")
    
    def infer(self, input_data):
        """Run inference on input data"""
        if self.session is None:
            raise ValueError("Model not compiled. Call compile() first.")
        
        input_name = self.session.get_inputs()[0].name
        
        # Ensure input is numpy array
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        # Run inference
        output = self.session.run(None, {input_name: input_data})
        
        return output[0]
    
    def get_model_info(self):
        """Get compiled model information"""
        if self.session is None:
            return None
        
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        return {
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'output_type': output_info.type,
            'providers': self.session.get_providers()
        }