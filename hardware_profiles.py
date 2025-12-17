import torch

class HardwareProfile:
    """Hardware profile definitions for heterogeneous computing platforms
    
    This class defines 15 different hardware profiles for model compilation covering:
    - High-performance GPUs (NVIDIA, AMD)
    - CPUs (x86, ARM)
    - Edge devices (Jetson, Coral TPU, ARM-based)
    - Specialized hardware (FPGA, Apple Silicon)
    - Various precision levels (FP32, FP16, INT8)
    
    The system automatically detects available hardware and enables compatible profiles.
    """
    
    PROFILES = {
        # High-Performance GPUs
        'nvidia_gpu': {
            'name': 'NVIDIA GTX 1650 GPU',
            'device': 'cuda',
            'precision': 'fp32',
            'memory': '4GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'High-performance GPU for inference',
            'category': 'GPU',
            'color': '#76B900'  # NVIDIA green
        },
        'nvidia_rtx': {
            'name': 'NVIDIA RTX GPU (Tensor Cores)',
            'device': 'cuda',
            'precision': 'fp16',
            'memory': '8GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'High-end GPU with Tensor Core acceleration',
            'category': 'GPU',
            'color': '#76B900'
        },
        'amd_gpu': {
            'name': 'AMD Radeon GPU (ROCm)',
            'device': 'rocm',
            'precision': 'fp32',
            'memory': '8GB',
            'execution_provider': 'ROCMExecutionProvider',
            'description': 'AMD GPU with ROCm support (requires ROCm installation)',
            'category': 'GPU',
            'color': '#ED1C24'  # AMD red
        },
        
        # CPUs (x86 and ARM)
        'cpu_x86': {
            'name': 'Intel/AMD CPU (x86-64)',
            'device': 'cpu',
            'precision': 'fp32',
            'memory': '8GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Standard x86 CPU inference with AVX optimizations',
            'category': 'CPU',
            'color': '#0071C5'  # Intel blue
        },
        'cpu_arm': {
            'name': 'ARM CPU (Cortex-A series)',
            'device': 'cpu',
            'precision': 'fp32',
            'memory': '4GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'ARM-based CPU for mobile and embedded systems',
            'category': 'CPU',
            'color': '#0091BD'  # ARM blue
        },
        'apple_silicon': {
            'name': 'Apple Silicon (M1/M2/M3)',
            'device': 'cpu',
            'precision': 'fp32',
            'memory': '8GB',
            'execution_provider': 'CoreMLExecutionProvider',
            'description': 'Apple Silicon with Neural Engine (requires macOS)',
            'category': 'CPU',
            'color': '#555555'  # Apple gray
        },
        
        # Edge Computing Devices
        'jetson_nano': {
            'name': 'NVIDIA Jetson Nano',
            'device': 'cuda',
            'precision': 'fp16',
            'memory': '4GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'Edge GPU device for AI at the edge',
            'category': 'Edge',
            'color': '#00A699'  # Teal
        },
        'jetson_xavier': {
            'name': 'NVIDIA Jetson Xavier NX',
            'device': 'cuda',
            'precision': 'fp16',
            'memory': '8GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'High-performance edge AI computing',
            'category': 'Edge',
            'color': '#00A699'
        },
        'coral_tpu': {
            'name': 'Google Coral Edge TPU',
            'device': 'tpu',
            'precision': 'int8',
            'memory': '1GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Ultra-low power edge TPU (simulated - requires Edge TPU runtime)',
            'category': 'Edge',
            'color': '#4285F4'  # Google blue
        },
        'edge_int8': {
            'name': 'Generic Edge Device (INT8)',
            'device': 'cpu',
            'precision': 'int8',
            'memory': '512MB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Quantized model for resource-constrained edge devices',
            'category': 'Edge',
            'color': '#FF6B35'  # Orange
        },
        'raspberry_pi': {
            'name': 'Raspberry Pi 4/5 (ARM)',
            'device': 'cpu',
            'precision': 'fp32',
            'memory': '2GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'ARM-based single-board computer',
            'category': 'Edge',
            'color': '#C51A4A'  # Raspberry red
        },
        
        # Specialized Hardware
        'fpga_xilinx': {
            'name': 'Xilinx FPGA (Simulated)',
            'device': 'fpga',
            'precision': 'int8',
            'memory': '2GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'FPGA-based inference (simulated - requires Vitis AI)',
            'category': 'FPGA',
            'color': '#E01F27'  # Xilinx red
        },
        'fpga_intel': {
            'name': 'Intel FPGA (Simulated)',
            'device': 'fpga',
            'precision': 'int8',
            'memory': '4GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Intel FPGA-based inference (simulated)',
            'category': 'FPGA',
            'color': '#0071C5'
        },
        
        # Cloud/Server
        'cloud_gpu': {
            'name': 'Cloud GPU Instance (A100/V100)',
            'device': 'cuda',
            'precision': 'fp16',
            'memory': '16GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'High-end cloud GPU for production inference',
            'category': 'GPU',
            'color': '#76B900'
        },
    }
    
    @staticmethod
    def get_available_profiles():
        """Get list of hardware profiles available on current system"""
        import platform
        available = []
        
        # Check CUDA availability for NVIDIA GPUs
        cuda_available = torch.cuda.is_available()
        
        # Detect system architecture
        system = platform.system().lower()
        machine = platform.machine().lower()
        is_arm = 'arm' in machine or 'aarch64' in machine
        is_apple_silicon = system == 'darwin' and is_arm
        
        for profile_id, profile in HardwareProfile.PROFILES.items():
            device = profile['device']
            
            # CUDA devices - require CUDA
            if device == 'cuda' and not cuda_available:
                continue
            
            # ROCm devices - typically not available in standard setups
            if device == 'rocm':
                try:
                    import onnxruntime as ort
                    if 'ROCMExecutionProvider' not in ort.get_available_providers():
                        continue
                except:
                    continue
            
            # Apple Silicon - only on macOS ARM
            if profile_id == 'apple_silicon' and not is_apple_silicon:
                continue
            
            # ARM-specific profiles
            if profile_id in ['cpu_arm', 'raspberry_pi'] and not is_arm:
                # Still include these as they can be simulated
                pass
            
            # TPU and FPGA are simulated, always available
            # All CPU profiles are always available
            
            available.append(profile_id)
        
        return available
    
    @staticmethod
    def get_profile(profile_id):
        """Get specific hardware profile"""
        if profile_id not in HardwareProfile.PROFILES:
            raise ValueError(f"Profile {profile_id} not found")
        return HardwareProfile.PROFILES[profile_id]
    
    @staticmethod
    def get_execution_providers(profile_id):
        """Get ONNX Runtime execution providers for profile"""
        profile = HardwareProfile.get_profile(profile_id)
        providers = []
        
        if profile['execution_provider'] == 'CUDAExecutionProvider':
            providers.append('CUDAExecutionProvider')
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    @staticmethod
    def get_system_info():
        """Get current system hardware information"""
        import platform
        import onnxruntime as ort
        
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': torch.get_num_threads(),
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'onnxruntime_providers': ort.get_available_providers(),
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    @staticmethod
    def get_profiles_by_category():
        """Get hardware profiles organized by category"""
        categorized = {}
        for profile_id, profile in HardwareProfile.PROFILES.items():
            category = profile.get('category', 'Other')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append({
                'id': profile_id,
                'name': profile['name'],
                'device': profile['device'],
                'precision': profile['precision'],
                'memory': profile['memory']
            })
        return categorized
    
    @staticmethod
    def get_total_profiles():
        """Get total number of hardware profiles"""
        return len(HardwareProfile.PROFILES)