import torch

class HardwareProfile:
    """Hardware profile definitions"""
    
    PROFILES = {
        'nvidia_gpu': {
            'name': 'NVIDIA GTX 1650 GPU',
            'device': 'cuda',
            'precision': 'fp32',
            'memory': '4GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'High-performance GPU for inference',
            'color': '#76B900'  # NVIDIA green
        },
        'cpu_x86': {
            'name': 'Intel/AMD CPU',
            'device': 'cpu',
            'precision': 'fp32',
            'memory': '8GB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Standard CPU inference',
            'color': '#0071C5'  # Intel blue
        },
        'edge_int8': {
            'name': 'Edge Device (Quantized)',
            'device': 'cpu',
            'precision': 'int8',
            'memory': '512MB',
            'execution_provider': 'CPUExecutionProvider',
            'description': 'Quantized model for edge devices (simulated)',
            'color': '#FF6B35'  # Orange for edge
        },
        'jetson_nano': {
            'name': 'NVIDIA Jetson Nano (Simulated)',
            'device': 'cuda',
            'precision': 'fp16',
            'memory': '4GB',
            'execution_provider': 'CUDAExecutionProvider',
            'description': 'Edge GPU device simulation',
            'color': '#00A699'  # Teal
        }
    }
    
    @staticmethod
    def get_available_profiles():
        """Get list of hardware profiles available on current system"""
        available = []
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        for profile_id, profile in HardwareProfile.PROFILES.items():
            if profile['device'] == 'cuda' and not cuda_available:
                continue  # Skip GPU profiles if CUDA not available
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
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': torch.get_num_threads(),
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info