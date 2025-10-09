import torch
import torchvision.models as models
import onnx
from pathlib import Path

class VisionModelHandler:
    """Handles loading and exporting vision models"""
    
    AVAILABLE_MODELS = {
        'mobilenet_v2': models.mobilenet_v2,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'efficientnet_b0': models.efficientnet_b0,
        'squeezenet1_0': models.squeezenet1_0,
    }
    
    def __init__(self, model_name='mobilenet_v2'):
        """Initialize with a pre-trained model"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.AVAILABLE_MODELS.keys())}")
        
        print(f"Loading {model_name}...")
        self.model_name = model_name
        self.model = self.AVAILABLE_MODELS[model_name](weights='DEFAULT')
        self.model.eval()
        print(f"✓ {model_name} loaded successfully")
    
    def export_to_onnx(self, output_path='model.onnx', input_shape=(1, 3, 224, 224)):
        """Export PyTorch model to ONNX format"""
        print(f"Exporting to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model exported and verified: {output_path}")
        
        return output_path
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)  # Assuming fp32
        }
    """Handles loading and exporting vision models"""
    
    AVAILABLE_MODELS = {
        'mobilenet_v2': models.mobilenet_v2,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'efficientnet_b0': models.efficientnet_b0,
        'squeezenet1_0': models.squeezenet1_0,
    }
    
    def __init__(self, model_name='mobilenet_v2'):
        """Initialize with a pre-trained model"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.AVAILABLE_MODELS.keys())}")
        
        print(f"Loading {model_name}...")
        self.model_name = model_name
        self.model = self.AVAILABLE_MODELS[model_name](weights='DEFAULT')
        self.model.eval()
        print(f"✓ {model_name} loaded successfully")
    
    def export_to_onnx(self, output_path='model.onnx', input_shape=(1, 3, 224, 224)):
        """Export PyTorch model to ONNX format"""
        print(f"Exporting to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model exported and verified: {output_path}")
        
        return output_path
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)  # Assuming fp33
        }