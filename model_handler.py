import torch
import torchvision.models as models
import onnx
from pathlib import Path

class VisionModelHandler:
    """Handles loading and exporting vision models
    
    Supports 64+ computer vision classification models organized by category:
    - Lightweight Models: MobileNet, SqueezeNet, ShuffleNet, MNASNet
    - ResNet Family: ResNet, Wide ResNet, ResNeXt
    - EfficientNet Family: EfficientNet B0-B7, EfficientNet V2
    - VGG Models: VGG11/13/16/19 with/without BatchNorm
    - DenseNet Models: DenseNet121/161/169/201
    - Vision Transformers: ViT, Swin Transformer
    - Modern Architectures: ConvNeXt, RegNet, MaxViT
    - Classic Models: AlexNet, GoogLeNet, Inception
    """
    
    AVAILABLE_MODELS = {
        # Lightweight Models (Mobile/Edge)
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        
        # ResNet Family
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnext101_32x8d': models.resnext101_32x8d,
        
        # EfficientNet Family
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'efficientnet_v2_s': models.efficientnet_v2_s,
        'efficientnet_v2_m': models.efficientnet_v2_m,
        'efficientnet_v2_l': models.efficientnet_v2_l,
        
        # VGG Models
        'vgg11': models.vgg11,
        'vgg11_bn': models.vgg11_bn,
        'vgg13': models.vgg13,
        'vgg13_bn': models.vgg13_bn,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'vgg19': models.vgg19,
        'vgg19_bn': models.vgg19_bn,
        
        # DenseNet Models
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        
        # Inception Models
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        
        # Vision Transformers
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'vit_l_32': models.vit_l_32,
        'vit_h_14': models.vit_h_14,
        
        # Other Modern Architectures
        'alexnet': models.alexnet,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
        'regnet_y_400mf': models.regnet_y_400mf,
        'regnet_y_800mf': models.regnet_y_800mf,
        'regnet_y_1_6gf': models.regnet_y_1_6gf,
        'regnet_y_3_2gf': models.regnet_y_3_2gf,
        'regnet_y_8gf': models.regnet_y_8gf,
        'regnet_y_16gf': models.regnet_y_16gf,
        'regnet_y_32gf': models.regnet_y_32gf,
        'mnasnet0_5': models.mnasnet0_5,
        'mnasnet1_0': models.mnasnet1_0,
        'swin_t': models.swin_t,
        'swin_s': models.swin_s,
        'swin_b': models.swin_b,
        'maxvit_t': models.maxvit_t,
    }
    
    # Model categories for organization
    MODEL_CATEGORIES = {
        'Lightweight (Edge/Mobile)': [
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'squeezenet1_0', 'squeezenet1_1', 'shufflenet_v2_x0_5', 
            'shufflenet_v2_x1_0', 'mnasnet0_5', 'mnasnet1_0'
        ],
        'ResNet Family': [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d'
        ],
        'EfficientNet Family': [
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
            'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
        ],
        'VGG Models': [
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
            'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
        ],
        'DenseNet Models': [
            'densenet121', 'densenet161', 'densenet169', 'densenet201'
        ],
        'Vision Transformers': [
            'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14',
            'swin_t', 'swin_s', 'swin_b'
        ],
        'Modern Architectures': [
            'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
            'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
            'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'maxvit_t'
        ],
        'Classic Models': [
            'alexnet', 'googlenet', 'inception_v3'
        ]
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
    
    @staticmethod
    def get_models_by_category():
        """Get models organized by category"""
        return VisionModelHandler.MODEL_CATEGORIES
    
    @staticmethod
    def get_total_models():
        """Get total number of available models"""
        return len(VisionModelHandler.AVAILABLE_MODELS)