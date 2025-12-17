# Quick Reference Guide

## Hardware Profiles (15 Total)

### GPU Profiles (4)
| Profile ID | Name | Precision | Memory | Use Case |
|------------|------|-----------|--------|----------|
| `nvidia_gpu` | NVIDIA GTX 1650 GPU | FP32 | 4GB | Desktop GPU inference |
| `nvidia_rtx` | NVIDIA RTX GPU | FP16 | 8GB | Tensor Core acceleration |
| `amd_gpu` | AMD Radeon GPU (ROCm) | FP32 | 8GB | AMD GPU computing |
| `cloud_gpu` | Cloud GPU (A100/V100) | FP16 | 16GB | Production cloud inference |

### CPU Profiles (3)
| Profile ID | Name | Precision | Memory | Use Case |
|------------|------|-----------|--------|----------|
| `cpu_x86` | Intel/AMD CPU (x86-64) | FP32 | 8GB | Standard server/desktop |
| `cpu_arm` | ARM CPU (Cortex-A) | FP32 | 4GB | Mobile/embedded systems |
| `apple_silicon` | Apple Silicon (M1/M2/M3) | FP32 | 8GB | macOS with Neural Engine |

### Edge Profiles (5)
| Profile ID | Name | Precision | Memory | Use Case |
|------------|------|-----------|--------|----------|
| `jetson_nano` | NVIDIA Jetson Nano | FP16 | 4GB | Edge AI GPU |
| `jetson_xavier` | NVIDIA Jetson Xavier NX | FP16 | 8GB | High-performance edge |
| `coral_tpu` | Google Coral Edge TPU | INT8 | 1GB | Ultra-low power TPU |
| `edge_int8` | Generic Edge Device | INT8 | 512MB | Quantized edge inference |
| `raspberry_pi` | Raspberry Pi 4/5 | FP32 | 2GB | ARM single-board computer |

### FPGA Profiles (2)
| Profile ID | Name | Precision | Memory | Use Case |
|------------|------|-----------|--------|----------|
| `fpga_xilinx` | Xilinx FPGA | INT8 | 2GB | FPGA-based inference |
| `fpga_intel` | Intel FPGA | INT8 | 4GB | Intel FPGA computing |

## Model Categories (64 Total)

### 1. Lightweight Models (9 models)
**Best for:** Mobile, Edge, IoT devices  
**Models:** `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `squeezenet1_0`, `squeezenet1_1`, `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `mnasnet0_5`, `mnasnet1_0`

### 2. ResNet Family (9 models)
**Best for:** General-purpose, Transfer learning  
**Models:** `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `wide_resnet50_2`, `wide_resnet101_2`, `resnext50_32x4d`, `resnext101_32x8d`

### 3. EfficientNet Family (11 models)
**Best for:** Balanced accuracy/efficiency  
**Models:** `efficientnet_b0` through `efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`

### 4. VGG Models (8 models)
**Best for:** Feature extraction, Classic CNNs  
**Models:** `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

### 5. DenseNet Models (4 models)
**Best for:** High accuracy, Feature reuse  
**Models:** `densenet121`, `densenet161`, `densenet169`, `densenet201`

### 6. Vision Transformers (8 models)
**Best for:** State-of-the-art accuracy  
**Models:** `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`, `swin_t`, `swin_s`, `swin_b`

### 7. Modern Architectures (13 models)
**Best for:** Latest research, Production  
**Models:** `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `regnet_y_400mf` through `regnet_y_32gf`, `maxvit_t`

### 8. Classic Models (3 models)
**Best for:** Educational, Baseline  
**Models:** `alexnet`, `googlenet`, `inception_v3`

## Hardware-Model Recommendations

### For Edge Devices (< 2GB RAM)
✅ **Recommended:**
- MobileNet v2/v3 (lightweight, fast)
- SqueezeNet (very small model size)
- ShuffleNet (efficient architecture)
- EfficientNet B0 (good accuracy-size tradeoff)

❌ **Not Recommended:**
- Large ViT models (too memory intensive)
- VGG models (large parameter count)
- ResNet152 (high memory usage)

### For Desktop GPUs (4-8GB)
✅ **Recommended:**
- ResNet50/101 (excellent performance)
- EfficientNet B4/B5 (high accuracy)
- DenseNet (good feature extraction)
- ConvNeXt (modern architecture)

### For Cloud GPUs (16GB+)
✅ **Recommended:**
- ResNet152 (maximum accuracy)
- EfficientNet B7 (state-of-the-art)
- Large Vision Transformers (ViT-L, ViT-H)
- Swin Transformers (best performance)

### For FPGAs
✅ **Recommended (INT8 quantized):**
- SqueezeNet (FPGA-friendly architecture)
- MobileNet (optimized for fixed-point)
- Small EfficientNet (B0-B2)

## Common Use Cases

### Real-time Inference (30+ FPS)
**Hardware:** NVIDIA RTX, Cloud GPU  
**Models:** MobileNet v3, EfficientNet B0-B2, ResNet18

### High Accuracy (Top-1 > 80%)
**Hardware:** Cloud GPU, NVIDIA RTX  
**Models:** EfficientNet B7, ViT-L/H, Swin-B, ResNet152

### Low Power Edge (< 5W)
**Hardware:** Coral TPU, Generic Edge (INT8)  
**Models:** MobileNet v2, SqueezeNet, ShuffleNet

### Batch Processing
**Hardware:** Cloud GPU, AMD GPU  
**Models:** Any model with hardware match

### IoT Devices (< 1GB RAM)
**Hardware:** Raspberry Pi, Generic Edge  
**Models:** MobileNet v3 Small, SqueezeNet, MNASNet 0.5

## API Endpoints

### Get System Information
```bash
GET /api/system-info
```
Returns: System hardware, available profiles, categories, total count

### Get Available Models
```bash
GET /api/available-models
```
Returns: Model list, categories, total count

### Load a Model
```bash
POST /api/load-model
Body: {"model_name": "mobilenet_v2"}
```

### Compile for Hardware
```bash
POST /api/compile
Body: {"profile_id": "nvidia_gpu"}
```

### Compile for All Available Hardware
```bash
POST /api/compile-all
```

### Get Results
```bash
GET /api/results
```

## Python API Usage

```python
# Load a model
from model_handler import VisionModelHandler
handler = VisionModelHandler('resnet50')
handler.export_to_onnx('model.onnx')

# Get hardware profiles
from hardware_profiles import HardwareProfile
profiles = HardwareProfile.get_available_profiles()
print(f"Available: {profiles}")

# Compile for hardware
from compiler import AdaptiveCompiler
profile = HardwareProfile.get_profile('nvidia_gpu')
compiler = AdaptiveCompiler(profile)
compiler.compile('model.onnx')
results = compiler.benchmark()
```

## Performance Tips

1. **Match Precision to Hardware:**
   - Use FP16 on Tensor Core GPUs (RTX series)
   - Use INT8 on edge devices and FPGAs
   - Use FP32 for maximum accuracy

2. **Model Selection:**
   - Smaller models → Lower latency
   - Efficient architectures (EfficientNet, MobileNet) → Better throughput
   - Transformers → Higher accuracy but slower

3. **Hardware Selection:**
   - GPU → Best for batch processing
   - CPU → Good for small batch/single inference
   - Edge → Optimized for real-time, low power

4. **Optimization:**
   - Enable graph optimizations in ONNX Runtime
   - Use appropriate execution providers
   - Warm up the model before benchmarking

## Troubleshooting

### CUDA Not Available
- Install CUDA toolkit
- Install PyTorch with CUDA support
- Verify with `torch.cuda.is_available()`

### ROCm Not Available
- Install ROCm framework
- Install ONNX Runtime with ROCm support
- May not be available on all systems

### Apple Silicon Not Detected
- Only works on macOS with ARM processors
- Requires CoreML execution provider

### Model Loading Slow
- Models are downloaded from internet on first use
- Subsequent loads use cached weights
- Check internet connection

### Out of Memory
- Reduce batch size
- Use smaller model variant
- Use quantized models (INT8)
- Choose hardware with more memory
