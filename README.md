# Hardware-Adaptive Vision Model Compiler

This project is a comprehensive web-based tool for compiling and benchmarking deep learning models across **heterogeneous hardware platforms**. It supports **15 different hardware profiles** and **95+ computer vision models**, allowing you to optimize and benchmark models for CPUs, GPUs, Edge devices, and FPGAs.

## Supported Hardware

This project supports **15 different hardware profiles** covering a wide range of computing platforms:

### High-Performance GPUs (5 profiles)
1. **NVIDIA GTX 1650 GPU** - Desktop GPU (FP32, 4GB)
2. **NVIDIA RTX GPU** - Tensor Core acceleration (FP16, 8GB)
3. **AMD Radeon GPU (ROCm)** - AMD GPU computing (FP32, 8GB)
4. **Cloud GPU (A100/V100)** - Production cloud inference (FP16, 16GB)

### CPUs (3 profiles)
5. **Intel/AMD CPU (x86-64)** - Standard server/desktop CPU with AVX (FP32, 8GB)
6. **ARM CPU (Cortex-A)** - ARM-based mobile/embedded systems (FP32, 4GB)
7. **Apple Silicon (M1/M2/M3)** - Apple Neural Engine (FP32, 8GB)

### Edge Computing Devices (5 profiles)
8. **NVIDIA Jetson Nano** - Edge AI GPU (FP16, 4GB)
9. **NVIDIA Jetson Xavier NX** - High-performance edge AI (FP16, 8GB)
10. **Google Coral Edge TPU** - Ultra-low power TPU (INT8, 1GB)
11. **Generic Edge Device** - Quantized edge inference (INT8, 512MB)
12. **Raspberry Pi 4/5** - ARM single-board computer (FP32, 2GB)

### Specialized Hardware (2 profiles)
13. **Xilinx FPGA** - FPGA-based inference (INT8, 2GB)
14. **Intel FPGA** - FPGA computing (INT8, 4GB)

**Note:** The system automatically detects available hardware and enables compatible profiles. GPU profiles require CUDA/ROCm. Some profiles (FPGA, TPU, Apple Silicon) may require specific runtime environments.

## Supported Models

This project includes **95+ pre-trained computer vision models** organized into 8 categories:

- **Lightweight Models (9)**: MobileNet v2/v3, SqueezeNet, ShuffleNet, MNASNet - optimized for mobile and edge devices
- **ResNet Family (9)**: ResNet18/34/50/101/152, Wide ResNet, ResNeXt - classic CNN architectures
- **EfficientNet Family (11)**: EfficientNet B0-B7, EfficientNet V2 - efficient scaling for accuracy/speed
- **VGG Models (8)**: VGG11/13/16/19 (with/without batch norm) - deep convolutional networks
- **DenseNet Models (4)**: DenseNet121/161/169/201 - densely connected networks
- **Vision Transformers (8)**: ViT, Swin Transformer - transformer-based vision models
- **Modern Architectures (13)**: ConvNeXt, RegNet, MaxViT - state-of-the-art models
- **Classic Models (3)**: AlexNet, GoogLeNet, Inception v3 - foundational architectures

## Features

- **Heterogeneous Hardware Support**: Compile and benchmark models on 15 different hardware platforms (GPUs, CPUs, Edge devices, FPGAs)
- **Extensive Model Library**: Access to 95+ pre-trained computer vision models from torchvision
- **Web-Based UI**: A Flask server provides an easy-to-use interface for all operations
- **ONNX Export**: Export PyTorch models to the standard ONNX format for cross-platform compatibility
- **Hardware-Adaptive Compilation**: Automatically optimize models for different hardware using ONNX Runtime
- **Performance Benchmarking**: Measure latency, throughput, and percentile latencies for each hardware target
- **Automatic Hardware Detection**: Detects available hardware (CUDA GPUs, CPU architecture, available execution providers)
- **Model Categories**: Browse models organized by architecture family and use case

## Model Categories and Count

| Category | Count | Examples | Best For |
|----------|-------|----------|----------|
| Lightweight | 9 | MobileNet v2/v3, SqueezeNet, ShuffleNet | Mobile, Edge, IoT devices |
| ResNet Family | 9 | ResNet18/50/152, Wide ResNet, ResNeXt | General-purpose, Transfer learning |
| EfficientNet | 11 | EfficientNet B0-B7, EfficientNet V2 | Balanced accuracy/efficiency |
| VGG Models | 8 | VGG11/13/16/19 + BatchNorm variants | Feature extraction, Classic CNNs |
| DenseNet | 4 | DenseNet121/161/169/201 | High accuracy, Feature reuse |
| Vision Transformers | 8 | ViT, Swin Transformer | State-of-the-art accuracy |
| Modern Architectures | 13 | ConvNeXt, RegNet, MaxViT | Latest research, Production |
| Classic Models | 3 | AlexNet, GoogLeNet, Inception | Educational, Baseline |

**Total: 95+ models** - All models are pre-trained on ImageNet and ready for inference or fine-tuning.

## Project Structure

- `app.py`: The main Flask application that serves the web interface and handles API requests
- `model_handler.py`: Manages 95+ vision models with category organization and ONNX export capabilities
- `hardware_profiles.py`: Defines 15 heterogeneous hardware profiles with automatic detection and categorization
- `compiler.py`: Core compilation engine that optimizes models for specific hardware using ONNX Runtime
- `requirements.txt`: Python dependencies for the project
- `test_demo.py`: Comprehensive test suite for validating all components

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd hardware-adaptable-compiler

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the test suite (optional but recommended)
python test_demo.py

# Start the web application
python app.py

# Open browser to http://localhost:5000
```

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd hardware-adaptable-compiler
    ```

2.  **Create a Python Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**:
    -   **Windows**:
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    -   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note*: If you have a CUDA-enabled GPU, you may need to install a specific build of PyTorch. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.

## How to Run

1.  **Start the Flask Server**:
    Run the `app.py` script from your terminal:
    ```bash
    python app.py
    ```

2.  **Open the Web Interface**:
    Open your web browser and navigate to:
    [http://localhost:5000](http://localhost:5000)

3.  **Using the Application**:
    -   The web page will show your current system hardware.
    -   Select a model from the dropdown and click "Load Model". This will download the model and convert it to ONNX.
    -   Once the model is loaded, you can choose a hardware profile and click "Compile" to benchmark it.
    -   Alternatively, click "Compile for All Available Profiles" to run benchmarks on all compatible hardware on your system.
    -   The results of each benchmark will be displayed on the page.

## Hardware Compatibility Matrix

### Total Hardware Profiles: 15

| Category | Profiles | Devices | Precision Support | Use Cases |
|----------|----------|---------|-------------------|-----------|
| **GPU** | 4 | NVIDIA (CUDA), AMD (ROCm), Cloud | FP32, FP16 | High-performance inference, training |
| **CPU** | 3 | x86-64, ARM, Apple Silicon | FP32 | General-purpose, cross-platform |
| **Edge** | 5 | Jetson, Coral TPU, Raspberry Pi | FP32, FP16, INT8 | IoT, mobile, embedded systems |
| **FPGA** | 2 | Xilinx, Intel | INT8 | Custom acceleration, low latency |

### Automatic Hardware Detection

The application intelligently detects your system's hardware capabilities:
- **CUDA Detection**: Automatically enables NVIDIA GPU profiles when CUDA is available
- **Platform Detection**: Identifies CPU architecture (x86, ARM, Apple Silicon)
- **Provider Detection**: Checks available ONNX Runtime execution providers (CUDA, ROCm, CoreML)
- **Fallback Support**: Always provides CPU-based fallback options
- **Simulated Profiles**: Includes simulated profiles for FPGA and specialized edge hardware

### Hardware-Model Recommendations

| Hardware Type | Recommended Models | Reason |
|---------------|-------------------|---------|
| **Edge Devices** | MobileNet, SqueezeNet, EfficientNet B0-B2 | Optimized for low memory and compute |
| **Desktop GPUs** | ResNet50, EfficientNet B4-B5, ViT | Balance of accuracy and performance |
| **Cloud GPUs** | ResNet152, EfficientNet B7, Swin Transformers | Maximum accuracy, high compute |
| **FPGAs** | SqueezeNet, MobileNet (INT8 quantized) | Fixed-point optimized architectures |
| **ARM CPUs** | MobileNet v3, ShuffleNet, MNASNet | ARM-optimized mobile architectures |

### Adding Custom Hardware Profiles

To add support for additional hardware platforms, edit the `PROFILES` dictionary in `hardware_profiles.py`. Each profile requires:
- `name`: Display name for the hardware
- `device`: Device type ('cuda', 'cpu', 'rocm', 'tpu', 'fpga')
- `precision`: Data precision (fp32, fp16, int8, etc.)
- `memory`: Available memory
- `execution_provider`: ONNX Runtime provider name
- `description`: Brief description of the use case
- `category`: Hardware category (GPU, CPU, Edge, FPGA)
- `color`: Display color for UI visualization (hex code)
