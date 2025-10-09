# Hardware-Adaptive Vision Model Compiler

This project is a web-based tool for compiling and benchmarking deep learning models for various hardware targets. It allows you to load pre-trained vision models, export them to ONNX, and then compile and benchmark them on different hardware profiles like CPUs and NVIDIA GPUs.

## Features

- **Web-Based UI**: A Flask server provides an easy-to-use interface for all operations.
- **Model Loading**: Load popular computer vision models from `torchvision` (e.g., MobileNetV2, ResNet, EfficientNet).
- **ONNX Export**: Export PyTorch models to the standard ONNX format.
- **Hardware-Adaptive Compilation**: Compile models for different hardware profiles using ONNX Runtime.
- **Performance Benchmarking**: Measure key performance metrics like latency, throughput, and percentile latencies for each hardware target.
- **System Information**: Automatically detects available hardware (CPU, CUDA-enabled GPU).

## Project Structure

- `app.py`: The main Flask application that serves the web interface and handles API requests.
- `model_handler.py`: A class responsible for loading `torchvision` models and exporting them to ONNX.
- `hardware_profiles.py`: Defines the different hardware targets (e.g., CPU, NVIDIA GPU) and provides system information.
- `compiler.py`: The core class that takes an ONNX model and a hardware profile, then compiles and benchmarks it using ONNX Runtime.
- `requirements.txt`: A list of all Python dependencies for the project.
- `templates/index.html`: (Assumed) The HTML file for the web interface.

## Setup and Installation

1.  **Clone the Repository** (or ensure you have the project files in a directory).

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
