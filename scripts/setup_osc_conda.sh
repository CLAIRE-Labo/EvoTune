#!/bin/bash
# OSC Ascend Setup Script - Conda/Pip Environment
# Run this on OSC Ascend login node

set -e

echo "================================"
echo "EvoTune Setup for OSC Ascend"
echo "================================"

# Step 1: Load required modules
echo "Step 1: Loading modules..."
module load python/3.10
module load cuda/12.1.1  # Or whatever latest CUDA version is available

# Step 2: Create conda environment
echo "Step 2: Creating conda environment..."
if conda env list | grep -q "evotune"; then
    echo "Environment 'evotune' already exists. Activate with: conda activate evotune"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n evotune -y
    else
        echo "Skipping environment creation."
        exit 0
    fi
fi

conda create -n evotune python=3.10 -y

# Step 3: Activate environment
echo "Step 3: Activating environment..."
source activate evotune

# Step 4: Install PyTorch with CUDA support
echo "Step 4: Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install vLLM
echo "Step 5: Installing vLLM..."
pip install vllm

# Step 6: Install dependencies from requirements.txt
echo "Step 6: Installing project dependencies..."
pip install -r installation/docker-amd64-cuda-vllm/requirements.txt

# Step 7: Install the package itself
echo "Step 7: Installing EvoTune package..."
pip install -e .

# Step 8: Verify installation
echo "Step 8: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import packing; print('EvoTune package imported successfully')"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "To use the environment:"
echo "  module load python/3.10 cuda/12.1.1"
echo "  conda activate evotune"
echo ""
echo "To test the setup:"
echo "  sinteractive -A <your_project> -p quad -g 1 -t 1:00:00"
echo "  module load python/3.10 cuda/12.1.1"
echo "  conda activate evotune"
echo "  PYTHONPATH=src python -c 'import packing; print(\"Ready!\")'"
echo ""
