#!/bin/bash
# OSC Ascend Setup Script - Conda/Pip Environment
# Run this on OSC Ascend login node

set -e

echo "================================"
echo "EvoTune Setup for OSC Ascend"
echo "================================"

# Step 1: Load required modules
echo "Step 1: Finding and loading modules..."

# Find available Python module
echo "Available Python modules:"
module spider python 2>/dev/null | grep "python/" || echo "  Use 'module spider python' to see available versions"

# Find available CUDA module
echo ""
echo "Available CUDA modules:"
module spider cuda 2>/dev/null | grep "cuda/" || echo "  Use 'module spider cuda' to see available versions"

echo ""
echo "Attempting to load modules..."
# Try to load Python (adjust version if needed)
if module load python/3.10 2>/dev/null; then
    echo "✓ Loaded python/3.10"
elif module load python 2>/dev/null; then
    echo "✓ Loaded default python"
else
    echo "✗ Could not load Python module. Please load manually:"
    echo "  module avail python"
    echo "  module load python/<version>"
    exit 1
fi

# Try to load CUDA (try multiple versions)
CUDA_LOADED=false
for cuda_ver in cuda/12.1.1 cuda/12.1 cuda/12 cuda/11.8 cuda; do
    if module load $cuda_ver 2>/dev/null; then
        echo "✓ Loaded $cuda_ver"
        CUDA_LOADED=true
        break
    fi
done

if [ "$CUDA_LOADED" = false ]; then
    echo "✗ Could not load CUDA module. Please load manually:"
    echo "  module avail cuda"
    echo "  module load cuda/<version>"
    exit 1
fi

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
