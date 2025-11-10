# OSC Ascend Quick Start Guide

**For Ohio Supercomputer Center's Ascend Cluster**

## TL;DR - Fast Setup (Recommended)

On OSC Ascend, **use Conda/Pip** instead of Docker (Docker daemon not available to regular users).

### 0. Check Available Modules (Do This First!)

```bash
# On OSC Ascend login node
cd /path/to/EvoTune
bash scripts/check_osc_modules.sh
```

This will show you what Python and CUDA versions are available on OSC.

### 1. One-Command Setup

```bash
bash scripts/setup_osc_conda.sh
```

This script will:
- Automatically detect and load available Python/CUDA modules
- Create conda environment named `evotune`
- Install PyTorch with CUDA support
- Install vLLM
- Install all dependencies
- Install EvoTune package

The script tries multiple CUDA versions (12.8.1, 12.6.2, 12.4.1, 11.8.0) based on what's available on OSC Ascend.

**Time:** ~10-15 minutes

### 2. Manual Setup (if script fails)

```bash
# Load modules (check available versions first with 'module avail python' and 'module avail cuda')
module load python/3.10   # or whatever version is available
module load cuda/12.8.1   # or 12.6.2, 12.4.1, 11.8.0 - use what's available on your system

# Create environment
conda create -n evotune python=3.10 -y
conda activate evotune

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install dependencies
pip install -r installation/docker-amd64-cuda-vllm/requirements.txt

# Install package
pip install -e .

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Quick Test

```bash
# Request interactive GPU node
sinteractive -A <your_project> -p quad -g 1 -t 1:00:00

# Load modules and activate environment
module load python/3.10   # or your available version
module load cuda/12.8.1   # or 12.6.2, 12.4.1, 11.8.0
conda activate evotune

# Set Python path
export PYTHONPATH=src:$PYTHONPATH

# Quick test run (10 rounds)
python src/experiments/main.py \
    task=bin \
    model=llama32 \
    train=dpo \
    cluster=osc_ascend \
    seed=0 \
    prefix=quick_test \
    gpu_nums=0 \
    num_rounds=10 \
    wandb=0
```

### 4. Submit Production Job

```bash
# Edit SLURM script to add your project code
nano scripts/osc_ascend_train_single.sh
# Change: #SBATCH --account=<YOUR_OSC_PROJECT>

# Create logs directory
mkdir -p logs

# Submit
sbatch scripts/osc_ascend_train_single.sh

# Monitor
squeue -u $USER
tail -f logs/evotune_*.out
```

---

## Alternative: Apptainer/Singularity (Advanced)

If you prefer containers, you have two options:

### Option A: Build Docker locally, push to registry, pull on OSC

```bash
# On your local machine (with Docker):
cd installation/docker-amd64-cuda-vllm/
./template.sh env
./template.sh build_generic

# Tag and push to registry (Docker Hub, GitHub Container Registry, etc.)
docker tag <image_name>:latest <your_registry>/<image_name>:latest
docker push <your_registry>/<image_name>:latest

# On OSC:
apptainer pull docker://<your_registry>/<image_name>:latest
```

### Option B: Build directly with Apptainer from Dockerfile

```bash
# On OSC login node
cd installation/docker-amd64-cuda-vllm/
apptainer build evotune.sif Dockerfile
```

**Note:** This requires Apptainer to support all Dockerfile instructions, which may not work for complex Dockerfiles.

---

## Troubleshooting

### Common Issues

**1. "The following module(s) are unknown: cuda/12.1.1" or similar**

The CUDA/Python version specified doesn't exist on your OSC system.

**Solution:**
```bash
# Check what's actually available
module avail python
module avail cuda

# Or use the helper script
bash scripts/check_osc_modules.sh

# Then load the available version, for example:
module load cuda/11.8.0  # OSC Ascend uses full version numbers
```

The updated setup script now automatically tries multiple versions!

**2. "module: command not found"**
```bash
# Add to your ~/.bashrc on OSC:
source /etc/profile.d/lmod.sh
```

**3. "conda: command not found"**
```bash
# Load Anaconda module first:
module load python/3.10
# Or use the system conda:
source /usr/local/anaconda3/etc/profile.d/conda.sh
```

**4. "CUDA out of memory"**
- Reduce `num_workers` in config
- Reduce `num_outputs_per_prompt`
- Request more GPU memory (use A100 80GB nodes)

**5. "ModuleNotFoundError: No module named 'packing'"**
```bash
export PYTHONPATH=src:$PYTHONPATH
```

**6. vLLM installation fails**
```bash
# Try with specific CUDA version:
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Resources

- **Full Setup Guide:** See `OSC_SETUP_GUIDE.md`
- **OSC Ascend Docs:** https://www.osc.edu/resources/technical_support/supercomputers/ascend
- **Paper:** https://arxiv.org/abs/2504.05108
- **Issues:** Report at GitHub repository

---

## Next Steps After Setup

1. **Verify CUDA access:**
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   # Should show: NVIDIA A100...
   ```

2. **Check vLLM:**
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

3. **Run short test** (10 rounds, ~5-10 minutes)

4. **Submit full experiment** (2701 rounds, ~24-48 hours)

5. **Analyze results:**
   ```bash
   python src/experiments/eval.py \
       task=bin \
       logs_dir=out/logs/<your_run_dir>
   ```
