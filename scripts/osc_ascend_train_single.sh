#!/bin/bash
#SBATCH --job-name=evotune_bin
#SBATCH --account=<YOUR_OSC_PROJECT>    # TODO: Replace with your OSC project allocation
#SBATCH --time=48:00:00                 # 48 hours (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # Matches num_workers in config
#SBATCH --gpus-per-node=1               # Request 1 GPU for single run
#SBATCH --partition=quad                # Options: nextgen (2 GPUs), quad (4 GPUs), batch (4 GPUs)
#SBATCH --mem=100G                      # Memory request
#SBATCH --output=logs/evotune_%j.out
#SBATCH --error=logs/evotune_%j.err

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Load necessary modules
# Note: Adjust versions based on what's available on your OSC system
# Use 'module avail python' and 'module avail cuda' to see options
module load python/3.10 || module load python
module load cuda/12.1.1 || module load cuda/12.1 || module load cuda/12 || module load cuda/11.8 || module load cuda

# Activate conda environment (RECOMMENDED for OSC)
conda activate evotune

# Alternative: If using Apptainer/Singularity container instead
# apptainer run --nv <your_docker_image.sif> ...

# Set environment variables
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs out/logs

# Run the training
python src/experiments/main.py \
    task=bin \
    model=llama32 \
    train=dpo \
    cluster=osc_ascend \
    seed=0 \
    prefix=osc_bin_test \
    gpu_nums=0 \
    wandb=0 \
    run_or_dev=run

echo "Job finished on $(date)"
