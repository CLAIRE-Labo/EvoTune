# EvoTune Setup Guide for OSC Ascend Cluster

This guide will help you set up and run EvoTune experiments on the Ohio Supercomputer Center's Ascend cluster.

## System Information

**OSC Ascend Cluster:**
- **Architecture:** x86_64 (AMD EPYC CPUs: Milan 7643 & 7H12)
- **GPUs:** NVIDIA A100 (80GB and 40GB variants)
- **Partitions:**
  - `nextgen`: 2 A100 GPUs, 120 cores, 472GB memory per node
  - `quad`: 4 A100 GPUs, 88 cores, 921GB memory per node
  - `batch`: 4 A100 GPUs, 88 cores, 921GB memory per node
- **Scheduler:** SLURM

**Recommended Setup:** `installation/docker-amd64-cuda-vllm/` (vLLM inference engine)

---

## Step-by-Step Setup

### 1. Initial Setup on OSC

```bash
# SSH into OSC Ascend
ssh <username>@ascend.osc.edu

# Navigate to your project directory
cd $PFSDIR  # Or your preferred project location

# Clone the repository if not already done
git clone <repository_url> EvoTune
cd EvoTune
```

### 2. Environment Setup

You have two options:

#### Option A: Using Apptainer/Singularity (Recommended for OSC)

OSC supports Apptainer (formerly Singularity) for containerization:

```bash
# Build the Docker image locally or pull from a registry
cd installation/docker-amd64-cuda-vllm/

# Convert Docker image to Apptainer/Singularity
# First build the Docker image, then:
docker save evotune:latest | apptainer build evotune.sif docker-archive:/dev/stdin
```

#### Option B: Using Conda Environment

```bash
# Load Python module
module load python/3.10

# Create conda environment
conda create -n evotune python=3.10
conda activate evotune

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configure for OSC

Update the cluster configuration:

**File:** `configs/cluster/osc_ascend.yaml` (already created)

```yaml
scratch_path: "$PFSDIR"  # Or your specific path
use_tgi: 0
use_vllm: 1
```

### 4. Download/Setup Models

You'll need access to the LLM models. Common options:

```bash
# Create a models directory
mkdir -p $PFSDIR/models

# For LLaMA models, you'll need HuggingFace access
# Set your HuggingFace token
export HF_TOKEN=<your_token>

# Models will be downloaded automatically on first run
# Or pre-download:
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"
```

### 5. Test Setup with a Quick Run

#### Interactive Test (for debugging)

```bash
# Request interactive GPU node
sinteractive -A <project> -p quad -g 1 -t 2:00:00

# Load modules
module load cuda/12.1.1 python/3.10

# Set environment
export PYTHONPATH=src:$PYTHONPATH

# Run a quick test (you may want to reduce num_rounds for testing)
python src/experiments/main.py \
    task=bin \
    model=llama32 \
    train=dpo \
    cluster=osc_ascend \
    seed=0 \
    prefix=test_run \
    gpu_nums=0 \
    num_rounds=10 \
    wandb=0
```

### 6. Submit Production Job

```bash
# Edit the job script to add your project allocation
nano scripts/osc_ascend_train_single.sh
# Replace <YOUR_OSC_PROJECT> with your actual project code

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/osc_ascend_train_single.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/evotune_<job_id>.out
```

### 7. Running Experiments

#### Single Bin Packing Experiment (Simplest)

```bash
sbatch scripts/osc_ascend_train_single.sh
```

#### Customize Parameters

Edit the script or override via command line:

```bash
python src/experiments/main.py \
    task=bin \
    model=llama32 \
    train=dpo \
    cluster=osc_ascend \
    seed=42 \
    prefix=my_experiment \
    num_rounds=1000 \
    num_workers=12 \
    gpu_nums=0
```

#### Multiple Seeds / Sweep

For running multiple experiments with different seeds:

```bash
# Submit multiple jobs with different seeds
for seed in 0 1 2 3 4; do
    sbatch --job-name=evotune_seed${seed} \
           --export=ALL,SEED=$seed \
           scripts/osc_ascend_train_single.sh
done
```

### 8. Monitoring and Results

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/evotune_<job_id>.out

# Check results directory
ls -lh out/logs/osc_bin_test/
```

**Results will be saved to:**
- `out/logs/{prefix}/{task}_{model}_{train}_{seed}/`
- Program database: `programbank_*.jsonl`
- Statistics: `statistics.jsonl`
- Best functions: `best_programs.txt`

### 9. Evaluation

After training, evaluate the discovered algorithms:

```bash
python src/experiments/eval.py \
    task=bin \
    logs_dir=out/logs/osc_bin_test/bin_llama32_dpo_0 \
    testset=testset
```

---

## Tips for OSC Ascend

1. **Resource Allocation:**
   - For bin packing experiments, 1 GPU is sufficient to start
   - Use `quad` or `batch` partitions for multi-GPU experiments
   - Request appropriate time: 24-48 hours for full runs

2. **Storage:**
   - Use `$PFSDIR` for project files and outputs
   - Use `$TMPDIR` for temporary files during job execution
   - Clean up intermediate files to save space

3. **GPU Memory:**
   - A100 80GB is ideal for larger models
   - Start with smaller models (llama32) to test setup

4. **Checkpointing:**
   - EvoTune saves program databases periodically
   - You can resume from saved databases if jobs time out

5. **Common Issues:**
   - **CUDA out of memory:** Reduce `num_outputs_per_prompt` or `num_workers`
   - **Timeout:** Increase `--time` in SLURM script
   - **Module not found:** Ensure `PYTHONPATH=src` is set

---

## Next Steps

1. Start with a short test run (10-100 rounds) to verify setup
2. Run full bin packing experiment (2701 rounds as in paper)
3. Experiment with different tasks (tsp, flatpack)
4. Try different models and hyperparameters

## Questions or Issues?

- Check OSC documentation: https://www.osc.edu/resources/technical_support/supercomputers/ascend
- EvoTune paper: https://arxiv.org/abs/2504.05108
- Report issues in the GitHub repository
