## Summary

This PR adds conda/pip-based environment setup for OSC Ascend cluster, eliminating the need for Docker which is not available to regular users on HPC clusters.

### Changes

- **`scripts/setup_osc_conda.sh`** - Automated conda environment setup script
  - Loads Python 3.10 and CUDA 12.1 modules
  - Creates `evotune` conda environment
  - Installs PyTorch with CUDA support
  - Installs vLLM inference engine
  - Installs all project dependencies

- **`OSC_QUICKSTART.md`** - Simplified quick start guide
  - One-command setup instructions
  - Manual setup fallback
  - Quick test examples
  - Troubleshooting tips

- **`scripts/osc_ascend_train_single.sh`** - Updated SLURM script
  - Default to conda environment activation
  - Clearer module loading
  - Documented Apptainer alternative

### Why This Change?

Docker daemon is not available to regular users on HPC clusters like OSC Ascend. The standard approach on HPC is to use conda/pip environments or container runtimes like Apptainer/Singularity. This PR provides the recommended and simplest setup path.

### Testing

Tested on OSC Ascend cluster:
- Environment setup completes successfully
- CUDA detection works correctly
- vLLM imports successfully
- EvoTune package installs and imports correctly

### Related

Addresses setup issues encountered when trying to use Docker on OSC login nodes.

## Checklist

- [x] Code follows project style
- [x] Documentation updated
- [x] Tested on target platform (OSC Ascend)
- [x] No breaking changes
