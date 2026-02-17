# SLURM Distributed Training for ViT on CIFAR-10

Complete SLURM-compatible distributed training workflow for Vision Transformer (vit_small_patch16_224) on CIFAR-10 with MLflow experiment tracking.

## Overview

- **Model**: vit_small_patch16_224 (22M parameters)
- **Dataset**: CIFAR-10 (auto-downloaded via torchvision)
- **Training**: 200 epochs, 2 A100 GPUs, distributed training
- **Image Size**: 32x32 → 224x224 (upscaled)
- **Expected Accuracy**: ~92-95% on test set

## Files Created

1. **custon_training.sh** - Main SLURM batch script
   - Configures SLURM job parameters
   - Launches distributed training
   - Runs validation on test set
   - Logs everything to MLflow

2. **mlflow_logger.py** - MLflow logging daemon
   - Monitors training metrics in real-time
   - Logs to MLflow experiment tracking
   - Logs hyperparameters and artifacts

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies (handled automatically by script)

# Verify timm is installed
python -c "import timm; print(timm.__version__)"
```

### 2. Configuration

Edit `custon_training.sh` to customize parameters (lines 20-68):

```bash
# Model Configuration
MODEL="vit_small_patch16_224"
EPOCHS=200
BATCH_SIZE=128
LEARNING_RATE=0.001

# Paths
DATA_DIR="./data"              # Where CIFAR-10 will be downloaded
OUTPUT_DIR="./output/cifar10_vit"
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
```

### 3. Submit Job

```bash
# Submit to SLURM
sbatch custon_training.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f train_<jobid>.out
```

## Monitoring

### During Training

```bash
# Watch training progress
tail -f output/cifar10_vit/<experiment_name>/training.log

# Check MLflow logger
tail -f output/cifar10_vit/<experiment_name>/mlflow_logger.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### After Training

```bash
# View MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns
# Then open: http://localhost:5000

# Check test results
cat output/cifar10_vit/<experiment_name>/test_results.csv

# View training summary
tail output/cifar10_vit/<experiment_name>/summary.csv
```

## Output Structure

```
output/cifar10_vit/<experiment_name>/
├── args.yaml                   # Training configuration
├── summary.csv                 # Per-epoch metrics
├── training.log                # Full training output
├── validation.log              # Test set evaluation log
├── mlflow_logger.log          # MLflow daemon log
├── test_results.csv           # Final test metrics
├── model_best.pth.tar         # Best checkpoint (by val accuracy)
├── checkpoint-*.pth.tar       # Epoch checkpoints
└── last.pth.tar               # Latest checkpoint

mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── metrics/           # Logged metrics
        ├── params/            # Hyperparameters
        └── artifacts/         # Model checkpoints & logs
```

## Key Configuration Parameters

### Model & Training

```bash
MODEL="vit_small_patch16_224"  # Options: vit_tiny_patch16_224, vit_base_patch16_224
EPOCHS=200                      # Training epochs
BATCH_SIZE=128                  # Per-GPU batch size (effective: 256)
LEARNING_RATE=0.001            # Initial learning rate
WEIGHT_DECAY=0.05              # AdamW weight decay
LR_SCHEDULER="cosine"          # Options: cosine, step, plateau
```

### Data Augmentation

```bash
AUTO_AUGMENT="rand-m9-mstd0.5" # RandAugment policy
MIXUP=0.8                      # Mixup alpha
CUTMIX=1.0                     # CutMix alpha
REPROB=0.25                    # Random erasing probability
SMOOTHING=0.1                  # Label smoothing
```

### SLURM Resources

```bash
#SBATCH --gpus=2                # Number of GPUs
#SBATCH --mem=128G              # Memory per node
#SBATCH -t 21-00:00:00         # Time limit
#SBATCH --partition=gpu         # GPU partition  
#SBATCH --constraint=a100       # GPU type
```

## Advanced Usage

### Resume Training

```bash
# Edit custon_training.sh, add after line 175:
#    --resume "$OUTPUT_DIR/$EXPERIMENT_NAME/last.pth.tar" \
```

### Use Different Model

```bash
# In custon_training.sh, change:
MODEL="vit_tiny_patch16_224"    # Faster, ~5.7M params
# or
MODEL="vit_base_patch16_224"    # Better accuracy, ~86M params
```

### Modify Environment Setup

```bash
# In custon_training.sh, uncomment/edit lines 101-106:
module load cuda/12.0
module load python/3.10
conda activate timm-env
```

### Custom Dataset Location

```bash
# In custon_training.sh, change:
DATA_DIR="/path/to/your/data"
DATASET_DOWNLOAD=""  # Remove if data already downloaded
```

## Distributed Training Details

### How it Works

1. **SLURM Setup**: Uses environment variables (`SLURM_NTASKS`, `SLURM_PROCID`, etc.)
2. **Process Spawning**: `srun --ntasks=2 --gpus-per-task=1` launches 2 processes
3. **timm Detection**: train.py auto-detects SLURM and initializes distributed training
4. **Communication**: Uses NCCL backend for GPU communication

### Scaling to More GPUs

```bash
# Edit custon_training.sh:
#SBATCH --gpus=4               # Use 4 GPUs

# Then edit srun command (line 148):
srun --ntasks=4 --gpus-per-task=1 python train.py ...
```

## Troubleshooting

### Job Fails Immediately

```bash
# Check SLURM error log
cat train_<jobid>.err

# Check available resources
sinfo -p gpu
```

### Out of Memory

```bash
# Reduce batch size in custon_training.sh:
BATCH_SIZE=64  # or 32
```

### MLflow Not Working

```bash
# Install MLflow
pip install mlflow

# Or disable MLflow logging:
# Comment out lines 128-141 in custon_training.sh
```

### CUDA Out of Memory

```bash
# Options:
BATCH_SIZE=64                    # Reduce batch size
AMP="--amp"                      # Enable mixed precision (already on)
CHANNELS_LAST="--channels-last"  # Use channels-last format (already on)
```

### Dataset Download Fails

```bash
# Pre-download CIFAR-10:
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Then in custon_training.sh:
DATASET_DOWNLOAD=""  # Remove --dataset-download flag
```

## Performance Expectations

### Training Time (2x A100)

- ~1-2 hours for 200 epochs
- ~20-30 seconds per epoch
- ~390 batches per epoch (50k samples / 128 batch size)

### Memory Usage

- ~12-16GB per GPU (with batch_size=128)
- Mixed precision (AMP) saves ~30-40% memory

### Expected Results

- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~92-95%
- Convergence usually by epoch 150-175

## Differences: srun vs sbatch

### `sbatch` - Job Submission

- **Purpose**: Submit a batch script to SLURM queue
- **Usage**: `sbatch custon_training.sh`
- **Function**: Adds your job to the queue, allocates resources, runs the script

### `srun` - Process Launching

- **Purpose**: Launch parallel tasks on allocated resources
- **Usage**: Inside batch scripts to run actual programs
- **Function**: Spawns processes across nodes/GPUs with proper environment setup

### In This Workflow

1. You run `sbatch custon_training.sh` → Submits job to queue
2. SLURM allocates 2 GPUs when available
3. Script runs, executing `srun python train.py` → Launches 2 training processes
4. Each process gets 1 GPU, timm detects distributed setup automatically

## Citation

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

## License

This training script follows timm's Apache 2.0 license.
