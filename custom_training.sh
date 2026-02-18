#!/usr/bin/env bash
################################################################################
# SLURM Distributed Training Script for Vision Transformer on CIFAR-10
# 
# This script trains a vit_small_patch16_224 model on CIFAR-10 using 
# distributed training across 2 A100 GPUs with MLflow experiment tracking.
#
# Usage: sbatch custom_training.sh
################################################################################

#SBATCH -J vit_cifar10_train         # Job name
#SBATCH -o vit_cifar10_basic/train_%j.out              # Output file (%j = job ID)
#SBATCH -e vit_cifar10_basic/train_%j.err              # Error file
#SBATCH -t 21-00:00:00               # Time limit (21 days)
#SBATCH --constraint=a100            # Require A100 GPUs
#SBATCH --partition=gpu              # GPU partition
#SBATCH --nodes=1                    # Single node
#SBATCH --gres=gpu:2                 # Request 2 GPUs  
#SBATCH --mem=128G                   # Memory requirement
#SBATCH --cpus-per-task=8            # CPUs for data loading

################################################################################
# CONFIGURATION - All parameters hardcoded here for easy modification
################################################################################

# Model Configuration
MODEL="vit_small_patch16_224"
NUM_CLASSES=10
IMG_SIZE=224

# Dataset Configuration
DATASET="torch/cifar10"
DATA_DIR="./data"
DATASET_DOWNLOAD="--dataset-download"  # Auto-download CIFAR-10

# Training Hyperparameters
EPOCHS=200
BATCH_SIZE=128                    # Per GPU batch size
OPTIMIZER="adamw"
LEARNING_RATE=0.001
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=10
COOLDOWN_EPOCHS=10
LR_SCHEDULER="cosine"

# Augmentation Parameters
AUTO_AUGMENT="rand-m9-mstd0.5"
MIXUP=0.8
CUTMIX=1.0
REPROB=0.25                       # Random erasing probability
SMOOTHING=0.1                     # Label smoothing

# Performance & Parallelization
NUM_WORKERS=8
AMP="--amp"                       # Automatic mixed precision
CHANNELS_LAST="--channels-last"   # Use channels-last memory format
PIN_MEMORY="--pin-mem"
SYNC_BN="--sync-bn"               # Synchronized batch normalization

# Output Configuration
OUTPUT_DIR="./output/cifar10_vit"
EXPERIMENT_NAME="vit_small_cifar10_$(date +%Y%m%d_%H%M%S)"
LOG_INTERVAL=50

# MLflow Configuration
MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
MLFLOW_EXPERIMENT="cifar10_vit_training"

# Validation Configuration
VAL_BATCH_SIZE=256
TEST_SPLIT="test"

################################################################################
# ENVIRONMENT SETUP
################################################################################

echo "=========================================="
echo "Starting SLURM Job: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="
echo ""

# Print configuration
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE per GPU"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""

# Load required modules (adjust based on your cluster)
# module load cuda/12.0
# module load python/3.10

# Activate virtual environment if needed
# source ~/venv/bin/activate
# Or conda environment:
# conda activate timm-env
# Alternative: pip install -r requirements-mlflow.txt

# Install required dependencies on GPU node
echo "Installing dependencies..."
# pip install --quiet mlflow pyyaml || echo "Warning: Could not install some dependencies"
pip install -r requirements-mlflow.txt || echo "Warning: Could not install some dependencies"
echo ""

# Verify GPU availability
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Set environment variables for better performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

################################################################################
# START MLFLOW LOGGER
################################################################################

echo " Starting MLflow logger..."

# Start MLflow logger in background
stdbuf -oL -eL python -u mlflow_logger.py \
    --summary-file "$OUTPUT_DIR/$EXPERIMENT_NAME/summary.csv" \
    --run-name "$EXPERIMENT_NAME" \
    --tracking-uri "$MLFLOW_TRACKING_URI" \
    --experiment-name "$MLFLOW_EXPERIMENT" \
    --poll-interval 10.0 \
    > "$OUTPUT_DIR/$EXPERIMENT_NAME/mlflow_logger.log" 2>&1 &

LOGGER_PID=$!
echo "MLflow logger started (PID: $LOGGER_PID)"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo ""

# Trap to cleanup logger on exit
trap "echo 'Stopping MLflow logger...'; kill $LOGGER_PID 2>/dev/null" EXIT INT TERM

################################################################################
# DISTRIBUTED TRAINING
################################################################################

echo "=========================================="
echo "Starting Distributed Training"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# Run distributed training with torchrun (simpler and more reliable than srun)
# torchrun handles GPU assignment automatically
stdbuf -oL -eL torchrun --nproc_per_node=2 train.py \
    --data-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    $DATASET_DOWNLOAD \
    --model "$MODEL" \
    --num-classes $NUM_CLASSES \
    --img-size $IMG_SIZE \
    --epochs $EPOCHS \
    -b $BATCH_SIZE \
    --opt "$OPTIMIZER" \
    --lr $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --sched "$LR_SCHEDULER" \
    --warmup-epochs $WARMUP_EPOCHS \
    --cooldown-epochs $COOLDOWN_EPOCHS \
    --aa "$AUTO_AUGMENT" \
    --mixup $MIXUP \
    --cutmix $CUTMIX \
    --reprob $REPROB \
    --smoothing $SMOOTHING \
    $AMP \
    $CHANNELS_LAST \
    $PIN_MEMORY \
    $SYNC_BN \
    --workers $NUM_WORKERS \
    --output "$OUTPUT_DIR" \
    --experiment "$EXPERIMENT_NAME" \
    --log-interval $LOG_INTERVAL \
    --checkpoint-hist 5 \
    2>&1 | stdbuf -oL -eL tee "$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Training duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo ""

# Check if training was successful
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check the error log: $OUTPUT_DIR/$EXPERIMENT_NAME/training.log"
    exit $TRAIN_EXIT_CODE
fi

################################################################################
# VALIDATION ON TEST SET
################################################################################

echo "=========================================="
echo "Running Validation on Test Set"
echo "=========================================="
echo ""

# Find the best checkpoint
BEST_CHECKPOINT="$OUTPUT_DIR/$EXPERIMENT_NAME/model_best.pth.tar"

if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "WARNING: Best checkpoint not found at $BEST_CHECKPOINT"
    echo "Looking for alternative checkpoint..."
    
    # Try to find latest checkpoint
    LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR/$EXPERIMENT_NAME"/checkpoint-*.pth.tar 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        BEST_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "Using checkpoint: $BEST_CHECKPOINT"
    else
        echo "ERROR: No checkpoint found"
        exit 1
    fi
fi

# Run validation on test set (single GPU)
stdbuf -oL -eL python -u validate.py \
    --data-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --split "$TEST_SPLIT" \
    --model "$MODEL" \
    --num-classes $NUM_CLASSES \
    --img-size $IMG_SIZE \
    --checkpoint "$BEST_CHECKPOINT" \
    -b $VAL_BATCH_SIZE \
    --results-file "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results.csv" \
    --workers $NUM_WORKERS \
    2>&1 | stdbuf -oL -eL tee "$OUTPUT_DIR/$EXPERIMENT_NAME/validation.log"

VAL_EXIT_CODE=$?
echo ""
echo "Validation completed with exit code: $VAL_EXIT_CODE"
echo ""

################################################################################
# LOG FINAL RESULTS TO MLFLOW
################################################################################

echo "=========================================="
echo "Finalizing MLflow Logging"
echo "=========================================="
echo ""

# Stop the background MLflow logger
if kill -0 $LOGGER_PID 2>/dev/null; then
    kill $LOGGER_PID
    wait $LOGGER_PID 2>/dev/null
fi

# Log test set metrics to MLflow
if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results.csv" ]; then
    echo "Logging test set metrics to MLflow..."
    
    # Parse test results and log to MLflow
    python3 - <<PYEOF
import csv
import os
try:
    import mlflow
    
    # Read test results
    results_file = "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results.csv"
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if results:
        row = results[0]
        test_metrics = {}
        for key, value in row.items():
            try:
                test_metrics[f"test_{key}"] = float(value)
            except (ValueError, TypeError):
                pass
        
        # Log to MLflow (start with same run name to append metrics)
        mlflow.set_tracking_uri("$MLFLOW_TRACKING_URI")
        mlflow.set_experiment("$MLFLOW_EXPERIMENT")
        
        # Find the run by name and log metrics
        experiment = mlflow.get_experiment_by_name("$MLFLOW_EXPERIMENT")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '$EXPERIMENT_NAME'"
            )
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics(test_metrics)
                    # Log artifacts
                    mlflow.log_artifact("$BEST_CHECKPOINT", artifact_path="checkpoints")
                    mlflow.log_artifact("$OUTPUT_DIR/$EXPERIMENT_NAME/training.log", artifact_path="logs")
                    mlflow.log_artifact("$OUTPUT_DIR/$EXPERIMENT_NAME/validation.log", artifact_path="logs")
                    mlflow.log_artifact("$OUTPUT_DIR/$EXPERIMENT_NAME/args.yaml", artifact_path="config")
                print(f"Logged test metrics to MLflow: {test_metrics}")
except ImportError:
    print("MLflow not available, skipping test metrics logging")
except Exception as e:
    print(f"Warning: Could not log test metrics to MLflow: {e}")
PYEOF
fi

################################################################################
# FINAL SUMMARY
################################################################################

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Training Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo ""
echo "Output Location: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "  - Training Log: training.log"
echo "  - Validation Log: validation.log"
echo "  - Best Checkpoint: model_best.pth.tar"
echo "  - Test Results: test_results.csv"
echo "  - Config: args.yaml"
echo ""

# Display test accuracy if available
if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results.csv" ]; then
    echo "Test Results:"
    cat "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results.csv"
    echo ""
fi

# Display MLflow UI command
echo "To view results in MLflow UI, run:"
echo "  mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI"
echo ""

# Display validation accuracy from summary
if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/summary.csv" ]; then
    echo "Training Progress Summary (last 5 epochs):"
    tail -n 6 "$OUTPUT_DIR/$EXPERIMENT_NAME/summary.csv" | column -t -s,
    echo ""
fi

echo "=========================================="
echo "Job Completed Successfully!"
echo "=========================================="
