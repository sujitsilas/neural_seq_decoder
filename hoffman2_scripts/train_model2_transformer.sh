#!/bin/bash
#$ -cwd
#$ -o joblog.model2_transformer
#$ -j y

# GPU Configuration
# Request 1x A100 GPU with 80GB VRAM (CUDA capability 8.0)
#$ -l gpu,A100,cuda=1,h_rt=4:00:00
#$ -l h_data=30G


# Email notifications
#$ -M $USER@g.ucla.edu
#$ -m bea

##############################################################################
# Model 2: SGD with Momentum + Advanced Training
#
# Architecture: 5-layer unidirectional GRU (same as Model 1)
# - 1024 hidden units
# - dropout=0.4
#
# Training Improvements:
# - SGD with momentum (0.9) - paper found competitive with Adam
# - Step LR decay (0.1 → 0.01 → 0.001 at 4000/8000 steps)
# - Coordinated dropout (speckled masking) 10% rate
# - Gradient clipping (norm=5.0)
#
##############################################################################

echo "============================================================================"
echo "Training Model 2: Advanced Training Techniques"
echo "============================================================================"
echo "Job ID: $JOB_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd /u/scratch/s/sujit009/neural_seq_decoder

# Load required modules
echo "Loading modules..."
. /etc/bashrc
module load python
module load cuda


# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set dataset path
DATASET_PATH="/u/scratch/s/sujit009/neural_seq_decoder/competitionData/ptDecoder_ctc"

echo ""
echo "Dataset: ${DATASET_PATH} (Raw features)"
echo "Optimizer: SGD with momentum (0.9)"
echo "Training: Step LR Decay + Coordinated Dropout 10% + Gradient Clipping"
echo "Output directory will be auto-generated with timestamp"
echo ""

# Run training (single GPU)
echo "============================================================================"
echo "Starting training on single GPU..."
echo "============================================================================"
echo ""

python scripts/train_ablation_hoffman2.py \
    --model_type conformer \
    --save_interval 500 \
    --plot_interval 100

# Check exit status
EXIT_STATUS=$?

# Deactivate environment
deactivate

echo ""
echo "============================================================================"
echo "Training complete!"
echo "Exit status: ${EXIT_STATUS}"
echo "End time: $(date)"
echo "============================================================================"
echo ""

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ Model 2 training completed successfully"
    echo "Results saved to: /u/scratch/s/sujit009/neural_seq_decoder/outputs/"
else
    echo "✗ Model 2 training failed with exit code ${EXIT_STATUS}"
fi

exit $EXIT_STATUS
