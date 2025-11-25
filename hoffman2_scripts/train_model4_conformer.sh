#!/bin/bash
#$ -cwd
#$ -o joblog.model4_conformer
#$ -j y

# GPU Configuration
#$ -l gpu,A100,cuda=1,h_rt=4:00:00
#$ -l h_data=30G

# Email notifications
#$ -M $USER@g.ucla.edu
#$ -m bea

##############################################################################
# Model 4: Log+Delta Features + SGD Nesterov
#
# Architecture: 5-layer unidirectional GRU
# - 1024 hidden units
# - dropout=0.4
# - 768 input features (static + velocity + acceleration)
#
# Data Features:
# - Log transform on spikePow features
# - Delta features: velocity (d/dt) and acceleration (d²/dt²)
# - Total: 256 static + 256 Δ + 256 ΔΔ = 768 dims
#
# Usage:
#   qsub hoffman2_scripts/train_model4_conformer.sh
##############################################################################

echo "============================================================================"
echo "Training Model 4: Log Transform + Delta Features"
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

# Set dataset path (Log + Delta features)
DATASET_PATH="/u/scratch/s/sujit009/neural_seq_decoder/competitionData/ptDecoder_ctc_log_delta"

echo ""
echo "Dataset: ${DATASET_PATH} (Log + Delta features)"
echo "Input: 768 dims (256 static + 256 Δ + 256 ΔΔ)"
echo "Optimizer: SGD with Nesterov momentum (0.9)"
echo "Training: Step LR Decay + Coordinated Dropout 15% + Gradient Clipping"
echo "Output directory will be auto-generated with timestamp"
echo ""

# Run training (single GPU)
echo "============================================================================"
echo "Starting training on single GPU..."
echo "============================================================================"
echo ""

python scripts/train_ablation_hoffman2.py \
    --model_type transformer \
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
    echo "✓ Model 4 training completed successfully"
    echo "Results saved to: /u/scratch/s/sujit009/neural_seq_decoder/outputs/"
else
    echo "✗ Model 4 training failed with exit code ${EXIT_STATUS}"
fi

exit $EXIT_STATUS
