#!/bin/bash
#$ -cwd
#$ -o joblog.model3_diphone
#$ -j y

# GPU Configuration
#$ -l gpu,A100,cuda=1,h_rt=4:00:00
#$ -l h_data=30G


# Email notifications
#$ -M $USER@g.ucla.edu
#$ -m bea

##############################################################################
# Model 3 (diphone): Log Transform + Layer Normalization
#
# Architecture: 5-layer unidirectional GRU
# - 1024 hidden units
# - dropout=0.4
# - Log transform on spikePow features (256 dims)
# - Layer normalization after GRU
#
# Data Augmentation:
# - Log transform stabilizes spike power features
# - Layer normalization reduces variability (biological quenching)
##############################################################################

echo "============================================================================"
echo "Training Model 3: Log Transform + Layer Normalization"
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

# Set dataset path (Log-transformed features)
DATASET_PATH="/u/scratch/s/sujit009/neural_seq_decoder/competitionData/ptDecoder_ctc_log"

echo ""
echo "Dataset: ${DATASET_PATH} (Log-transformed spikePow)"
echo "Input features: 256 dimensions (128 tx1 + 128 log(spikePow))"
echo "Output directory will be auto-generated with timestamp"
echo ""

# Run training (single GPU)
echo "============================================================================"
echo "Starting training on single GPU..."
echo "============================================================================"
echo ""

python scripts/train_ablation_hoffman2.py \
    --model_type diphone \
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
    echo "✓ Model 3 training completed successfully"
    echo "Results saved to: /u/scratch/s/sujit009/neural_seq_decoder/outputs/"
else
    echo "✗ Model 3 training failed with exit code ${EXIT_STATUS}"
fi

exit $EXIT_STATUS
