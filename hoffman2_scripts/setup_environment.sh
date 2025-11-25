#!/bin/bash
#$ -cwd
#$ -o joblog.setup_environment
#$ -j y
#$ -l h_rt=1:00:00
#$ -l h_data=8G

# Email notifications
#$ -M $USER@g.ucla.edu
#$ -m bea

##############################################################################
# Environment Setup Script for Hoffman2
#
# This script sets up the Python environment using uv for fast, reproducible
# package management. Only needs to be run ONCE.
#
# Usage:
#   qsub hoffman2_scripts/setup_environment.sh
#
##############################################################################

echo "============================================================================"
echo "Setting up Python environment for Neural Speech Decoder"
echo "============================================================================"
echo "Start time: $(date)"
echo ""

# Navigate to project directory
cd /u/scratch/s/sujit009/neural_seq_decoder

# Load required modules
echo "Loading modules..."
. /etc/bashrc
module load python
module load cuda/12.3

# Install uv if not already installed
echo ""
echo "Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed"
fi

# Create virtual environment with uv
echo ""
echo "Creating virtual environment..."
uv venv --python 3.9 .venv

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "Installing local package..."
uv pip install -e .

source ~/.bashrc

# Deactivate
deactivate

echo ""
echo "============================================================================"
echo "Environment setup complete!"
echo "End time: $(date)"
echo "============================================================================"
echo ""
echo "To activate the environment in future jobs, use:"
echo "  source /u/scratch/s/sujit009/neural_seq_decoder/.venv/bin/activate"
echo ""
