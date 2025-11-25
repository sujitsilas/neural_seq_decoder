# Neural Speech Decoder

4-model ablation study exploring data augmentation and training techniques for neural speech decoding. Based on 2024 BCI competition findings.

## Results

| Model | Architecture | Key Features | Best PER |
|-------|-------------|--------------|----------|
| **Model 1** | GRU Baseline | 5-layer GRU (1024 units) + Adam | **21.84%** |
| **Model 2** | GRU + Training | SGD momentum + Coordinated dropout 10% | **19.66%** ✅ |
| **Model 3** | GRU + Log | Log transform + Layer normalization | **21.54%** |
| **Model 4** | GRU + Delta | Log + Δ/ΔΔ features + SGD Nesterov + Dropout 15% | **19.75%** ✅ |

**Key Finding**: Training techniques (optimizer, LR schedule, regularization) outperformed architectural changes.

---

## Quick Start

### 1. Format Datasets
```bash
source .venv/bin/activate
jupyter notebook notebooks/formatCompetitionData.ipynb
# Creates: ptDecoder_ctc, ptDecoder_ctc_log, ptDecoder_ctc_log_delta
```

### 2. Setup & Train
```bash
ssh your_username@hoffman2.idre.ucla.edu
bash hoffman2_scripts/setup_environment.sh

# Submit jobs
qsub hoffman2_scripts/train_model1_baseline.sh
qsub hoffman2_scripts/train_model2_transformer.sh
qsub hoffman2_scripts/train_model3_diphone.sh
qsub hoffman2_scripts/train_model4_conformer.sh
```

---

## Model Details

### Model 1: Baseline
- **Architecture**: 5-layer unidirectional GRU, 1024 hidden units
- **Dataset**: Raw features (256 dims: 128 tx1 + 128 spikePow)
- **Optimizer**: Adam (lr=0.02, linear decay)
- **Regularization**: White noise (0.8), constant offset (0.2), Gaussian smoothing (2.0)
- **Parameters**: 56.7M
- **PER**: 21.84%

### Model 2: Advanced Training
- **Architecture**: Same as Model 1
- **Dataset**: Raw features (256 dims)
- **Optimizer**: **SGD with momentum (0.9)**
- **LR Schedule**: Step decay (0.1 → 0.01 → 0.001 at 4k/8k batches)
- **Regularization**: Base + **Coordinated dropout (10%)**
- **Gradient Clipping**: 5.0
- **Parameters**: 56.7M
- **PER**: 19.66% (**2.2% improvement**)

**Experimenting**: Speckled masking from - randomly masks 10% of input features per batch.

### Model 3: Log Transform
- **Architecture**: 5-layer GRU + **Layer normalization**
- **Dataset**: **Log-transformed spikePow** (256 dims: 128 tx1 + 128 log(spikePow))
- **Optimizer**: Adam (lr=0.02)
- **Regularization**: Base augmentations
- **Parameters**: 56.7M + 2K (LayerNorm)
- **PER**: 21.54% (slight improvement)

**Experimenting**: Log transform stabilizes spike power distribution.

### Model 4: Delta Features
- **Architecture**: Same as Model 1
- **Dataset**: **Log + Delta features (768 dims)**
  - 256 static: log-transformed features
  - 256 velocity (Δ): first-order temporal derivative
  - 256 acceleration (ΔΔ): second-order temporal derivative
- **Optimizer**: **SGD with Nesterov momentum (0.9)**
- **LR Schedule**: Step decay (0.1 → 0.01 at 5k batches)
- **Regularization**: Base + **Coordinated dropout (15%)**
- **Gradient Clipping**: 5.0
- **Parameters**: 56.7M
- **PER**: 19.75% (**2.1% improvement**)

**Experimenting**: Delta features capture temporal dynamics; Nesterov momentum's "lookahead" complements velocity/acceleration features.

---

### Data Augmentations
- **Base** (all models): White noise (0.8), constant offset (0.2), Gaussian smoothing (2.0)
- **Log Transform** (Models 3 & 4): `log(spikePow + ε)` stabilizes distribution
- **Delta Features** (Model 4): Velocity and acceleration computed via central difference

---

## Project Structure

```
neural_seq_decoder/
├── src/neural_decoder/
│   ├── model.py              # Baseline GRU decoder
│   ├── model_diphone.py      # GRU + Layer normalization
│   ├── dataset.py            # PyTorch dataset
│   ├── augmentations.py      # Gaussian smoothing
│   └── plotting.py           # Metrics tracking
├── scripts/
│   └── train_ablation_hoffman2.py  # Main training script
├── hoffman2_scripts/         # Cluster submission scripts
└── notebooks/                # Data formatting (creates 3 datasets)
```

---

## Monitor Training

```bash
# Check job status
qstat -u $USER

# View logs
tail -f joblog.model*

# Check results
ls -lh /u/scratch/s/$USER/neural_seq_decoder/outputs/
```

---

## Citation

Based on findings from:
- [2024 BCI Competition Report](https://arxiv.org/html/2412.17227v1)
- [Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding](https://arxiv.org/html/2507.02800v2)
