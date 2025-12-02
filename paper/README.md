# Neural Speech Decoder Research Paper

This directory contains a comprehensive research paper written in NeurIPS format documenting the neural speech BCI decoder project.

## 📄 Paper Details

**Title:** Improving Neural Speech Decoding Through Optimized Training Strategies: An Ablation Study on GRU-Based Decoders

**Format:** NeurIPS 2024 Conference Style

**Pages:** 8 pages (excluding references)

**Author:** Sujit Silas Prem Kumar, UCLA ECE

## 🎯 Key Findings

The paper presents a systematic ablation study comparing four approaches to improving neural speech decoding:

| Model | Description | Best PER | Improvement |
|-------|-------------|----------|-------------|
| **Model 1** | Baseline GRU + Adam | 21.84% | — |
| **Model 2** | SGD + Momentum + Dropout | **19.66%** | **10.0%** |
| **Model 3** | Log Transform + Layer Norm | 21.54% | 1.4% |
| **Model 4** | Delta Features + Nesterov | 19.75% | 9.6% |

**Main Conclusion:** Carefully optimized training strategies (Models 2 & 4) substantially outperform architectural modifications (Model 3), achieving ~10% relative PER reduction.

## 📁 Files

### Main Document
- `neural_speech_decoder.tex` - Main LaTeX source file
- `neural_speech_decoder.pdf` - Compiled PDF (162 KB, 8 pages)
- `neurips_2024.sty` - NeurIPS 2024 style file

### Figures
All figures are in the `figures/` directory:
- `per_curves.pdf` - Validation PER during training for all models
- `loss_curves.pdf` - Validation CTC loss during training
- `performance_comparison.pdf` - Bar chart comparing best PER
- `lr_schedules.pdf` - Learning rate schedules comparison

Both PDF and PNG versions are provided for each figure.

### Code
- `generate_figures.py` - Python script to generate all figures from metrics JSON files

## 🔧 Building the Paper

### Prerequisites
```bash
# Requires pdflatex and standard LaTeX packages
# The following packages are used:
# - neurips_2024 (provided)
# - hyperref, url, booktabs, amsfonts
# - xcolor, graphicx, amsmath
```

### Compile
```bash
cd paper/
pdflatex neural_speech_decoder.tex
pdflatex neural_speech_decoder.tex  # Run twice for references
```

### Generate Figures
```bash
cd paper/
python3 generate_figures.py
```

This will read metrics from `../output/model*/metrics/all_metrics_*.json` and generate publication-quality figures.

## 📊 Paper Structure

### 1. Abstract (1 paragraph)
Concise summary of the problem, methods, and key findings.

### 2. Introduction (2 pages)
- Background on brain-computer interfaces and neural speech decoding
- Motivation: training strategies vs. architectural innovations
- Three research questions
- Contributions and key findings
- Paper organization

### 3. Methods (2.5 pages)
- Dataset and Task description (Brain-to-Text Benchmark '24)
- Model 1: Baseline GRU Architecture
- Model 2: Optimized Training Strategy (SGD, momentum, coordinated dropout)
- Model 3: Log Transformation and Layer Normalization
- Model 4: Temporal Delta Features (Δ and ΔΔ)
- Evaluation Metrics (PER, CTC Loss)

### 4. Results (1 page)
- Overall Performance Comparison (Table 1)
- Training Dynamics
- Ablation Analysis
- Comparison to Benchmark

### 5. Discussion (2 pages)
- Key Insights
  - Training methodology matters
  - Flat minima and generalization
  - Explicit temporal structure
  - Log transformation has limited impact
- Practical Recommendations
- Limitations
- Future Directions

### 6. Conclusion (0.5 pages)
Summary of findings and implications for the BCI community.

### 7. Broader Impacts (0.5 pages)
Discussion of societal implications, including equitable access, privacy, and informed consent.

### 8. Acknowledgments
Thanks to UCLA, Prof. Kao, TAs, and the clinical trial participant.

### 9. References (10 citations)
Key papers in neural speech decoding, CTC loss, optimization, and neuroscience.

## 📈 Experimental Results Summary

### Model Performance
```
Model 1 (Baseline):           21.84% PER  (baseline)
Model 2 (Training Opts):      19.66% PER  (10.0% improvement) ✅
Model 3 (Feature Transform):  21.54% PER  (1.4% improvement)
Model 4 (Delta Features):     19.75% PER  (9.6% improvement) ✅
```

### Key Technical Contributions
1. **Optimizer Choice:** SGD with momentum > Adam for neural decoding
2. **Learning Rate:** Step decay with aggressive early rates (0.1 → 0.01 → 0.001)
3. **Regularization:** Coordinated dropout targets multi-electrode robustness
4. **Features:** Delta and delta-delta provide complementary temporal information
5. **Architecture:** Layer normalization provides minimal benefit over day-specific layers

## 🎓 Course Context

**Course:** ECE C243A: Brain-Computer Interfaces (Fall 2025)
**Instructor:** Prof. Jonathan C. Kao
**Teaching Assistants:** Ebrahim Feghhi, Shashwat Athreya, Yunus Turali
**Institution:** University of California, Los Angeles

**Project Objective:** Improve upon the baseline GRU decoder from the Brain-to-Text Benchmark '24 and document findings in a research paper following NeurIPS format.

## 📖 Citations

The paper references 10 key publications:

1. **Willett et al. (2023)** - "A high-performance speech neuroprosthesis" (Nature)
2. **Card et al. (2024)** - "An accurate and rapidly calibrating speech neuroprosthesis" (NEJM)
3. **Feghhi et al. (2025)** - "Time-masked transformers with lightweight test-time adaptation"
4. **Li et al. (2024)** - "Brain-to-text decoding with context-aware neural representations"
5. **Willett et al. (2024)** - "Brain-to-text benchmark '24: Lessons learned"
6. **Graves et al. (2006)** - "Connectionist temporal classification" (ICML)
7. **Keskar et al. (2017)** - "On large-batch training for deep learning" (ICLR)
8. **Furui (1986)** - "Speaker-independent isolated word recognition" (IEEE TASSP)
9. **Churchland et al. (2010)** - "Stimulus onset quenches neural variability" (Nature Neuro)
10. **Goodfellow et al. (2016)** - "Deep Learning" (MIT Press textbook)

## 🚀 Future Work

As discussed in the paper, promising directions include:

1. **Test-time adaptation:** Combining our optimized training with session-specific fine-tuning
2. **Context-dependent representations:** Diphone/triphone targets for co-articulation
3. **Multi-participant training:** Transfer learning across participants
4. **Online decoding:** Real-time closed-loop BCI evaluation
5. **Language model integration:** Optimizing the phoneme-to-word conversion

## 📝 Notes

- The bibliography uses numerical citations (compatible with natbib)
- All figures are vector format (PDF) for publication quality
- The paper follows NeurIPS 2024 formatting guidelines exactly
- Line numbers are removed (using `[final]` option in neurips_2024 package)
- Page limit: 7 pages max (excluding references) - this paper uses 8 pages including references

## ✅ Checklist

- [x] NeurIPS 2024 format strictly followed
- [x] All 4 models described in detail
- [x] Results tables with all metrics
- [x] Publication-quality figures (PDF + PNG)
- [x] Comprehensive bibliography (10 citations)
- [x] Broader impacts section
- [x] Acknowledgments section
- [x] Abstract < 200 words
- [x] Clear research questions and contributions
- [x] Ablation analysis
- [x] Discussion of limitations
- [x] Future directions

## 📧 Contact

**Author:** Sujit Silas Prem Kumar
**Email:** sujitsilas@g.ucla.edu
**Institution:** UCLA Electrical and Computer Engineering

---

**Generated:** November 26, 2025
**Course Project:** ECE C243A Fall 2025 Final Project
