# Paper Updates - November 26, 2025

## Summary of Changes

The research paper has been updated with the following improvements:

### 1. **Author Name Corrected**
- Changed from "Sujit Silas Prem Kumar" to **"Sujit Silas Armstrong Suthahar"**

### 2. **Figures Added**
Added two high-quality figures to the Results section:

- **Figure 1 (per_curves.pdf):** Validation PER trajectories for all four models during training
  - Shows Models 2 and 4 achieving substantially lower PER than Models 1 and 3
  - Clearly visualizes the ~10% improvement from optimized training strategies

- **Figure 2 (loss_curves.pdf):** Validation CTC loss curves
  - Demonstrates smooth convergence without overfitting
  - Validates effective regularization through dropout and augmentation

Both figures are referenced in the Training Dynamics subsection (Section 3.2).

### 3. **Model 4 Parameters Corrected**
Updated the Model 4 (Temporal Delta Features) section to include complete architectural and training details:

**Previous:** Only listed optimization hyperparameters in bullet points

**Updated:** Full paragraph describing:
- Same 5-layer GRU architecture as baseline (1024 hidden units)
- Orthogonal weight initialization
- Expanded input dimension to 768 features (256 static + 256 Δ + 256 ΔΔ)
- Complete training configuration:
  - SGD with Nesterov momentum (0.9)
  - Step LR schedule: 0.1 → 0.01 at batch 5000
  - Coordinated dropout: 15%
  - Batch size: 64
  - Gradient clipping: max norm 5.0
  - Standard augmentations (white noise, offset, Gaussian smoothing)
- Total parameter count: ~56.7M (only input layer changes)

### 4. **Converted Bullet Points to Paragraphs**
Transformed all discussion sections from bullet-point lists to flowing prose paragraphs for better readability:

#### Key Insights (Section 4.1)
- **Before:** 4 separate bullet points
- **After:** 4 cohesive paragraphs discussing:
  1. Training methodology importance
  2. SGD vs. Adam and flat minima
  3. Explicit temporal structure in delta features
  4. Limited impact of log transformation

#### Practical Recommendations (Section 4.2)
- **Before:** 5 enumerated points
- **After:** 4 connected paragraphs covering:
  1. Hyperparameter tuning priority
  2. SGD with momentum for final training
  3. Delta features consideration
  4. Coordinated dropout for robustness

#### Limitations (Section 4.3)
- **Before:** 5 separate bold-labeled paragraphs
- **After:** 3 flowing paragraphs discussing:
  1. Single participant and validation-only limitations
  2. Limited architecture exploration
  3. Computational constraints and hyperparameter search

#### Future Directions (Section 4.4)
- **Before:** 5 separate bold-labeled paragraphs
- **After:** 3 comprehensive paragraphs covering:
  1. Test-time adaptation integration
  2. Context-dependent representations and multi-participant training
  3. Online deployment and language model integration

#### Broader Impacts
- **Before:** 3 bold-labeled bullet points
- **After:** 3 cohesive paragraphs:
  1. Positive impact on quality of life
  2. Equitable access concerns
  3. Privacy and informed consent considerations

## Technical Details

### Compilation
```bash
cd paper/
pdflatex neural_speech_decoder.tex
pdflatex neural_speech_decoder.tex  # Second pass for references
```

### File Statistics
- **Pages:** 10 (increased from 8 due to figures and expanded text)
- **File size:** ~201 KB (increased from 162 KB due to embedded figures)
- **Format:** NeurIPS 2024 (final option, no line numbers)
- **Figures:** 2 embedded PDF figures at 85% linewidth

### Writing Quality Improvements

1. **Better Flow:** Paragraphs connect ideas more naturally than bullet points
2. **Academic Tone:** Maintains formal but readable prose throughout
3. **Transitions:** Added smooth transitions between related concepts
4. **Depth:** Expanded explanations provide more context and reasoning
5. **Readability:** Easier to follow narrative structure

## Before/After Examples

### Example 1: Key Insights
**Before:**
```
\textbf{Training methodology matters:} Our central finding is that...
\textbf{Flat minima and generalization:} The superior performance of...
```

**After:**
```
Our central finding is that careful optimization... This challenges the field's
emphasis on architectural novelty... The results demonstrate that training
methodology matters as much as, if not more than, architectural choices...

The superior performance of SGD with momentum... aligns with findings in
computer vision... The step learning rate schedule with aggressive early rates...
```

### Example 2: Practical Recommendations
**Before:**
```
\begin{enumerate}
    \item \textbf{Invest in hyperparameter tuning:} Allocate significant effort...
    \item \textbf{Prefer SGD with momentum:} While Adam enables...
```

**After:**
```
Based on our findings, we recommend several key practices... First, practitioners
should allocate significant effort... Our results demonstrate that hyperparameter
tuning can yield gains...

Second, while Adam enables faster initial prototyping... The evidence suggests
that the flatter minima discovered by SGD translate to more robust performance...
```

## Validation

All sections reviewed for:
- [x] Grammatical correctness
- [x] Consistent verb tense (present for findings, past for methods)
- [x] Proper citation formatting
- [x] Logical flow and transitions
- [x] Technical accuracy
- [x] Figure references and captions
- [x] Paragraph coherence

## Next Steps (Optional)

If further improvements are desired:

1. **Add comparison bar chart:** Use `performance_comparison.pdf` figure
2. **Add learning rate schedule figure:** Use `lr_schedules.pdf`
3. **Expand results section:** Add more detailed ablation analysis
4. **Include appendix:** Detailed hyperparameter tables

## Files Modified

- `neural_speech_decoder.tex` - Main LaTeX source (updated)
- `neural_speech_decoder.pdf` - Compiled paper (regenerated, 10 pages)

## Files Unchanged

- `neurips_2024.sty` - NeurIPS style file
- `generate_figures.py` - Figure generation script
- `figures/*.pdf` - All 4 figures (already generated)
- `README.md` - Paper documentation

---

**Date:** November 26, 2025
**Updated by:** Claude (Sonnet 4.5)
**Total Changes:** 4 major improvements (author name, figures, Model 4 params, prose conversion)
