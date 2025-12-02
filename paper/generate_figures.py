"""
Generate figures for the neural speech decoder paper.

This script reads the metrics JSON files from all four models and generates
publication-quality figures for the NeurIPS paper.

Usage:
    python generate_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set publication-quality plotting defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 11
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 0.8

# Define colors for each model
COLORS = {
    'Model 1': '#1f77b4',  # Blue
    'Model 2': '#ff7f0e',  # Orange
    'Model 3': '#2ca02c',  # Green
    'Model 4': '#d62728',  # Red
}

def load_metrics(model_dir, model_key):
    """Load metrics from JSON file."""
    base_path = Path(__file__).parent.parent / 'output'
    json_path = base_path / model_dir / 'metrics' / f'all_metrics_{model_key}.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data['metrics']

def filter_valid_metrics(batches, values):
    """Filter out zero and NaN values from metrics."""
    batches = np.array(batches)
    values = np.array(values)

    # Keep only valid evaluation points (non-zero, non-NaN)
    valid_mask = (values > 0) & (~np.isnan(values))
    return batches[valid_mask], values[valid_mask]

def plot_per_curves():
    """Generate PER vs training batches plot."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    models = {
        'Model 1': ('model1_baseline_gru_20251123_153533', 'model1'),
        'Model 2': ('model2_sgd_momentum_dropout_20251124_165547', 'model2'),
        'Model 3': ('model3_log_layer_norm_20251124_181018', 'model3'),
        'Model 4': ('model4_sgd_nesterov_delta_20251124_151323', 'model4'),
    }

    for model_name, (model_dir, model_key) in models.items():
        metrics = load_metrics(model_dir, model_key)
        batches, per_values = filter_valid_metrics(metrics['batch'], metrics['val_per'])

        # Convert to percentage
        per_values = per_values * 100

        # Plot with smoothing
        ax.plot(batches, per_values, label=model_name,
                color=COLORS[model_name], alpha=0.8)

    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Validation Phoneme Error Rate (%)')
    ax.set_title('Validation PER During Training')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 10000)
    ax.set_ylim(18, 35)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'per_curves.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(Path(__file__).parent / 'figures' / 'per_curves.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated per_curves.pdf/png")

def plot_loss_curves():
    """Generate validation loss vs training batches plot."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    models = {
        'Model 1': ('model1_baseline_gru_20251123_153533', 'model1'),
        'Model 2': ('model2_sgd_momentum_dropout_20251124_165547', 'model2'),
        'Model 3': ('model3_log_layer_norm_20251124_181018', 'model3'),
        'Model 4': ('model4_sgd_nesterov_delta_20251124_151323', 'model4'),
    }

    for model_name, (model_dir, model_key) in models.items():
        metrics = load_metrics(model_dir, model_key)
        batches, loss_values = filter_valid_metrics(metrics['batch'], metrics['val_loss'])

        ax.plot(batches, loss_values, label=model_name,
                color=COLORS[model_name], alpha=0.8)

    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Validation CTC Loss')
    ax.set_title('Validation Loss During Training')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 10000)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'loss_curves.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(Path(__file__).parent / 'figures' / 'loss_curves.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated loss_curves.pdf/png")

def plot_comparison_bars():
    """Generate bar chart comparing best PER across models."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    models = {
        'Model 1': ('model1_baseline_gru_20251123_153533', 'model1'),
        'Model 2': ('model2_sgd_momentum_dropout_20251124_165547', 'model2'),
        'Model 3': ('model3_log_layer_norm_20251124_181018', 'model3'),
        'Model 4': ('model4_sgd_nesterov_delta_20251124_151323', 'model4'),
    }

    model_names = []
    best_pers = []
    colors = []

    for model_name, (model_dir, model_key) in models.items():
        metrics = load_metrics(model_dir, model_key)
        _, per_values = filter_valid_metrics(metrics['batch'], metrics['val_per'])

        best_per = np.min(per_values) * 100  # Convert to percentage

        model_names.append(model_name)
        best_pers.append(best_per)
        colors.append(COLORS[model_name])

    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, best_pers, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, per) in enumerate(zip(bars, best_pers)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{per:.2f}%', ha='center', va='bottom', fontsize=9)

    # Add relative improvement labels
    baseline_per = best_pers[0]
    for i, (bar, per) in enumerate(zip(bars, best_pers)):
        if i > 0:  # Skip baseline
            improvement = ((baseline_per - per) / baseline_per) * 100
            ax.text(bar.get_x() + bar.get_width()/2., per/2,
                    f'{improvement:+.1f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Best Validation PER (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, max(best_pers) * 1.15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'performance_comparison.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(Path(__file__).parent / 'figures' / 'performance_comparison.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated performance_comparison.pdf/png")

def plot_learning_rate_schedules():
    """Generate learning rate schedule comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    batches = np.arange(0, 10000)

    # Model 1: Linear decay
    lr1_start, lr1_end = 0.02, 0.002
    lr1 = lr1_start + (lr1_end - lr1_start) * batches / 10000
    ax.plot(batches, lr1, label='Model 1 (Linear Decay)',
            color=COLORS['Model 1'], linewidth=2)

    # Model 2: Step decay
    lr2 = np.ones_like(batches, dtype=float) * 0.1
    lr2[batches >= 4000] = 0.01
    lr2[batches >= 8000] = 0.001
    ax.plot(batches, lr2, label='Model 2 (Step Decay)',
            color=COLORS['Model 2'], linewidth=2)

    # Model 4: Step decay (different schedule)
    lr4 = np.ones_like(batches, dtype=float) * 0.1
    lr4[batches >= 5000] = 0.01
    ax.plot(batches, lr4, label='Model 4 (Step Decay)',
            color=COLORS['Model 4'], linewidth=2)

    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.set_yscale('log')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 10000)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'lr_schedules.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig(Path(__file__).parent / 'figures' / 'lr_schedules.png',
                dpi=300, bbox_inches='tight')
    print("✓ Generated lr_schedules.pdf/png")

def main():
    """Generate all figures."""
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("Generating figures for neural speech decoder paper...")
    print("=" * 60)

    try:
        plot_per_curves()
        plot_loss_curves()
        plot_comparison_bars()
        plot_learning_rate_schedules()

        print("=" * 60)
        print("✓ All figures generated successfully!")
        print(f"  Output directory: {figures_dir}")
        print("\nGenerated files:")
        for f in sorted(figures_dir.glob('*.pdf')):
            print(f"  - {f.name}")

    except Exception as e:
        print(f"✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
