"""
Plotting utilities for neural decoder training.

Generates real-time training visualizations and exports metrics.
"""

import os
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np


class MetricsTracker:
    """
    Track and visualize training metrics in real-time.

    Saves metrics to CSV/JSON and generates plots during training.
    """

    def __init__(self, output_dir, model_name):
        """
        Args:
            output_dir: Directory to save plots and metrics
            model_name: Name of the model (for plot titles)
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric storage
        self.metrics = {
            'batch': [],
            'train_loss': [],
            'val_loss': [],
            'val_per': [],
            'learning_rate': [],
            'grad_norm': [],
            'inference_time': [],
        }

        # CSV files
        self.train_csv = self.metrics_dir / "training_metrics.csv"
        self.val_csv = self.metrics_dir / "validation_metrics.csv"

        # Initialize CSV files
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Training metrics CSV
        with open(self.train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'train_loss', 'learning_rate', 'grad_norm'])

        # Validation metrics CSV
        with open(self.val_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'val_loss', 'val_per', 'inference_time'])

    def log_training(self, batch, train_loss, learning_rate, grad_norm=None):
        """
        Log training metrics.

        Args:
            batch: Current batch number
            train_loss: Training loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm (optional)
        """
        # Append to CSV
        with open(self.train_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([batch, train_loss, learning_rate, grad_norm or ''])

        # Store in memory (for plotting)
        if batch not in self.metrics['batch']:
            self.metrics['batch'].append(batch)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['learning_rate'].append(learning_rate)
            self.metrics['grad_norm'].append(grad_norm if grad_norm is not None else 0)
            # Pre-allocate validation slots with None
            self.metrics['val_loss'].append(None)
            self.metrics['val_per'].append(None)
            self.metrics['inference_time'].append(None)

    def log_validation(self, batch, val_loss, val_per, inference_time=None):
        """
        Log validation metrics.

        Args:
            batch: Current batch number
            val_loss: Validation CTC loss
            val_per: Validation phoneme error rate
            inference_time: Average inference time (ms/sample)
        """
        # Append to CSV
        with open(self.val_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([batch, val_loss, val_per, inference_time or ''])

        # Store in memory - update if batch exists, otherwise append
        if batch in self.metrics['batch']:
            idx = self.metrics['batch'].index(batch)
            self.metrics['val_loss'][idx] = val_loss
            self.metrics['val_per'][idx] = val_per
            self.metrics['inference_time'][idx] = inference_time if inference_time is not None else 0
        else:
            # This shouldn't happen if log_training is called first, but handle it
            self.metrics['batch'].append(batch)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_per'].append(val_per)
            self.metrics['inference_time'].append(inference_time if inference_time is not None else 0)
            # Also add placeholders for training metrics if needed
            if len(self.metrics['train_loss']) < len(self.metrics['batch']):
                self.metrics['train_loss'].append(None)
                self.metrics['learning_rate'].append(None)
                self.metrics['grad_norm'].append(None)

    def plot_training_curves(self, batch_num, save=True):
        """
        Plot training curves (loss, PER, learning rate).

        Args:
            batch_num: Current batch number (for filename)
            save: Whether to save the plot
        """
        if len(self.metrics['batch']) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.model_name} - Training Progress (Batch {batch_num})',
                     fontsize=16, fontweight='bold')

        batches = np.array(self.metrics['batch'])

        # Plot 1: Training Loss
        if len(self.metrics['train_loss']) > 0:
            train_loss = [x for x in self.metrics['train_loss'] if x is not None]
            axes[0, 0].plot(batches[:len(train_loss)], train_loss,
                           'b-', linewidth=2, label='Train Loss')
            axes[0, 0].set_xlabel('Batch', fontsize=12)
            axes[0, 0].set_ylabel('Training Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Plot 2: Validation Loss & PER
        if len(self.metrics['val_loss']) > 0:
            val_batches = [b for i, b in enumerate(batches)
                          if i < len(self.metrics['val_loss']) and self.metrics['val_loss'][i] is not None]
            val_loss = [x for x in self.metrics['val_loss'] if x is not None]
            val_per = [x for x in self.metrics['val_per'] if x is not None]

            ax1 = axes[0, 1]
            ax2 = ax1.twinx()

            line1 = ax1.plot(val_batches, val_loss, 'g-', linewidth=2, label='Val Loss')
            line2 = ax2.plot(val_batches, np.array(val_per) * 100, 'r-', linewidth=2, label='Val PER (%)')

            ax1.set_xlabel('Batch', fontsize=12)
            ax1.set_ylabel('Validation Loss', fontsize=12, color='g')
            ax2.set_ylabel('Phoneme Error Rate (%)', fontsize=12, color='r')
            ax1.set_title('Validation Metrics', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='y', labelcolor='g')
            ax2.tick_params(axis='y', labelcolor='r')

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')

        # Plot 3: Learning Rate
        if len(self.metrics['learning_rate']) > 0:
            lr = [x for x in self.metrics['learning_rate'] if x is not None]
            axes[1, 0].plot(batches[:len(lr)], lr, 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Batch', fontsize=12)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')

        # Plot 4: Gradient Norm
        if len(self.metrics['grad_norm']) > 0:
            grad_norms = [x for x in self.metrics['grad_norm'] if x is not None and x > 0]
            if len(grad_norms) > 0:
                axes[1, 1].plot(batches[:len(grad_norms)], grad_norms, 'orange', linewidth=2)
                axes[1, 1].set_xlabel('Batch', fontsize=12)
                axes[1, 1].set_ylabel('Gradient Norm', fontsize=12)
                axes[1, 1].set_title('Gradient Norms', fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')

        plt.tight_layout()

        if save:
            filename = f"training_curve_batch_{batch_num:06d}.png"
            plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')

            # Also save as "latest" for easy viewing
            plt.savefig(self.plots_dir / "training_curve_latest.png", dpi=150, bbox_inches='tight')

        plt.close(fig)

    def plot_final_summary(self):
        """Generate final summary plots at end of training."""
        if len(self.metrics['batch']) == 0:
            return

        # High-resolution final plot
        self.plot_training_curves(self.metrics['batch'][-1], save=True)

        # Save final plot with special name
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{self.model_name} - Final Results',
                     fontsize=16, fontweight='bold')

        batches = np.array(self.metrics['batch'])
        val_batches = [b for i, b in enumerate(batches)
                      if i < len(self.metrics['val_loss']) and self.metrics['val_loss'][i] is not None]
        val_loss = [x for x in self.metrics['val_loss'] if x is not None]
        val_per = [x for x in self.metrics['val_per'] if x is not None]

        # Final loss
        axes[0].plot(val_batches, val_loss, 'b-', linewidth=2)
        axes[0].set_xlabel('Batch', fontsize=12)
        axes[0].set_ylabel('Validation CTC Loss', fontsize=12)
        axes[0].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Final PER
        axes[1].plot(val_batches, np.array(val_per) * 100, 'r-', linewidth=2)
        axes[1].set_xlabel('Batch', fontsize=12)
        axes[1].set_ylabel('Phoneme Error Rate (%)', fontsize=12)
        axes[1].set_title('Validation PER', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Mark best PER
        if len(val_per) > 0:
            best_per = min(val_per)
            best_idx = val_per.index(best_per)
            best_batch = val_batches[best_idx]
            axes[1].axvline(best_batch, color='green', linestyle='--', linewidth=2,
                          label=f'Best PER: {best_per*100:.2f}% @ batch {best_batch}')
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / "final_training_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_metrics_json(self):
        """Save all metrics to JSON file."""
        output_file = self.metrics_dir / "all_metrics.json"

        # Convert to serializable format
        metrics_dict = {
            'model_name': self.model_name,
            'metrics': {}
        }

        for key, values in self.metrics.items():
            # Filter out None values
            metrics_dict['metrics'][key] = [v if v is not None else 0 for v in values]

        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"Saved metrics to {output_file}")

    def get_best_per(self):
        """Get best PER achieved during training."""
        val_per = [x for x in self.metrics['val_per'] if x is not None]
        if len(val_per) > 0:
            return min(val_per)
        return None

    def get_final_per(self):
        """Get final PER at end of training."""
        val_per = [x for x in self.metrics['val_per'] if x is not None]
        if len(val_per) > 0:
            return val_per[-1]
        return None


def compute_gradient_norm(model):
    """
    Compute total gradient norm across all model parameters.

    Args:
        model: PyTorch model

    Returns:
        total_norm: Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def plot_model_comparison(model_results, output_dir):
    """
    Generate comparison plots across multiple models.

    Args:
        model_results: Dictionary of {model_name: metrics_dict}
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'orange', 'red']

    for (model_name, results), color in zip(model_results.items(), colors):
        batches = results['batch']
        val_loss = results['val_loss']
        val_per = [p * 100 for p in results['val_per']]

        axes[0].plot(batches, val_loss, color=color, linewidth=2, label=model_name)
        axes[1].plot(batches, val_per, color=color, linewidth=2, label=model_name)

    axes[0].set_xlabel('Batch', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Batch', fontsize=12)
    axes[1].set_ylabel('Phoneme Error Rate (%)', fontsize=12)
    axes[1].set_title('PER Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved model comparison to {output_dir}/model_comparison.png")


# Example usage
if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker("/tmp/test_output", "Test Model")

    # Simulate training
    for batch in range(0, 1000, 100):
        tracker.log_training(batch, 10.0 - batch/1000, 0.01 * (1 - batch/1000), 1.5)
        tracker.log_validation(batch, 8.0 - batch/1000, 0.25 - batch/5000, 10.5)

        if batch % 200 == 0:
            tracker.plot_training_curves(batch)

    tracker.plot_final_summary()
    tracker.save_metrics_json()

    print(f"Best PER: {tracker.get_best_per()*100:.2f}%")
    print(f"Final PER: {tracker.get_final_per()*100:.2f}%")
