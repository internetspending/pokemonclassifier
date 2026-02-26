"""
Experiment Tracker for Pokemon Classifier
Logs configurations, metrics, and model checkpoints for each training run
"""

import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd


class ExperimentTracker:
    """
    Tracks experiments by saving configs, metrics, and checkpoints.
    Creates a unique folder for each experiment run.
    """
    
    def __init__(self, experiment_name, base_dir='experiments', description=None):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment (e.g., 'efficientnet_b0_baseline')
            base_dir: Base directory to store all experiments
            description: Optional description of what you're testing
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.plot_dir = self.exp_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        
        # Storage for metrics and config
        self.metrics = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'val_top5_acc': [],
            'learning_rates': [],
            'epoch': []
        }
        self.config = {}
        self.best_val_f1 = 0.0
        self.description = description
        
        # Save initial info
        self._save_experiment_info()
        
    def _save_experiment_info(self):
        """Save basic experiment information"""
        info = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'description': self.description,
            'experiment_directory': str(self.exp_dir)
        }
        
        with open(self.exp_dir / 'experiment_info.json', 'w') as f:
            json.dump(info, f, indent=4)
    
    def log_config(self, config_dict):
        """
        Log experiment configuration
        
        Args:
            config_dict: Dictionary containing all hyperparameters and settings
        """
        self.config = config_dict
        
        # Save config as JSON
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Also save as readable text
        with open(self.exp_dir / 'config.txt', 'w') as f:
            f.write("=" * 50 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            if self.description:
                f.write(f"Description: {self.description}\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
    
    def log_epoch_metrics(self, epoch, train_loss=None, train_f1=None, 
                         val_loss=None, val_f1=None, val_top5_acc=None, 
                         learning_rate=None):
        """
        Log metrics for a single epoch
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_f1: Training F1 score
            val_loss: Validation loss
            val_f1: Validation F1 score
            val_top5_acc: Validation top-5 accuracy
            learning_rate: Current learning rate
        """
        self.metrics['epoch'].append(epoch)
        
        if train_loss is not None:
            self.metrics['train_loss'].append(float(train_loss))
        if train_f1 is not None:
            self.metrics['train_f1'].append(float(train_f1))
        if val_loss is not None:
            self.metrics['val_loss'].append(float(val_loss))
        if val_f1 is not None:
            self.metrics['val_f1'].append(float(val_f1))
        if val_top5_acc is not None:
            self.metrics['val_top5_acc'].append(float(val_top5_acc))
        if learning_rate is not None:
            self.metrics['learning_rates'].append(float(learning_rate))
        
        # Save metrics after each epoch
        self._save_metrics()
        
        # Update plots
        self._update_plots()
    
    def _save_metrics(self):
        """Save metrics to JSON and CSV"""
        # Save as JSON
        with open(self.exp_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save as CSV for easy viewing in Excel/spreadsheets
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.exp_dir / 'metrics.csv', index=False)
    
    def _update_plots(self):
        """Create/update training plots"""
        if len(self.metrics['epoch']) == 0:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.experiment_name} - Training Progress', fontsize=16)
        
        epochs = self.metrics['epoch']
        
        # Plot 1: Loss
        if self.metrics['train_loss'] and self.metrics['val_loss']:
            axes[0, 0].plot(epochs, self.metrics['train_loss'], label='Train Loss', marker='o')
            axes[0, 0].plot(epochs, self.metrics['val_loss'], label='Val Loss', marker='o')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: F1 Score
        if self.metrics['train_f1'] and self.metrics['val_f1']:
            axes[0, 1].plot(epochs, self.metrics['train_f1'], label='Train F1', marker='o')
            axes[0, 1].plot(epochs, self.metrics['val_f1'], label='Val F1', marker='o')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('F1 Score Progress')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Top-5 Accuracy
        if self.metrics['val_top5_acc']:
            axes[1, 0].plot(epochs, self.metrics['val_top5_acc'], 
                           label='Val Top-5 Acc', marker='o', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Validation Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot 4: Learning Rate
        if self.metrics['learning_rates']:
            axes[1, 1].plot(epochs, self.metrics['learning_rates'], 
                           marker='o', color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, model, optimizer, epoch, val_f1, is_best=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            val_f1: Validation F1 score for this checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_val_f1 = val_f1
            
            # Log best model info
            with open(self.exp_dir / 'best_model_info.txt', 'w') as f:
                f.write(f"Best Model Checkpoint\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Validation F1: {val_f1:.4f}\n")
                f.write(f"Checkpoint: {best_path}\n")
        
        # Keep only last 3 checkpoints to save space (optional)
        self._cleanup_old_checkpoints(keep_last=3)
    
    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoints, keeping only the last N and best model"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def save_final_summary(self):
        """Save final summary of the experiment"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'description': self.description,
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'total_epochs': len(self.metrics['epoch']),
            'final_metrics': {
                'train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
                'train_f1': self.metrics['train_f1'][-1] if self.metrics['train_f1'] else None,
                'val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
                'val_f1': self.metrics['val_f1'][-1] if self.metrics['val_f1'] else None,
                'val_top5_acc': self.metrics['val_top5_acc'][-1] if self.metrics['val_top5_acc'] else None,
            }
        }
        
        with open(self.exp_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create a nice readable summary
        with open(self.exp_dir / 'SUMMARY.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {self.timestamp}\n")
            if self.description:
                f.write(f"Description: {self.description}\n")
            f.write(f"\nTotal Epochs: {len(self.metrics['epoch'])}\n")
            f.write(f"Best Validation F1: {self.best_val_f1:.4f}\n\n")
            
            f.write("Final Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in summary['final_metrics'].items():
                if value is not None:
                    f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Configuration:\n")
            f.write("=" * 60 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Results saved to: {self.exp_dir}")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
    
    def get_experiment_path(self):
        """Return the path to this experiment's directory"""
        return str(self.exp_dir)


# Helper function to compare multiple experiments
def compare_experiments(experiment_dirs):
    """
    Compare multiple experiments side by side
    
    Args:
        experiment_dirs: List of paths to experiment directories
    """
    results = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        
        # Load summary
        summary_path = exp_path / 'final_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                results.append(summary)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results:
        row = {
            'Experiment': result['experiment_name'],
            'Best Val F1': result['best_val_f1'],
            'Final Train F1': result['final_metrics'].get('train_f1', 'N/A'),
            'Final Val F1': result['final_metrics'].get('val_f1', 'N/A'),
            'Epochs': result['total_epochs'],
        }
        
        # Add key config parameters
        config = result.get('config', {})
        row['Learning Rate'] = config.get('learning_rate', 'N/A')
        row['Batch Size'] = config.get('batch_size', 'N/A')
        row['Weight Decay'] = config.get('weight_decay', 'N/A')
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Best Val F1', ascending=False)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Experiment Tracker Example")
    print("="*50)
    
    # Create tracker
    tracker = ExperimentTracker(
        experiment_name="efficientnet_b0_baseline",
        description="Testing baseline model with default hyperparameters"
    )
    
    # Log configuration
    config = {
        'model': 'efficientnet_b0',
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 50,
        'weight_decay': 0.01,
        'dropout': 0.4,
        'optimizer': 'AdamW'
    }
    tracker.log_config(config)
    
    # Simulate training for 5 epochs
    print("\nSimulating training...")
    for epoch in range(1, 6):
        # Simulate metrics (replace with actual training loop)
        train_loss = 2.0 - (epoch * 0.3)
        train_f1 = 0.2 + (epoch * 0.15)
        val_loss = 2.5 - (epoch * 0.2)
        val_f1 = 0.15 + (epoch * 0.05)
        val_top5_acc = 0.4 + (epoch * 0.1)
        lr = 0.0001 * (0.9 ** epoch)
        
        tracker.log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_f1=train_f1,
            val_loss=val_loss,
            val_f1=val_f1,
            val_top5_acc=val_top5_acc,
            learning_rate=lr
        )
        
        # Save checkpoint (simulate best at epoch 4)
        is_best = (epoch == 4)
        # tracker.save_checkpoint(model, optimizer, epoch, val_f1, is_best)
        
        print(f"Epoch {epoch}: Train F1={train_f1:.3f}, Val F1={val_f1:.3f}")
    
    # Save final summary
    tracker.save_final_summary()
    
    print(f"\nExperiment saved to: {tracker.get_experiment_path()}")