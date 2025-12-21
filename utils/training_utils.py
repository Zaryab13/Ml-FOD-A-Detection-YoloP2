"""
Training Utilities and Helper Functions
Includes callbacks, logging, and model management utilities
"""

import os
import yaml
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Custom logger for tracking training progress"""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}_log.json"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"
        
        self.logs = []
        self.best_metrics = {}
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for a single epoch"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.logs.append(entry)
        
        # Save to JSON
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        # Append to CSV
        if epoch == 0:
            # Write header
            with open(self.metrics_file, 'w') as f:
                f.write(','.join(['epoch'] + list(metrics.keys())) + '\n')
        
        with open(self.metrics_file, 'a') as f:
            values = [str(epoch)] + [str(metrics.get(k, '')) for k in metrics.keys()]
            f.write(','.join(values) + '\n')
    
    def update_best_metrics(self, metrics: Dict[str, float], epoch: int):
        """Track best metrics achieved"""
        for key, value in metrics.items():
            if key not in self.best_metrics or value > self.best_metrics[key]['value']:
                self.best_metrics[key] = {
                    'value': value,
                    'epoch': epoch
                }
    
    def save_summary(self):
        """Save training summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.logs),
            'best_metrics': self.best_metrics,
            'final_metrics': self.logs[-1] if self.logs else {}
        }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


class ModelCheckpoint:
    """Handle model checkpoint saving and loading"""
    
    def __init__(self, checkpoint_dir: Path, save_top_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.checkpoints = []
    
    def save_checkpoint(self, 
                       model: Any,
                       epoch: int,
                       metrics: Dict[str, float],
                       optimizer_state: Optional[Dict] = None):
        """Save model checkpoint"""
        
        checkpoint_name = f"epoch_{epoch}_map{metrics.get('mAP50', 0):.4f}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer_state:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'mAP50': metrics.get('mAP50', 0)
        })
        
        # Keep only top-k checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only top-k by mAP"""
        if len(self.checkpoints) <= self.save_top_k:
            return
        
        # Sort by mAP50 descending
        sorted_ckpts = sorted(self.checkpoints, 
                            key=lambda x: x['mAP50'], 
                            reverse=True)
        
        # Remove checkpoints beyond top-k
        for ckpt in sorted_ckpts[self.save_top_k:]:
            if ckpt['path'].exists():
                ckpt['path'].unlink()
        
        # Update checkpoint list
        self.checkpoints = sorted_ckpts[:self.save_top_k]
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint by mAP"""
        if not self.checkpoints:
            return None
        
        best = max(self.checkpoints, key=lambda x: x['mAP50'])
        return best['path']


def plot_training_curves(metrics_csv: Path, output_dir: Path):
    """Plot training curves from metrics CSV"""
    import pandas as pd
    
    df = pd.read_csv(metrics_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # mAP curves
    if 'mAP50' in df.columns:
        axes[0, 0].plot(df['epoch'], df['mAP50'], label='mAP50', linewidth=2)
    if 'mAP50-95' in df.columns:
        axes[0, 0].plot(df['epoch'], df['mAP50-95'], label='mAP50-95', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].set_title('Mean Average Precision')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
    for loss_col in loss_cols:
        axes[0, 1].plot(df['epoch'], df[loss_col], label=loss_col, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision/Recall
    if 'precision' in df.columns:
        axes[1, 0].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    if 'recall' in df.columns:
        axes[1, 0].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in df.columns:
        axes[1, 1].plot(df['epoch'], df['lr'], color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {plot_path}")


def save_training_config(config: Dict[str, Any], output_path: Path):
    """Save training configuration"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Training config saved to {output_path}")


def load_training_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def compute_class_weights(dataset_yaml: Path) -> np.ndarray:
    """Compute class weights for imbalanced dataset"""
    # This would require loading the dataset and counting instances
    # For now, return uniform weights
    # TODO: Implement actual class weight computation from annotations
    
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    nc = data.get('nc', 41)
    return np.ones(nc)


def print_training_summary(results_dir: Path):
    """Print summary of training results"""
    results_file = results_dir / "results.csv"
    
    if not results_file.exists():
        print("⚠️ No results file found")
        return
    
    import pandas as pd
    df = pd.read_csv(results_file)
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    if len(df) > 0:
        last_row = df.iloc[-1]
        
        print(f"\n📊 Final Metrics (Epoch {len(df)}):")
        for col in df.columns:
            if col != 'epoch':
                value = last_row[col]
                print(f"   {col}: {value:.4f}" if isinstance(value, float) else f"   {col}: {value}")
        
        # Best metrics
        print(f"\n🏆 Best Metrics:")
        for col in df.columns:
            if col != 'epoch' and df[col].dtype in [np.float32, np.float64]:
                best_val = df[col].max()
                best_epoch = df[df[col] == best_val]['epoch'].values[0]
                print(f"   {col}: {best_val:.4f} (Epoch {best_epoch})")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Training utilities module loaded successfully!")
