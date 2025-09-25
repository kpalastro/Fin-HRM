#!/usr/bin/env python3
"""
NIFTY Financial HRM Training Script
Uses cfg_pretrain.yaml configuration and NIFTY_DAILY.csv data
"""

import argparse
import time
import numpy as np
import os
import pickle
import glob
import re
from typing import Tuple, List, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from mlx_adam_atan2_exact import AdamATan2Exact
from dual_optimizer import DualAdamATan2
from lr_scheduler import CosineScheduleWithWarmup
from config_loader import load_config, HRMConfig

# Import financial model and data loader
from models.financial_hrm import FinancialHierarchicalReasoningModel, FinancialHRMCarry
from models.financial_losses import compute_financial_act_loss, compute_simple_regression_loss
from financial_data_loader import load_financial_data, create_financial_batch, denormalize_predictions


class NIFTYFinancialHRMTrainer:
    """Trainer for NIFTY Financial HRM model using pretrain configuration"""

    def __init__(
        self,
        model: FinancialHierarchicalReasoningModel,
        train_data: List[Dict],
        val_data: List[Dict],
        config: HRMConfig,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        # Extract training parameters from config
        self.batch_size = config.batch_size
        self.max_epochs = config.max_epochs
        self.eval_interval = config.eval_interval
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

        # Set embedding learning rate
        self.embedding_lr = config.embedding_lr
        
        # Use dual optimizer if embedding LR is different from base LR
        if abs(config.embedding_lr - config.learning_rate) > 1e-8:
            print("🔧 Using dual optimizer setup (different embedding LR)")
            self.optimizer = DualAdamATan2(
                base_lr=config.learning_rate,
                embedding_lr=config.embedding_lr,
                weight_decay=config.weight_decay,
                embedding_weight_decay=config.embedding_weight_decay,
                betas=(config.beta1, config.beta2),
                a=1.27,
                b=1.0
            )
            self.use_dual_optimizer = True
        else:
            print("🔧 Using single optimizer setup (same LR for all parameters)")
            self.optimizer = AdamATan2Exact(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
                a=1.27,
                b=1.0
            )
            self.use_dual_optimizer = False
        
        # Initialize optimizer with model parameters
        self.optimizer.init(self.model.trainable_parameters())
        
        # Calculate total training steps for LR scheduler
        steps_per_epoch = len(self.train_data) // self.batch_size
        total_steps = self.max_epochs * steps_per_epoch
        
        # Initialize learning rate scheduler
        self.lr_scheduler = CosineScheduleWithWarmup(
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
        
        print(f"📊 Base learning rate: {config.learning_rate}")
        print(f"📊 Embedding learning rate: {self.embedding_lr}")
        print(f"📊 LR scheduler: {config.warmup_steps} warmup steps, {total_steps} total steps")
        print(f"📊 Min LR ratio: {config.min_lr_ratio}")
        
        self.grad_clip_norm = 1.0
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoints directory
        self.checkpoint_dir = "checkpoints_nifty"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.recent_checkpoints = []

    def create_batch(self, indices: np.ndarray) -> Dict[str, mx.array]:
        """Create a batch from indices"""
        batch_samples = [self.train_data[i] for i in indices]
        return create_financial_batch(batch_samples, len(indices))

    def evaluate(self, n_batches: int = 20) -> Dict[str, float]:
        """Evaluate model"""
        total_mse = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        n_samples = 0

        # Set model to eval mode
        self.model.eval()

        for i in range(n_batches):
            # Random validation batch
            val_indices = np.random.choice(
                len(self.val_data),
                size=min(self.batch_size, len(self.val_data)),
                replace=False
            )

            batch_samples = [self.val_data[i] for i in val_indices]
            batch = create_financial_batch(batch_samples, len(val_indices))

            # Initialize carry and run model
            carry = self.model.initial_carry(batch)

            # Run until all sequences halt
            all_halted = False
            step_count = 0
            while not all_halted and step_count < self.model.halt_max_steps:
                carry, outputs = self.model(carry, batch)
                all_halted = carry.halted.all()
                step_count += 1

            # Compute metrics
            loss, metrics = compute_simple_regression_loss(outputs["predictions"], batch["targets"])

            total_mse += metrics["mse_loss"]
            total_mae += metrics["mae_loss"]
            total_r2 += metrics["r2_overall"]
            n_samples += 1

        # Set model back to train mode
        self.model.train()

        return {
            'val_mse': total_mse / n_samples,
            'val_mae': total_mae / n_samples,
            'val_r2': total_r2 / n_samples,
        }

    def train(self):
        """Main training loop"""
        print(f"🚀 Starting NIFTY Financial HRM Training...")
        print(f"📊 Training samples: {len(self.train_data):,}")
        print(f"📊 Validation samples: {len(self.val_data):,}")
        print(f"🔧 Model: {self.model.inner.d_model}d, {self.model.inner.H_cycles}×{self.model.inner.L_cycles} reasoning")
        print(f"🎯 Task: Predict next high and low values for NIFTY")
        print(f"⚙️  Config: {self.config.d_model}d, {self.config.H_cycles}×{self.config.L_cycles} cycles")
        
        # Gradient accumulation settings
        grad_accum_steps = getattr(self, 'gradient_accumulation_steps', 1)
        effective_batch_size = self.batch_size * grad_accum_steps
        print(f"🔄 Gradient accumulation: {grad_accum_steps} steps (effective batch size: {effective_batch_size})")

        # Calculate total steps for progress tracking
        steps_per_epoch = len(self.train_data) // effective_batch_size
        total_steps = self.max_epochs * steps_per_epoch

        # Initialize accumulated gradients
        accumulated_grads = None

        for epoch in range(self.max_epochs):
            # Shuffle training data
            train_indices = np.random.permutation(len(self.train_data))
            n_batches = len(train_indices) // self.batch_size

            for batch_idx in range(n_batches):
                batch_start = time.time()

                # Create batch
                batch_indices = train_indices[
                    batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size
                ]
                batch = self.create_batch(batch_indices)

                # Forward pass with ACT
                def loss_fn(model):
                    carry = model.initial_carry(batch)

                    total_loss = mx.array(0.0)
                    step_count = 0
                    last_metrics = None

                    # Unroll ACT loop
                    for step_count in range(model.halt_max_steps):
                        carry, outputs = model(carry, batch)
                        loss, metrics = compute_financial_act_loss(outputs, batch["targets"])
                        total_loss = total_loss + loss
                        last_metrics = metrics

                        # Early stopping if all halted
                        if carry.halted.all():
                            break

                    # Scale loss by accumulation steps
                    return total_loss / grad_accum_steps, (step_count + 1, last_metrics)

                loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
                (loss, (step_count, metrics)), grads = loss_and_grad_fn(self.model)

                # Gradient clipping
                def clip_grads(grads, max_norm=1.0):
                    total_norm = 0.0

                    def compute_norm(g):
                        nonlocal total_norm
                        if g is not None:
                            total_norm += mx.sum(g ** 2)
                        return g

                    nn.utils.tree_map(compute_norm, grads)
                    total_norm = mx.sqrt(total_norm)

                    clip_coef = max_norm / (total_norm + 1e-6)
                    clip_coef = mx.minimum(clip_coef, 1.0)

                    def clip_grad(g):
                        return g * clip_coef if g is not None else g

                    return nn.utils.tree_map(clip_grad, grads)

                grads = clip_grads(grads, self.grad_clip_norm)

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    def add_grads(acc_g, new_g):
                        if acc_g is None or new_g is None:
                            return new_g if acc_g is None else acc_g
                        return acc_g + new_g
                    
                    accumulated_grads = nn.utils.tree_map(add_grads, accumulated_grads, grads)

                # Check if we should update parameters
                is_accumulation_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = batch_idx == n_batches - 1
                
                if is_accumulation_step or is_last_batch:
                    # Update learning rate
                    if self.use_dual_optimizer:
                        current_lr = self.lr_scheduler.get_lr(self.step)
                        self.optimizer.update_learning_rate(current_lr)
                    else:
                        current_lr = self.lr_scheduler.update_optimizer_lr(self.optimizer, self.step)

                    # Update with accumulated gradients
                    self.optimizer.update(self.model, accumulated_grads)
                    mx.eval(self.model.parameters(), self.optimizer.state)
                    
                    # Reset accumulated gradients
                    accumulated_grads = None

                # Memory cleanup
                if self.step % 50 == 0:
                    mx.eval(mx.zeros(1))
                    import gc
                    gc.collect()

                batch_time = time.time() - batch_start
                samples_per_sec = self.batch_size / batch_time

                # Logging
                if self.step % 100 == 0:
                    progress_pct = (self.step / total_steps) * 100
                    print(f"Step {self.step:6d}/{total_steps} ({progress_pct:5.1f}%) | "
                          f"Epoch {epoch+1:2d} | "
                          f"Loss: {float(loss):.4f} | "
                          f"MSE: {float(metrics['mse_loss']):.4f} | "
                          f"R²: {float(metrics['r2_overall']):.3f} | "
                          f"LR: {float(self.optimizer.learning_rate):.2e} | "
                          f"Speed: {samples_per_sec:.0f} smp/s | "
                          f"Steps: {step_count:.1f}")

                # Periodic checkpoint saving
                if self.step % 500 == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.npz", cleanup_old=True)

                # Evaluation
                if self.step % self.eval_interval == 0 and self.step > 0:
                    val_metrics = self.evaluate()

                    print(f"\n{'='*60}")
                    print(f"📈 VALIDATION (Step {self.step})")
                    print(f"{'='*60}")
                    print(f"Val MSE: {val_metrics['val_mse']:.4f}")
                    print(f"Val MAE: {val_metrics['val_mae']:.4f}")
                    print(f"Val R²:  {val_metrics['val_r2']:.3f}")
                    print(f"{'='*60}\n")

                    if val_metrics['val_mse'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_mse']
                        # Save best model
                        self.save_checkpoint(f"best_model_step_{self.step}.npz", is_best=True)

                    # Save regular checkpoint
                    self.save_checkpoint(f"checkpoint_step_{self.step}.npz")

                self.step += 1

        # Save final checkpoint
        self.save_checkpoint("final_model.npz")

    def save_checkpoint(self, filename: str, is_best: bool = False, cleanup_old: bool = False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        try:
            self.model.save_weights(checkpoint_path)
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return

        # Save training state
        state_path = checkpoint_path.replace('.npz', '_state.pkl')
        try:
            training_state = {
                'step': self.step,
                'best_val_loss': self.best_val_loss,
                'config': self.config.to_dict(),
            }

            with open(state_path, 'wb') as f:
                pickle.dump(training_state, f)
        except Exception as e:
            print(f"⚠️  Warning: Could not save training state: {e}")

        # Cleanup old checkpoints
        if cleanup_old and not is_best:
            self.recent_checkpoints.append((checkpoint_path, state_path))

            while len(self.recent_checkpoints) > 3:
                old_checkpoint, old_state = self.recent_checkpoints.pop(0)
                try:
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                    if os.path.exists(old_state):
                        os.remove(old_state)
                except OSError:
                    pass

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            self.model.load_weights(checkpoint_path)

            state_path = checkpoint_path.replace('.npz', '_state.pkl')
            if os.path.exists(state_path):
                with open(state_path, 'rb') as f:
                    training_state = pickle.load(f)

                self.step = training_state['step']
                self.best_val_loss = training_state['best_val_loss']

                print(f"📂 Loaded checkpoint: {checkpoint_path}")
                print(f"   Step: {self.step}, Best Val Loss: {self.best_val_loss:.4f}")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Continuing with fresh model...")


def main():
    parser = argparse.ArgumentParser(description="NIFTY Financial HRM Training")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="config/cfg_pretrain.yaml", help="Path to YAML configuration file")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/NIFTY_DAILY.csv", help="Path to NIFTY data CSV")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to load (None for all)")
    parser.add_argument("--sequence_length", type=int, default=10, help="Input sequence length")
    parser.add_argument("--target_horizon", type=int, default=1, help="Days ahead to predict")
    
    # Override config parameters
    parser.add_argument("--d_model", type=int, help="Model dimension (override config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (override config)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (override config)")
    parser.add_argument("--max_epochs", type=int, help="Maximum epochs (override config)")
    parser.add_argument("--eval_interval", type=int, help="Evaluation interval (override config)")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_nifty", help="Checkpoint directory")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic checkpoint resuming")

    args = parser.parse_args()

    print("📈 NIFTY Financial HRM Training (MLX Implementation)")
    print("=" * 60)
    print("🎯 Task: Predict next high and low values for NIFTY")
    print("🔧 Using pretrain configuration")
    print()

    # Load configuration
    config = load_config(args.config, args)
    
    print("📋 Configuration:")
    for key, value in config.to_dict().items():
        print(f"   {key}: {value}")
    print()

    # Load NIFTY data
    print("📊 Loading NIFTY data...")
    train_data, val_data = load_financial_data(
        args.data_path, 
        max_samples=args.max_samples or config.train_samples,
        sequence_length=args.sequence_length,
        target_horizon=args.target_horizon
    )

    print(f"✅ Train samples: {len(train_data):,}")
    print(f"✅ Val samples: {len(val_data):,}")
    print()

    # Create model
    print("🤖 Creating NIFTY Financial HRM model...")
    model = FinancialHierarchicalReasoningModel(
        n_features=12,  # Number of financial features
        d_model=config.d_model,
        n_heads=8,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        H_layers=config.H_layers,
        L_layers=config.L_layers,
        halt_max_steps=config.halt_max_steps,
        halt_exploration_prob=config.halt_exploration_prob,
        seq_len=args.sequence_length,
    )

    # Count parameters
    total_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"✅ Model parameters: {total_params:,}")
    print(f"✅ Architecture: {config.H_cycles}×{config.L_cycles} cycles, {config.H_layers}+{config.L_layers} layers")
    print()

    # Create trainer
    print("🏋️ Creating Trainer...")
    trainer = NIFTYFinancialHRMTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
    )

    # Set checkpoint directory
    trainer.checkpoint_dir = args.checkpoint_dir
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    print()
    print("=" * 60)
    print("🚀 STARTING NIFTY FINANCIAL HRM TRAINING")
    print("=" * 60)

    # Train
    trainer.train()

    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Best Validation Loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
