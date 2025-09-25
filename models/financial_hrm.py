"""
Financial HRM Model for Regression Tasks
Modified HRM model to predict next high and low values from financial time series
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn

from .hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_Inner, 
    HRMInnerCarry, 
    HRMCarry,
    HRMTransformerBlock,
    HRMReasoningModule
)
from .common import trunc_normal_init_
from .layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding,
    CastedEmbedding, CastedLinear
)


@dataclass
class FinancialHRMCarry:
    """Carry state for Financial HRM"""
    inner_carry: HRMInnerCarry
    steps: mx.array
    halted: mx.array
    current_data: Dict[str, mx.array]


class FinancialHierarchicalReasoningModel_Inner(nn.Module):
    """Inner Financial HRM model for regression"""
    
    def __init__(
        self,
        n_features: int = 12,  # Number of financial features
        d_model: int = 512,
        n_heads: int = 8,
        H_cycles: int = 2,
        L_cycles: int = 2,
        H_layers: int = 4,
        L_layers: int = 4,
        expansion: float = 2.0,
        rms_norm_eps: float = 1e-5,
        seq_len: int = 10,  # Sequence length for financial data
        pos_encodings: str = "learned",
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.seq_len = seq_len
        self.pos_encodings = pos_encodings
        
        # Input projection from features to model dimension
        self.input_projection = CastedLinear(n_features, d_model, bias=True)
        
        # Output heads for regression (predict high and low)
        self.high_head = CastedLinear(d_model, 1, bias=True)
        self.low_head = CastedLinear(d_model, 1, bias=True)
        
        # Q-head for ACT (halt/continue decision)
        self.q_head = CastedLinear(d_model, 2, bias=True)
        
        # Position encodings
        if self.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.d_model // n_heads,
                max_position_embeddings=self.seq_len,
                base=rope_theta
            )
        elif self.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.seq_len, self.d_model, init_std=1.0/math.sqrt(d_model), cast_to=mx.float32)
        else:
            raise NotImplementedError()
        
        # Reasoning modules
        self.H_level = HRMReasoningModule(H_layers, d_model, n_heads, expansion)
        self.L_level = HRMReasoningModule(L_layers, d_model, n_heads, expansion)
        
        # Initial states
        self.H_init = trunc_normal_init_(shape=(d_model,), std=1.0)
        self.L_init = trunc_normal_init_(shape=(d_model,), std=1.0)
        
        # Initialize Q-head
        self._init_q_head()
    
    def _init_q_head(self):
        """Initialize Q-head for faster learning during bootstrapping"""
        self.q_head.weight = mx.zeros_like(self.q_head.weight)
        self.q_head.bias = mx.array([-5.0, -5.0])
    
    def _input_embeddings(self, features: mx.array) -> mx.array:
        """Convert input features to embeddings"""
        # features shape: (batch_size, seq_len, n_features)
        batch_size, seq_len, n_features = features.shape
        
        # Project features to model dimension
        # Reshape to (batch_size * seq_len, n_features) for linear layer
        features_flat = features.reshape(-1, n_features)
        embeddings = self.input_projection(features_flat)
        
        # Reshape back to (batch_size, seq_len, d_model)
        embeddings = embeddings.reshape(batch_size, seq_len, self.d_model)
        
        # Add position embeddings
        if self.pos_encodings == "learned":
            pos_emb = self.embed_pos.embedding_weight.astype(mx.float32)
            embeddings = embeddings + pos_emb[None, :, :]  # Broadcast over batch
        
        return embeddings
    
    def empty_carry(self, batch_size: int) -> HRMInnerCarry:
        """Create empty inner carry"""
        return HRMInnerCarry(
            z_H=mx.zeros((batch_size, self.seq_len, self.d_model)),
            z_L=mx.zeros((batch_size, self.seq_len, self.d_model))
        )
    
    def reset_carry(self, reset_flag: mx.array, carry: HRMInnerCarry) -> HRMInnerCarry:
        """Reset carry state"""
        return HRMInnerCarry(
            z_H=mx.where(reset_flag[..., None, None], self.H_init, carry.z_H),
            z_L=mx.where(reset_flag[..., None, None], self.L_init, carry.z_L)
        )
    
    def __call__(self, carry: HRMInnerCarry, batch: Dict[str, mx.array]) -> Tuple[HRMInnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass"""
        
        # Get sequence info for RoPE
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb()
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["features"])
        
        # Forward iterations (same as original HRM)
        z_H, z_L = carry.z_H, carry.z_L
        
        for _H_step in range(self.H_cycles):
            for _L_step in range(self.L_cycles):
                if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                    z_L = mx.stop_gradient(self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin))
            
            if not (_H_step == self.H_cycles - 1):
                z_H = mx.stop_gradient(self.H_level(z_H, z_L, cos_sin=cos_sin))
        
        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)
        
        # Regression outputs - use mean pooling over sequence
        pooled_hidden = mx.mean(z_H, axis=1)  # (batch_size, d_model)
        
        high_pred = self.high_head(pooled_hidden)  # (batch_size, 1)
        low_pred = self.low_head(pooled_hidden)    # (batch_size, 1)
        
        # Combine predictions
        predictions = mx.concatenate([high_pred, low_pred], axis=1)  # (batch_size, 2)
        
        # Q-head for ACT
        q_logits = self.q_head(pooled_hidden).astype(mx.float32)
        
        # Update carry
        new_carry = HRMInnerCarry(
            z_H=mx.stop_gradient(z_H), 
            z_L=mx.stop_gradient(z_L)
        )
        
        return new_carry, predictions, (q_logits[..., 0], q_logits[..., 1])


class FinancialHierarchicalReasoningModel(nn.Module):
    """Complete Financial HRM with ACT wrapper"""
    
    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        H_cycles: int = 2,
        L_cycles: int = 2,
        H_layers: int = 4,
        L_layers: int = 4,
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,
        expansion: float = 2.0,
        seq_len: int = 10,
        pos_encodings: str = "learned",
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self._is_training = True
        
        self.inner = FinancialHierarchicalReasoningModel_Inner(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            H_layers=H_layers,
            L_layers=L_layers,
            expansion=expansion,
            seq_len=seq_len,
            pos_encodings=pos_encodings,
            rope_theta=rope_theta,
        )
    
    def train(self):
        """Set model to training mode"""
        self._is_training = True
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self._is_training = False
        return self
    
    def initial_carry(self, batch: Dict[str, mx.array]) -> FinancialHRMCarry:
        """Initialize carry state"""
        batch_size = batch["features"].shape[0]
        
        # Only include array data in current_data, skip metadata like batch_size
        current_data = {}
        for k, v in batch.items():
            if isinstance(v, mx.array):
                current_data[k] = mx.zeros_like(v)
        
        return FinancialHRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),
            current_data=current_data
        )
    
    def __call__(self, carry: FinancialHRMCarry, batch: Dict[str, mx.array]) -> Tuple[FinancialHRMCarry, Dict[str, mx.array]]:
        """Forward pass with ACT"""
        
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = mx.where(carry.halted, 0, carry.steps)
        
        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry.current_data.items():
            halted_shape = (-1,) + (1,) * (batch[k].ndim - 1)
            halted_expanded = carry.halted.reshape(halted_shape)
            new_current_data[k] = mx.where(halted_expanded, batch[k], v)
        
        # Forward inner model
        new_inner_carry, predictions, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "predictions": predictions,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Step and halting logic
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.halt_max_steps
        
        halted = is_last_step
        
        # Training mode ACT logic
        if self._is_training and (self.halt_max_steps > 1):
            # Halt signal
            halted = halted | (q_halt_logits > q_continue_logits)
            
            # Exploration
            exploration_mask = mx.random.uniform(shape=q_halt_logits.shape) < self.halt_exploration_prob
            random_steps = mx.random.randint(2, self.halt_max_steps + 1, shape=new_steps.shape)
            min_halt_steps = exploration_mask.astype(mx.int32) * random_steps
            
            halted = halted & (new_steps >= min_halt_steps)
            
            # Compute target Q
            next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
            
            outputs["target_q_continue"] = mx.sigmoid(mx.where(is_last_step, next_q_halt_logits, mx.maximum(next_q_halt_logits, next_q_continue_logits)))
        
        return FinancialHRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
