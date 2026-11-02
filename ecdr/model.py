"""model.py — Theoretical Conv1D-ReLU-GAP-Affine CNN model (K=12).

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 3–7).

Architecture:
  Input: (6, 128)
  Conv1: valid (padding=0), kernel 5, stride 1, dilation 1, 6 -> 32  => (32, 124)
  ReLU1
  Conv2: valid (padding=0), kernel 3, stride 1, dilation 1, 32 -> 64 => (64, 122)
  ReLU2
  Conv3: valid (padding=0), kernel 3, stride 1, dilation 1, 64 -> 64 => (64, 120)
  ReLU3
  GAP: global average pool over time dimension (120)                => (64,)
  Head: affine map 64 -> K (default K=12)                           => (K,)

Parameters: P(K) = 19552 + 65*K  (For K=12: 20,332)
MAC count:  M(K) = 2343168 + 64*K (For K=12: 2,343,936)
Receptive fields: Conv1=5, Conv2=7, Conv3=9, GAP/Head=128
"""

import hashlib
import json
import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from ecdr.schema import HAPT_12_CLASSES


def compute_layer_specs(num_classes: int = 12, input_length: int = 128) -> Dict[str, dict]:
    """Compute exact layer shapes, parameters, MACs, and receptive fields.

    Dynamically computes all lengths, parameter counts, and MACs for any input_length >= 9.
    For input_length == 128, results strictly match Table 2 of 02_ECDR_CNN_Model_Analysis_VI.tex.
    """
    if input_length < 9:
        raise ValueError(f"input_length {input_length} is too short; minimum valid receptive field span is 9.")

    t1 = input_length - 5 + 1  # Conv1 valid k=5
    t2 = t1 - 3 + 1            # Conv2 valid k=3
    t3 = t2 - 3 + 1            # Conv3 valid k=3

    conv1_params = 32 * (6 * 5 + 1)  # 992
    conv2_params = 64 * (32 * 3 + 1) # 6208
    conv3_params = 64 * (64 * 3 + 1) # 12352
    head_params = num_classes * (64 + 1) # 65 * K

    conv1_mac = t1 * 32 * 6 * 5
    conv2_mac = t2 * 64 * 32 * 3
    conv3_mac = t3 * 64 * 64 * 3
    head_mac = num_classes * 64

    specs = {
        "input": {"shape": (6, input_length), "rf": 1, "jump": 1, "params": 0, "mac": 0},
        "conv1": {
            "in_c": 6, "out_c": 32, "kernel": 5, "stride": 1, "pad": 0,
            "out_len": t1,
            "rf": 1 + (5 - 1) * 1,            # 5
            "jump": 1,
            "weights": 32 * 6 * 5,            # 960
            "biases": 32,
            "params": conv1_params,           # 992
            "mac": conv1_mac,                 # 119,040 for W=128
        },
        "conv2": {
            "in_c": 32, "out_c": 64, "kernel": 3, "stride": 1, "pad": 0,
            "out_len": t2,
            "rf": 5 + (3 - 1) * 1,            # 7
            "jump": 1,
            "weights": 64 * 32 * 3,           # 6144
            "biases": 64,
            "params": conv2_params,           # 6208
            "mac": conv2_mac,                 # 749,568 for W=128
        },
        "conv3": {
            "in_c": 64, "out_c": 64, "kernel": 3, "stride": 1, "pad": 0,
            "out_len": t3,
            "rf": 7 + (3 - 1) * 1,            # 9
            "jump": 1,
            "weights": 64 * 64 * 3,           # 12288
            "biases": 64,
            "params": conv3_params,           # 12352
            "mac": conv3_mac,                 # 1,474,560 for W=128
        },
        "gap": {
            "out_c": 64,
            "rf": input_length,
            "params": 0,
            "mac": 0,
            "additions": 64 * (t3 - 1),
            "divisions": 64,
        },
        "head": {
            "in_features": 64, "out_features": num_classes,
            "weights": num_classes * 64,
            "biases": num_classes,
            "params": head_params,
            "mac": head_mac,
        },
        "total": {
            "conv_params": conv1_params + conv2_params + conv3_params,  # 19552
            "head_params": head_params,
            "total_params": conv1_params + conv2_params + conv3_params + head_params,  # 20332 for K=12
            "conv_mac": conv1_mac + conv2_mac + conv3_mac,              # 2343168 for W=128
            "head_mac": head_mac,
            "total_mac": conv1_mac + conv2_mac + conv3_mac + head_mac,  # 2343936 for K=12, W=128
        }
    }
    return specs


class ECDRConv1DNet(nn.Module):
    """PyTorch implementation of the theoretical valid Conv1D-ReLU-GAP-Affine CNN.

    Strictly conforms to Section 6 of 02_ECDR_CNN_Model_Analysis_VI.tex:
    - No BatchNorm
    - No MaxPool
    - No Dropout
    - Valid convolutions (zero padding)
    - Stride 1, dilation 1, groups 1
    """

    def __init__(self, num_classes: int = 12):
        super().__init__()
        self.num_classes = num_classes

        # Conv1: 6 -> 32, k=5, valid (out length 128 - 5 + 1 = 124)
        self.conv1 = nn.Conv1d(
            in_channels=6, out_channels=32, kernel_size=5, stride=1, padding=0, bias=True
        )
        self.relu1 = nn.ReLU()

        # Conv2: 32 -> 64, k=3, valid (out length 124 - 3 + 1 = 122)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.relu2 = nn.ReLU()

        # Conv3: 64 -> 64, k=3, valid (out length 122 - 3 + 1 = 120)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.relu3 = nn.ReLU()

        # GAP and Affine Head
        self.head = nn.Linear(in_features=64, out_features=num_classes, bias=True)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return intermediate activation feature maps (after ReLU) for analysis/incremental cache."""
        h1 = self.relu1(self.conv1(x))
        h2 = self.relu2(self.conv2(h1))
        h3 = self.relu3(self.conv3(h2))
        return h1, h2, h3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass taking (B, 6, 128) -> (B, K) logits."""
        _, _, h3 = self.extract_features(x)
        # GAP: average over length dimension (shape: B, 64)
        m = h3.mean(dim=-1)
        # Affine head: (B, K)
        logits = self.head(m)
        return logits


class OperationalSequentialNumpyCNN:
    """Explicit sequential NumPy float64 reference execution without BLAS.

    Vectorizes output channels and time positions simultaneously while retaining
    the explicit sequential term loop over input-channel/kernel taps (no FMA).
    This keeps each output's term order identical, separate multiply/add,
    same proof semantics, and sub-millisecond execution per window.
    Weights are immutable (writeable=False) and tracked via a cryptographic model_id.

    NOTE on Immutability and Lifecycle:
    All weights and parameters are frozen with flags.writeable = False.
    The model_id is a cryptographic binding of architecture, arithmetic semantics,
    class schema (fixed HAPT class ordering for K=12), and parameter values.
    Callers must instantiate a fresh IncrementalEngine or StreamingReplayEngine
    when replacing or updating the frozen model reference.
    """

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
        w_head: np.ndarray, b_head: np.ndarray,
    ):
        """Initialize with float64 weights and biases with strict shape validation."""
        self.w1 = np.copy(np.asarray(w1, dtype=np.float64))
        self.b1 = np.copy(np.asarray(b1, dtype=np.float64))
        self.w2 = np.copy(np.asarray(w2, dtype=np.float64))
        self.b2 = np.copy(np.asarray(b2, dtype=np.float64))
        self.w3 = np.copy(np.asarray(w3, dtype=np.float64))
        self.b3 = np.copy(np.asarray(b3, dtype=np.float64))
        self.w_head = np.copy(np.asarray(w_head, dtype=np.float64))
        self.b_head = np.copy(np.asarray(b_head, dtype=np.float64))

        # Strict validation of fixed architecture shapes
        if self.w1.shape != (32, 6, 5) or self.b1.shape != (32,):
            raise ValueError(f"Malformed Conv1 shapes: w1 {self.w1.shape}, b1 {self.b1.shape}")
        if self.w2.shape != (64, 32, 3) or self.b2.shape != (64,):
            raise ValueError(f"Malformed Conv2 shapes: w2 {self.w2.shape}, b2 {self.b2.shape}")
        if self.w3.shape != (64, 64, 3) or self.b3.shape != (64,):
            raise ValueError(f"Malformed Conv3 shapes: w3 {self.w3.shape}, b3 {self.b3.shape}")
        if self.w_head.ndim != 2 or self.w_head.shape[1] != 64 or self.w_head.shape[0] < 2:
            raise ValueError(f"Malformed head weight shape: {self.w_head.shape}; expected (K, 64) with K>=2")
        if self.b_head.shape != (self.w_head.shape[0],):
            raise ValueError(f"Malformed head bias shape: {self.b_head.shape}; expected ({self.w_head.shape[0]},)")

        self.num_classes = self.w_head.shape[0]

        # Verify finite values and freeze arrays
        hasher = hashlib.sha256()
        hasher.update(b"ECDR-Conv1D-ReLU-GAP-Affine-v1-float64-sequential:")
        hasher.update(b"arithmetic=float64_sequential_no_blas_directed_outward_v1:")
        if self.num_classes == 12:
            schema_str = ",".join(f"{k}:{v}" for k, v in sorted(HAPT_12_CLASSES.items()))
            hasher.update(f"schema=HAPT_12_CLASSES:{schema_str}:".encode("utf-8"))
        else:
            hasher.update(f"schema=generic_{self.num_classes}_classes:".encode("utf-8"))

        for arr, name in [
            (self.w1, "w1"), (self.b1, "b1"),
            (self.w2, "w2"), (self.b2, "b2"),
            (self.w3, "w3"), (self.b3, "b3"),
            (self.w_head, "w_head"), (self.b_head, "b_head")
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite values detected in parameter {name}")
            arr.flags.writeable = False
            hasher.update(arr.tobytes())

        self.model_id = hasher.hexdigest()[:16]

    @classmethod
    def from_torch_module(cls, net: ECDRConv1DNet) -> "OperationalSequentialNumpyCNN":
        """Construct from PyTorch ECDRConv1DNet by extracting parameters into float64."""
        return cls(
            w1=net.conv1.weight.detach().cpu().numpy().astype(np.float64),
            b1=net.conv1.bias.detach().cpu().numpy().astype(np.float64),
            w2=net.conv2.weight.detach().cpu().numpy().astype(np.float64),
            b2=net.conv2.bias.detach().cpu().numpy().astype(np.float64),
            w3=net.conv3.weight.detach().cpu().numpy().astype(np.float64),
            b3=net.conv3.bias.detach().cpu().numpy().astype(np.float64),
            w_head=net.head.weight.detach().cpu().numpy().astype(np.float64),
            b_head=net.head.bias.detach().cpu().numpy().astype(np.float64),
        )

    def conv_relu_sequential(
        self, x: np.ndarray, w: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform vectorized-channel/position Conv1D and ReLU with sequential term reduction.

        x: (C_in, T_in)
        w: (C_out, C_in, k)
        b: (C_out,)
        Returns (pre_activation, post_relu) with shape (C_out, T_out).
        """
        c_out, c_in, k = w.shape
        _, t_in = x.shape
        t_out = t_in - k + 1

        # Initialize accumulator with bias broadcast over time positions
        acc = np.zeros((c_out, t_out), dtype=np.float64)
        acc = np.add(acc, b[:, None])

        # Explicit sequential loop over input channels and kernel taps (no BLAS reduction)
        for c in range(c_in):
            for r in range(k):
                term = np.multiply(w[:, c, r, None], x[c, r : r + t_out][None, :])
                acc = np.add(acc, term)

        pre = acc
        if not np.all(np.isfinite(pre)):
            raise ValueError("Preactivation overflow or non-finite in Conv layer before ReLU")
        post = np.maximum(0.0, acc)
        return pre, post

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass on single window (6, 128) in float64.

        Returns dict containing all intermediate and final outputs:
          'h1', 'h2', 'h3', 'm', 'logits', 'pred_class', 'margins'
        """
        x_f64 = np.asarray(x, dtype=np.float64)
        if x_f64.shape != (6, 128):
            raise ValueError(f"Expected input shape (6, 128), got {x_f64.shape}")
        if not np.all(np.isfinite(x_f64)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        _, h1 = self.conv_relu_sequential(x_f64, self.w1, self.b1)
        if not np.all(np.isfinite(h1)):
            raise ValueError("Intermediate activation overflow/non-finite in Conv1/ReLU1")

        _, h2 = self.conv_relu_sequential(h1, self.w2, self.b2)
        if not np.all(np.isfinite(h2)):
            raise ValueError("Intermediate activation overflow/non-finite in Conv2/ReLU2")

        _, h3 = self.conv_relu_sequential(h2, self.w3, self.b3)
        if not np.all(np.isfinite(h3)):
            raise ValueError("Intermediate activation overflow/non-finite in Conv3/ReLU3")

        # GAP: average over time (120 positions) in canonical order
        d_h, n_pos = h3.shape  # 64, 120
        m = np.zeros(d_h, dtype=np.float64)
        for i in range(n_pos):
            m = np.add(m, h3[:, i])
        m = np.divide(m, float(n_pos))
        if not np.all(np.isfinite(m)):
            raise ValueError("Intermediate activation overflow/non-finite in GAP")

        # Affine head: K classes, sequential term accumulation across 64 features
        logits = np.copy(self.b_head)
        for a in range(d_h):
            term = np.multiply(self.w_head[:, a], m[a])
            logits = np.add(logits, term)

        if not np.all(np.isfinite(logits)):
            raise ValueError("Logits overflow/non-finite before argmax")

        pred_class = int(np.argmax(logits))

        # Compute pairwise margins against winning class: g_cj = F_c - F_j
        margins = np.zeros(self.num_classes, dtype=np.float64)
        for j in range(self.num_classes):
            margins[j] = logits[pred_class] - logits[j]

        return {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "m": m,
            "logits": logits,
            "pred_class": pred_class,
            "margins": margins,
        }
