"""incremental.py — Exact incremental shift/cache and computational reuse engine.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 7, 8, 11).

Key theorems and identities:
1. Theorem 7.1 (Exact aligned feature equality):
   H_i(Z') = H_{i+s}(Z) for 0 <= i < n - s (with n = 120, s = 64 => 56 shared positions).
2. Eq 8.4 (GAP Delta Identity):
   m(Z') - m(Z) = (D_new(U) - D_drop) / n
   where D_drop = sum_{i=0}^{s-1} H_i(Z) and D_new(U) = sum_{i=n-s}^{n-1} H_i(Z').
3. Bitwise-exact Canonical Reduction:
   Combines shifted shared positions H3[:, s:] with newly computed tail positions,
   and evaluates GAP in identical canonical order (0..119).
4. Strict anti-aliasing:
   Internal cache state stores immutable/detached copies, and returned dictionaries
   return detached copies so caller modifications cannot corrupt cache state.
5. Exact overlap validation:
   Requires exact np.array_equal on shared window segment.
6. Execution mode reporting:
   Explicitly reports mode ("full" vs "incremental") and exact MAC count.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np

from ecdr.model import OperationalSequentialNumpyCNN


@dataclass
class IncrementalState:
    """Stores intermediate layer activations from previous window for incremental reuse."""
    z: np.ndarray      # Normalized window (6, 128)
    h1: np.ndarray     # Conv1/ReLU1 features (32, 124)
    h2: np.ndarray     # Conv2/ReLU2 features (64, 122)
    h3: np.ndarray     # Conv3/ReLU3 features (64, 120)
    m: np.ndarray      # GAP vector (64,)
    logits: np.ndarray # Head logits (K,)
    model_id: str      # Invariant model identifier
    step_idx: int = 0


class IncrementalEngine:
    """Incremental forward execution engine maintaining per-layer activation cache."""

    def __init__(self, net: OperationalSequentialNumpyCNN, shift: int = 64):
        self.net = net
        self.shift = shift
        self.n_pos = 120  # final feature length
        assert 1 <= shift <= self.n_pos, f"Shift {shift} must be between 1 and {self.n_pos}"
        self.state: Optional[IncrementalState] = None

    def reset(self) -> None:
        """Clear cache (e.g. at recording boundary or cache invalidation)."""
        self.state = None

    def forward_full(self, z: np.ndarray, step_idx: int = 0) -> Dict[str, Union[np.ndarray, int, float, str]]:
        """Perform full forward pass and populate incremental cache."""
        z_f64 = np.asarray(z, dtype=np.float64)
        if z_f64.shape != (6, 128):
            raise ValueError(f"Expected input shape (6, 128), got {z_f64.shape}")
        if not np.all(np.isfinite(z_f64)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        res = self.net.forward(z_f64)
        self.state = IncrementalState(
            z=np.copy(z_f64),
            h1=np.copy(res["h1"]),
            h2=np.copy(res["h2"]),
            h3=np.copy(res["h3"]),
            m=np.copy(res["m"]),
            logits=np.copy(res["logits"]),
            model_id=self.net.model_id,
            step_idx=step_idx,
        )

        return {
            "h1": np.copy(res["h1"]),
            "h2": np.copy(res["h2"]),
            "h3": np.copy(res["h3"]),
            "m": np.copy(res["m"]),
            "logits": np.copy(res["logits"]),
            "pred_class": int(res["pred_class"]),
            "mode": "full",
            "mac_count": 2343936 if self.net.num_classes == 12 else (2343168 + 64 * self.net.num_classes),
            "d_drop": np.zeros(64, dtype=np.float64),
            "d_new": np.zeros(64, dtype=np.float64),
            "m_delta": np.copy(res["m"]),
            "gap_discrepancy": 0.0,
        }

    def forward_incremental(
        self, z_new: np.ndarray, step_idx: int = 0
    ) -> Dict[str, Union[np.ndarray, int, float, str]]:
        """Execute incremental computation for window z_new shifted by s from previous z.

        If cache is empty, model changed, or exact bitwise overlap is violated,
        falls back to full forward pass safely.
        """
        z_f64 = np.asarray(z_new, dtype=np.float64)
        if z_f64.shape != (6, 128):
            raise ValueError(f"Expected input shape (6, 128), got {z_f64.shape}")
        if not np.all(np.isfinite(z_f64)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        s = self.shift

        # Check if cache is valid and model_id matches
        if self.state is None or self.state.model_id != self.net.model_id:
            return self.forward_full(z_f64, step_idx=step_idx)

        # Exact bitwise equality check on overlap segment: z_new[:, :128-s] == z_old[:, s:]
        overlap_len = 128 - s
        overlap_new = z_f64[:, :overlap_len]
        overlap_old = self.state.z[:, s:]

        if not np.array_equal(overlap_new, overlap_old):
            # Overlap mismatch, perturbation, or discontinuity -> fallback to full forward
            return self.forward_full(z_f64, step_idx=step_idx)

        w1, b1 = self.net.w1, self.net.b1
        w2, b2 = self.net.w2, self.net.b2
        w3, b3 = self.net.w3, self.net.b3

        # Layer 1
        shared_h1 = self.state.h1[:, s:]  # (32, 60)
        tail_in_1 = z_f64[:, (124 - s):]  # (6, 68)
        _, tail_h1 = self.net.conv_relu_sequential(tail_in_1, w1, b1)
        if not np.all(np.isfinite(tail_h1)):
            return self.forward_full(z_f64, step_idx=step_idx)
        h1_new = np.concatenate([shared_h1, tail_h1], axis=1)  # (32, 124)

        # Layer 2
        shared_h2 = self.state.h2[:, s:]  # (64, 58)
        tail_in_2 = h1_new[:, (122 - s):]  # (32, 66)
        _, tail_h2 = self.net.conv_relu_sequential(tail_in_2, w2, b2)
        if not np.all(np.isfinite(tail_h2)):
            return self.forward_full(z_f64, step_idx=step_idx)
        h2_new = np.concatenate([shared_h2, tail_h2], axis=1)  # (64, 122)

        # Layer 3
        shared_h3 = self.state.h3[:, s:]  # (64, 56)
        tail_in_3 = h2_new[:, (120 - s):]  # (64, 66)
        _, tail_h3 = self.net.conv_relu_sequential(tail_in_3, w3, b3)
        if not np.all(np.isfinite(tail_h3)):
            return self.forward_full(z_f64, step_idx=step_idx)
        h3_new = np.concatenate([shared_h3, tail_h3], axis=1)  # (64, 120)

        # Canonical GAP reduction
        d_h, n_pos = h3_new.shape
        m_canonical = np.zeros(d_h, dtype=np.float64)
        for i in range(n_pos):
            m_canonical = np.add(m_canonical, h3_new[:, i])
        m_canonical = np.divide(m_canonical, float(n_pos))
        if not np.all(np.isfinite(m_canonical)):
            return self.forward_full(z_f64, step_idx=step_idx)

        # Head evaluation
        logits_canonical = np.copy(self.net.b_head)
        for a in range(d_h):
            term = np.multiply(self.net.w_head[:, a], m_canonical[a])
            logits_canonical = np.add(logits_canonical, term)

        if not np.all(np.isfinite(logits_canonical)):
            return self.forward_full(z_f64, step_idx=step_idx)

        pred_class = int(np.argmax(logits_canonical))

        # Diagnostic GAP delta computation
        d_drop = np.sum(self.state.h3[:, :s], axis=1)
        d_new = np.sum(tail_h3, axis=1)
        m_delta = self.state.m + (d_new - d_drop) / float(n_pos)

        # Update cache state with owned copies
        self.state = IncrementalState(
            z=np.copy(z_f64),
            h1=np.copy(h1_new),
            h2=np.copy(h2_new),
            h3=np.copy(h3_new),
            m=np.copy(m_canonical),
            logits=np.copy(logits_canonical),
            model_id=self.net.model_id,
            step_idx=step_idx,
        )

        mac_incremental = 64 * 32 * 30 + 64 * 64 * 96 + 64 * 64 * 192 + self.net.num_classes * 64

        return {
            "h1": np.copy(h1_new),
            "h2": np.copy(h2_new),
            "h3": np.copy(h3_new),
            "m": np.copy(m_canonical),
            "logits": np.copy(logits_canonical),
            "pred_class": pred_class,
            "mode": "incremental",
            "mac_count": mac_incremental,
            "d_drop": d_drop,
            "d_new": d_new,
            "m_delta": m_delta,
            "gap_discrepancy": float(np.max(np.abs(m_canonical - m_delta))),
        }
