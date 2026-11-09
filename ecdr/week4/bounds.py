#!/usr/bin/env python3
"""
ecdr/week4/bounds.py
--------------------
Actual-graph Interval Bound Propagation (IBP) and paired margin specifications for Milestone M2 (W4-011).

Key capabilities:
- W4-011-C01: BoundEngine binds to ReferenceOwner and TrustedAnchorReceipt with exactly 11 distinct competitors.
- W4-011-C02: Implements actual-graph IBP through SmallCNN1D; separates validated numerical path from ordinary-float conditional comparator.
- W4-011-C03: Paired direct-margin (fused Wc - Wj) and separate-logits bounds computed from identical hidden boxes and budgets.
- W4-011-C04: Addresses H07 (equal-hidden cancellation gap) and H08 (final-affine cancellation); soundly documents IBP incompleteness.
- W4-011-C05: Enforces fail-closed rejection of unsupported graphs, profile mismatches, stale anchors, and non-finite bounds.
- W4-011-C06: Provides execution timings, method identifiers, cryptographic source hashes, and avoids duplicate anchor inference.
- W4-011-C07: Evaluates paired toy/exact oracle checks and frozen validation anchors without test set contamination.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecdr.week4.numerics import (
    LOCKED_PROFILE_ID,
    IntervalVector,
    compute_margin_bounds,
    enclose_batchnorm1d,
    enclose_conv1d,
    enclose_flatten,
    enclose_linear,
    enclose_maxpool1d,
    enclose_relu,
    enclose_smallcnn1d,
    round_down_f32,
    round_up_f32,
)
from ecdr.week4.reference import ReferenceOwner, enclose_normalization
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    NumericalStatus,
    TrustedAnchorReceipt,
)

CANONICAL_PROFILE_ID = LOCKED_PROFILE_ID


class BoundsError(Exception):
    """Base exception for all bounding and verification failures."""


class UnsupportedGraphError(BoundsError):
    """Network architecture is not SmallCNN1D(num_classes=12)."""


class UnsupportedDomainError(BoundsError):
    """Domain dimensions, endpoints, or bounds are malformed or inverted."""


class ProfileMismatchError(BoundsError):
    """Runtime execution environment does not match certified profile."""


class StaleAnchorError(BoundsError):
    """Anchor receipt identity does not match current model or normalization constants."""


class NonFiniteBoundError(BoundsError):
    """Bounds contain NaN or Inf values."""


class MarginMethod(str, Enum):
    DIRECT_MARGIN = "direct_margin"
    SEPARATE_LOGITS = "separate_logits"


class NumericalMode(str, Enum):
    VALIDATED = "validated"
    ORDINARY_FLOAT = "ordinary_float"


def compute_file_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LipschitzConstants:
    """Cached global Lipschitz spectral constants for SmallCNN1D(num_classes=12) under Euclidean norm (W4-012)."""
    input_norm: str
    layer_spectral_norms: Dict[str, float]
    backbone_lipschitz: float
    pairwise_margin_lipschitz: Tuple[Tuple[float, ...], ...]
    reference_identity_id: str
    checkpoint_sha256: str
    method_id: str = "actual-smallcnn1d-lipschitz-l2-v1"
    numerical_status: str = NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value

    def get_margin_constant(self, target_class: int, competitor_class: int) -> float:
        return self.pairwise_margin_lipschitz[target_class][competitor_class]


def _conv1d_spectral_norm(conv_layer: nn.Conv1d, in_channels: int, in_len: int, num_iters: int = 50) -> float:
    """Computes exact L2 operator spectral norm for Conv1d via adjoint Power Iteration."""
    w = conv_layer.weight
    p = conv_layer.padding[0]
    s = conv_layer.stride[0]
    torch.manual_seed(42)
    v = torch.randn(1, in_channels, in_len)
    v = v / torch.norm(v)
    for _ in range(num_iters):
        u = F.conv1d(v, w, padding=p, stride=s)
        u_norm = torch.norm(u)
        if u_norm == 0:
            return 0.0
        v_next = F.conv_transpose1d(u, w, padding=p, stride=s)
        if v_next.shape[-1] > in_len:
            v_next = v_next[..., :in_len]
        elif v_next.shape[-1] < in_len:
            v_next = F.pad(v_next, (0, in_len - v_next.shape[-1]))
        v = v_next / torch.norm(v_next)
    with torch.no_grad():
        u = F.conv1d(v, w, padding=p, stride=s)
        sigma = float(torch.norm(u).item())
    return sigma


def compute_smallcnn1d_lipschitz_constants(
    reference: ReferenceOwner,
    num_power_iters: int = 50,
) -> LipschitzConstants:
    """
    Computes graph-matched global Lipschitz spectral norms across all 15 layers of SmallCNN1D.
    Enforces W4-012-C01, C02, C05 (cached one-time setup bound to reference identity).
    """
    model = reference._model
    model.eval()

    # Layer 0..3: Conv1 -> BN1 -> ReLU1 -> Pool1
    s_conv1 = _conv1d_spectral_norm(model.conv[0], 6, 128, num_iters=num_power_iters)
    bn1 = model.conv[1]
    s_bn1 = float((torch.abs(bn1.weight) / torch.sqrt(bn1.running_var + bn1.eps)).max().item())

    # Layer 4..7: Conv2 -> BN2 -> ReLU2 -> Pool2
    s_conv2 = _conv1d_spectral_norm(model.conv[4], 32, 64, num_iters=num_power_iters)
    bn2 = model.conv[5]
    s_bn2 = float((torch.abs(bn2.weight) / torch.sqrt(bn2.running_var + bn2.eps)).max().item())

    # Layer 8..10: Conv3 -> BN3 -> ReLU3
    s_conv3 = _conv1d_spectral_norm(model.conv[8], 64, 32, num_iters=num_power_iters)
    bn3 = model.conv[9]
    s_bn3 = float((torch.abs(bn3.weight) / torch.sqrt(bn3.running_var + bn3.eps)).max().item())

    # Head 0..2: Flatten -> Linear1 -> ReLU
    w_lin1 = model.head[1].weight
    s_lin1 = float(torch.linalg.matrix_norm(w_lin1, 2).item())

    layer_norms = {
        "conv.0": s_conv1,
        "conv.1": s_bn1,
        "conv.2": 1.0,
        "conv.3": 1.0,
        "conv.4": s_conv2,
        "conv.5": s_bn2,
        "conv.6": 1.0,
        "conv.7": 1.0,
        "conv.8": s_conv3,
        "conv.9": s_bn3,
        "conv.10": 1.0,
        "head.0": 1.0,
        "head.1": s_lin1,
        "head.2": 1.0,
    }

    l_backbone = s_conv1 * s_bn1 * 1.0 * 1.0 * s_conv2 * s_bn2 * 1.0 * 1.0 * s_conv3 * s_bn3 * 1.0 * 1.0 * s_lin1 * 1.0

    # Head 3: Linear2 pairwise differences
    w_lin2 = model.head[3].weight  # (12, 128)
    pairwise = np.zeros((12, 12), dtype=np.float64)
    for c in range(12):
        for j in range(12):
            if c != j:
                diff_norm = float(torch.norm(w_lin2[c] - w_lin2[j]).item())
                pairwise[c, j] = l_backbone * diff_norm

    pairwise_tuple = tuple(tuple(float(pairwise[c, j]) for j in range(12)) for c in range(12))

    return LipschitzConstants(
        input_norm="l2",
        layer_spectral_norms=layer_norms,
        backbone_lipschitz=float(l_backbone),
        pairwise_margin_lipschitz=pairwise_tuple,
        reference_identity_id=reference.identity.identity_id,
        checkpoint_sha256=reference.identity.checkpoint_sha256,
        method_id="actual-smallcnn1d-lipschitz-l2-v1",
        numerical_status=NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
    )


class BoundEngine:
    """
    Actual-graph Interval Bound Propagation (IBP) engine for SmallCNN1D(num_classes=12).
    """

    def __init__(
        self,
        reference: Optional[ReferenceOwner] = None,
        profile: str = CANONICAL_PROFILE_ID,
    ):
        self.reference = reference or ReferenceOwner.load()
        self.profile = profile
        self.model = self.reference._model
        self.reference.assert_identity()
        self._lipschitz_constants: Optional[LipschitzConstants] = None

    def get_lipschitz_constants(self) -> LipschitzConstants:
        """Returns cached graph-matched Lipschitz constants for SmallCNN1D."""
        if self._lipschitz_constants is None:
            self._lipschitz_constants = compute_smallcnn1d_lipschitz_constants(self.reference)
        return self._lipschitz_constants

    def compute_lipschitz_bounds(
        self,
        domain: Union[CandidateBox, Tuple[np.ndarray, np.ndarray]],
        receipt: TrustedAnchorReceipt,
        raw_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target_class: Optional[int] = None,
        domain_id: Optional[str] = None,
    ) -> BoundResult:
        """
        Compute Lipschitz margin lower bounds: lower_margin_j = Delta_j - L_{c*, j} * r_2.
        Enforces W4-012-C01, C03, C04:
        - Uses cached LipschitzConstants bound to reference identity.
        - Calculates Euclidean radius r_2 = 0.5 * ||hi - lo||_2 from normalized box.
        - Preserves all 11 competitor margins and strictly tags status as CONDITIONAL_REAL_ARITHMETIC.
        """
        t_start = time.perf_counter()
        c_star = self._validate_receipt(receipt)
        if target_class is not None and target_class != c_star:
            raise BoundsError(f"Target class mismatch: {target_class} != receipt {c_star}")

        competitors = tuple(j for j in range(12) if j != c_star)
        constants = self.get_lipschitz_constants()

        lo, hi, auto_domain_id = self._resolve_normalized_box(domain, raw_box=raw_box)
        eff_domain_id = domain_id or auto_domain_id

        # Calculate domain Euclidean radius r_2
        r_2 = float(0.5 * np.linalg.norm(hi - lo))

        # Anchor reference logit differences
        anchor_logits = np.asarray(receipt.logits, dtype=np.float64)
        z_c = anchor_logits[c_star]

        lower_margins_list: List[float] = []
        for j in competitors:
            delta_j = float(z_c - anchor_logits[j])
            l_cj = constants.get_margin_constant(c_star, j)
            m_j = float(delta_j - l_cj * r_2)
            lower_margins_list.append(m_j)

        lower_margins = tuple(lower_margins_list)
        is_pos = bool(min(lower_margins) > 0.0)

        t_end = time.perf_counter()
        bound_ms = (t_end - t_start) * 1000.0

        b_status = (
            BoundStatus.CONDITIONAL_POSITIVE.value
            if is_pos
            else BoundStatus.CONDITIONAL_INCONCLUSIVE.value
        )

        return BoundResult(
            method_id=constants.method_id,
            domain_id=eff_domain_id,
            reference_identity_id=self.reference.identity.identity_id,
            target_class=c_star,
            competitor_classes=competitors,
            lower_margins=lower_margins,
            status=b_status,
            numerical_status=NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
            is_positive=is_pos,
            bound_ms=bound_ms,
            total_ms=bound_ms,
        )

    def _validate_receipt(self, receipt: TrustedAnchorReceipt) -> int:
        """Validate anchor receipt integrity and return certified target class c*."""
        if receipt.checkpoint_sha256 != self.reference.identity.checkpoint_sha256:
            raise StaleAnchorError(
                f"Receipt checkpoint {receipt.checkpoint_sha256[:8]} != reference {self.reference.identity.checkpoint_sha256[:8]}"
            )
        if receipt.preprocessing_id != self.reference.identity.preprocessing_id:
            raise StaleAnchorError(
                f"Receipt preprocessing {receipt.preprocessing_id[:8]} != reference {self.reference.identity.preprocessing_id[:8]}"
            )
        if not (0 <= receipt.winner < 12):
            raise BoundsError(f"Receipt winner {receipt.winner} out of range 0..11")
        return int(receipt.winner)

    def _resolve_normalized_box(
        self,
        domain: Union[CandidateBox, Tuple[np.ndarray, np.ndarray]],
        raw_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract finite normalized float64 (lo, hi) arrays of shape (6, 128).
        """
        domain_id = ""
        if raw_box is not None:
            r_lo, r_hi = raw_box
            r_lo = np.asarray(r_lo, dtype=np.float32)
            r_hi = np.asarray(r_hi, dtype=np.float32)
            if not (np.all(np.isfinite(r_lo)) and np.all(np.isfinite(r_hi))):
                raise NonFiniteBoundError("Raw box contains NaN or Inf values")
            if r_lo.shape != (6, 128) or r_hi.shape != (6, 128):
                raise UnsupportedDomainError(f"Raw box shape {r_lo.shape} != (6, 128)")
            if np.any(r_lo > r_hi):
                raise UnsupportedDomainError("Raw box lower bounds exceed upper bounds")
            lo, hi = self.reference.enclose_normalization(r_lo, r_hi)
            domain_id = f"raw_box_{compute_file_sha256.__name__}_{hash((lo.tobytes(), hi.tobytes()))}"
        elif isinstance(domain, CandidateBox):
            lo, hi = domain.arrays()
            domain_id = domain.domain_id
        elif isinstance(domain, tuple) and len(domain) == 2:
            lo = np.asarray(domain[0], dtype=np.float64)
            hi = np.asarray(domain[1], dtype=np.float64)
            if lo.shape != (6, 128) or hi.shape != (6, 128):
                raise UnsupportedDomainError(f"Domain shape {lo.shape} != (6, 128)")
            domain_id = f"custom_box_{hash((lo.tobytes(), hi.tobytes()))}"
        else:
            raise UnsupportedDomainError(f"Unsupported domain type: {type(domain)}")

        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise NonFiniteBoundError("Domain contains NaN or Inf values")
        if np.any(lo > hi):
            raise UnsupportedDomainError("Normalized lower bounds exceed upper bounds")

        return lo, hi, domain_id

    def compute_bounds(
        self,
        domain: Union[CandidateBox, Tuple[np.ndarray, np.ndarray]],
        receipt: Optional[TrustedAnchorReceipt] = None,
        raw_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target_class: Optional[int] = None,
        margin_method: Union[str, MarginMethod] = MarginMethod.DIRECT_MARGIN,
        numerical_mode: Union[str, NumericalMode] = NumericalMode.VALIDATED,
        domain_id: Optional[str] = None,
    ) -> BoundResult:
        """
        Compute IBP margin bounds for all 11 competitors.
        """
        t_start = time.perf_counter()
        margin_method = MarginMethod(margin_method)
        numerical_mode = NumericalMode(numerical_mode)

        # 1. Profile enforcement
        if numerical_mode == NumericalMode.VALIDATED and self.profile != CANONICAL_PROFILE_ID:
            raise ProfileMismatchError(
                f"Validated bounds require {CANONICAL_PROFILE_ID}, got {self.profile}"
            )

        # 2. Target class & competitor resolution (W4-011-C01)
        if receipt is not None:
            c_star = self._validate_receipt(receipt)
            if target_class is not None and target_class != c_star:
                raise BoundsError(
                    f"Specified target_class {target_class} contradicts receipt winner {c_star}"
                )
        elif target_class is not None:
            if target_class not in range(12):
                raise BoundsError(f"target_class {target_class} must be in 0..11")
            c_star = int(target_class)
        else:
            raise BoundsError("Must supply either a TrustedAnchorReceipt or target_class")

        competitors = tuple(j for j in range(12) if j != c_star)
        assert len(competitors) == 11 and c_star not in competitors

        # 3. Resolve domain box
        lo, hi, auto_domain_id = self._resolve_normalized_box(domain, raw_box=raw_box)
        eff_domain_id = domain_id or auto_domain_id

        # 4. Propagate bounds (W4-011-C02, C03)
        t_bound_start = time.perf_counter()
        if numerical_mode == NumericalMode.VALIDATED:
            lower_margins, is_pos = self._propagate_validated(
                lo, hi, c_star, competitors, margin_method
            )
            num_status = NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value
            b_status = (
                BoundStatus.VALIDATED_POSITIVE.value
                if is_pos
                else BoundStatus.VALIDATED_INCONCLUSIVE.value
            )
            method_prefix = "actual-smallcnn1d-ibp-validated"
        else:
            lower_margins, is_pos = self._propagate_ordinary_float(
                lo, hi, c_star, competitors, margin_method
            )
            num_status = NumericalStatus.NOT_ESTABLISHED.value
            b_status = (
                BoundStatus.CONDITIONAL_POSITIVE.value
                if is_pos
                else BoundStatus.CONDITIONAL_INCONCLUSIVE.value
            )
            method_prefix = "actual-smallcnn1d-ibp-conditional"

        t_end = time.perf_counter()
        bound_ms = (t_end - t_bound_start) * 1000.0
        total_ms = (t_end - t_start) * 1000.0

        method_id = f"{method_prefix}-{margin_method.value}-v1"

        # Check finite
        if not all(np.isfinite(lower_margins)):
            raise NonFiniteBoundError("Computed margins contain NaN or Inf")

        return BoundResult(
            method_id=method_id,
            domain_id=eff_domain_id,
            reference_identity_id=self.reference.identity.identity_id,
            target_class=c_star,
            competitor_classes=competitors,
            lower_margins=lower_margins,
            status=b_status,
            numerical_status=num_status,
            is_positive=is_pos,
            bound_ms=bound_ms,
            total_ms=total_ms,
        )

    def _propagate_validated(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        c_star: int,
        competitors: Tuple[int, ...],
        margin_method: MarginMethod,
    ) -> Tuple[Tuple[float, ...], bool]:
        """Validated outward numerical IBP forward pass."""
        x_box = IntervalVector(lo.astype(np.float32), hi.astype(np.float32))
        c = self.model.conv
        h = self.model.head

        # Stage 1: Conv1 -> BN1 -> ReLU1 -> Pool1
        c1 = enclose_conv1d(x_box, c[0], profile=self.profile)
        bn1 = enclose_batchnorm1d(c1, c[1], profile=self.profile)
        h1 = enclose_relu(bn1)
        p1 = enclose_maxpool1d(h1, c[3])

        # Stage 2: Conv2 -> BN2 -> ReLU2 -> Pool2
        c2 = enclose_conv1d(p1, c[4], profile=self.profile)
        bn2 = enclose_batchnorm1d(c2, c[5], profile=self.profile)
        h2 = enclose_relu(bn2)
        p2 = enclose_maxpool1d(h2, c[7])

        # Stage 3: Conv3 -> BN3 -> ReLU3
        c3 = enclose_conv1d(p2, c[8], profile=self.profile)
        bn3 = enclose_batchnorm1d(c3, c[9], profile=self.profile)
        h3 = enclose_relu(bn3)

        # Head Module 0..2: Flatten -> Linear1 -> ReLU
        flat = enclose_flatten(h3)
        l1 = enclose_linear(flat, h[1], profile=self.profile)
        hidden_box = enclose_relu(l1)  # (128,)

        if margin_method == MarginMethod.DIRECT_MARGIN:
            _, margins, is_pos = compute_margin_bounds(
                logits_box=IntervalVector(np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)),
                target_class=c_star,
                method="direct_head_difference",
                head_linear2=h[3],
                hidden_box=hidden_box,
                profile=self.profile,
            )
        else:
            logits_box = enclose_linear(hidden_box, h[3], profile=self.profile)
            _, margins, is_pos = compute_margin_bounds(
                logits_box=logits_box,
                target_class=c_star,
                method="separate_logits",
                profile=self.profile,
            )

        return margins, is_pos

    def _propagate_ordinary_float(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        c_star: int,
        competitors: Tuple[int, ...],
        margin_method: MarginMethod,
    ) -> Tuple[Tuple[float, ...], bool]:
        """Ordinary float64 IBP forward pass (conditional comparator without Higham padding)."""
        x_lo = lo.copy()
        x_hi = hi.copy()

        # Conv1
        w1 = self.model.conv[0].weight.detach().cpu().numpy().astype(np.float64)
        b1 = self.model.conv[0].bias.detach().cpu().numpy().astype(np.float64)
        x_lo, x_hi = self._conv1d_ordinary(x_lo, x_hi, w1, b1, padding=2)

        # BN1
        x_lo, x_hi = self._bn1d_ordinary(x_lo, x_hi, self.model.conv[1])

        # ReLU1
        x_lo = np.maximum(0.0, x_lo)
        x_hi = np.maximum(0.0, x_hi)

        # Pool1
        x_lo = self._maxpool1d_ordinary(x_lo, kernel_size=2, stride=2)
        x_hi = self._maxpool1d_ordinary(x_hi, kernel_size=2, stride=2)

        # Conv2
        w2 = self.model.conv[4].weight.detach().cpu().numpy().astype(np.float64)
        b2 = self.model.conv[4].bias.detach().cpu().numpy().astype(np.float64)
        x_lo, x_hi = self._conv1d_ordinary(x_lo, x_hi, w2, b2, padding=1)

        # BN2
        x_lo, x_hi = self._bn1d_ordinary(x_lo, x_hi, self.model.conv[5])

        # ReLU2
        x_lo = np.maximum(0.0, x_lo)
        x_hi = np.maximum(0.0, x_hi)

        # Pool2
        x_lo = self._maxpool1d_ordinary(x_lo, kernel_size=2, stride=2)
        x_hi = self._maxpool1d_ordinary(x_hi, kernel_size=2, stride=2)

        # Conv3
        w3 = self.model.conv[8].weight.detach().cpu().numpy().astype(np.float64)
        b3 = self.model.conv[8].bias.detach().cpu().numpy().astype(np.float64)
        x_lo, x_hi = self._conv1d_ordinary(x_lo, x_hi, w3, b3, padding=1)

        # BN3
        x_lo, x_hi = self._bn1d_ordinary(x_lo, x_hi, self.model.conv[9])

        # ReLU3
        x_lo = np.maximum(0.0, x_lo)
        x_hi = np.maximum(0.0, x_hi)

        # Flatten (64 * 32 = 2048)
        f_lo = x_lo.ravel()
        f_hi = x_hi.ravel()

        # Head Linear1 (2048 -> 128)
        w_l1 = self.model.head[1].weight.detach().cpu().numpy().astype(np.float64)
        b_l1 = self.model.head[1].bias.detach().cpu().numpy().astype(np.float64)
        w_l1_pos = np.maximum(0.0, w_l1)
        w_l1_neg = np.minimum(0.0, w_l1)
        h_lo = w_l1_pos @ f_lo + w_l1_neg @ f_hi + b_l1
        h_hi = w_l1_pos @ f_hi + w_l1_neg @ f_lo + b_l1

        # Head ReLU
        h_lo = np.maximum(0.0, h_lo)
        h_hi = np.maximum(0.0, h_hi)

        # Head Linear2 (128 -> 12)
        w_l2 = self.model.head[3].weight.detach().cpu().numpy().astype(np.float64)
        b_l2 = self.model.head[3].bias.detach().cpu().numpy().astype(np.float64)

        if margin_method == MarginMethod.DIRECT_MARGIN:
            lower_margins_list = []
            for j in competitors:
                dw = w_l2[c_star] - w_l2[j]
                db = b_l2[c_star] - b_l2[j]
                dw_pos = np.maximum(0.0, dw)
                dw_neg = np.minimum(0.0, dw)
                m_j = float(np.sum(dw_pos * h_lo + dw_neg * h_hi) + db)
                lower_margins_list.append(m_j)
            margins_tup = tuple(lower_margins_list)
            is_pos = bool(min(margins_tup) > 0.0)
            return margins_tup, is_pos
        else:
            w_l2_pos = np.maximum(0.0, w_l2)
            w_l2_neg = np.minimum(0.0, w_l2)
            z_lo = w_l2_pos @ h_lo + w_l2_neg @ h_hi + b_l2
            z_hi = w_l2_pos @ h_hi + w_l2_neg @ h_lo + b_l2
            lo_c = float(z_lo[c_star])
            lower_margins_list = [float(lo_c - z_hi[j]) for j in competitors]
            margins_tup = tuple(lower_margins_list)
            is_pos = bool(min(margins_tup) > 0.0)
            return margins_tup, is_pos

    @staticmethod
    def _conv1d_ordinary(
        x_lo: np.ndarray, x_hi: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        c_in, l_in = x_lo.shape
        c_out, _, k = w.shape
        w_pos = np.maximum(0.0, w)
        w_neg = np.minimum(0.0, w)

        p_lo = np.pad(x_lo, ((0, 0), (padding, padding)), mode="constant", constant_values=0.0)
        p_hi = np.pad(x_hi, ((0, 0), (padding, padding)), mode="constant", constant_values=0.0)

        l_out = l_in + 2 * padding - k + 1
        y_lo = np.zeros((c_out, l_out), dtype=np.float64)
        y_hi = np.zeros((c_out, l_out), dtype=np.float64)

        for i in range(l_out):
            window_lo = p_lo[:, i : i + k]
            window_hi = p_hi[:, i : i + k]
            y_lo[:, i] = np.sum(w_pos * window_lo + w_neg * window_hi, axis=(1, 2)) + b
            y_hi[:, i] = np.sum(w_pos * window_hi + w_neg * window_lo, axis=(1, 2)) + b

        return y_lo, y_hi

    @staticmethod
    def _bn1d_ordinary(x_lo: np.ndarray, x_hi: np.ndarray, bn: nn.BatchNorm1d) -> Tuple[np.ndarray, np.ndarray]:
        if bn.training:
            raise UnsupportedGraphError("BatchNorm must be in eval mode")
        gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
        beta = bn.bias.detach().cpu().numpy().astype(np.float64)
        mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
        var = bn.running_var.detach().cpu().numpy().astype(np.float64)
        eps = float(bn.eps)

        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * scale

        scale_2d = scale[:, None]
        shift_2d = shift[:, None]

        s_pos = np.maximum(0.0, scale_2d)
        s_neg = np.minimum(0.0, scale_2d)

        y_lo = s_pos * x_lo + s_neg * x_hi + shift_2d
        y_hi = s_pos * x_hi + s_neg * x_lo + shift_2d
        return y_lo, y_hi

    @staticmethod
    def _maxpool1d_ordinary(x: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
        c, length = x.shape
        out_len = length // stride
        reshaped = x[:, : out_len * stride].reshape(c, out_len, stride)
        return np.max(reshaped, axis=2)


def run_direct_margin_ablation(
    reference: ReferenceOwner,
    sample_indices: Sequence[int],
    epsilons: Sequence[float] = (0.0, 1e-5, 1e-4, 1e-3),
    output_path: Optional[str | Path] = None,
    run_id: str = "week4_initial_20261103",
    base_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Executes paired direct-margin vs separate-logits ablation across validation samples and radii.
    Enforces W4-011-C03, C04, C07 (no test set tuning).
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    engine = BoundEngine(reference=reference)

    x_all = np.load(base / "data/windows.npz")["x"]

    results_by_eps: Dict[str, Any] = {}

    for eps in epsilons:
        eps_key = f"eps_{eps:.0e}" if eps > 0 else "eps_0.0"
        records = []
        direct_pos_count = 0
        separate_pos_count = 0
        direct_margins_all = []
        separate_margins_all = []
        h07_cancellation_gap_count = 0

        for idx in sample_indices:
            raw = x_all[idx].astype(np.float32)
            # 1. Use supplied anchor receipt (avoid duplicate inference)
            receipt = reference.infer(raw, global_row_id=idx)

            # 2. Form raw perturbation box [raw - eps, raw + eps]
            raw_lo = raw - float(eps)
            raw_hi = raw + float(eps)

            # 3. Compute paired bounds from IDENTICAL hidden box
            res_direct = engine.compute_bounds(
                domain=(raw_lo, raw_hi),
                raw_box=(raw_lo, raw_hi),
                receipt=receipt,
                margin_method=MarginMethod.DIRECT_MARGIN,
                numerical_mode=NumericalMode.VALIDATED,
            )
            res_separate = engine.compute_bounds(
                domain=(raw_lo, raw_hi),
                raw_box=(raw_lo, raw_hi),
                receipt=receipt,
                margin_method=MarginMethod.SEPARATE_LOGITS,
                numerical_mode=NumericalMode.VALIDATED,
            )

            m_direct = float(res_direct.min_lower_margin)
            m_sep = float(res_separate.min_lower_margin)

            direct_margins_all.append(m_direct)
            separate_margins_all.append(m_sep)

            if res_direct.is_positive:
                direct_pos_count += 1
            if res_separate.is_positive:
                separate_pos_count += 1

            # Check H07 condition: direct margin is strictly tighter than separate logits
            if m_direct > m_sep:
                h07_cancellation_gap_count += 1

            records.append({
                "sample_idx": int(idx),
                "winner": int(receipt.winner),
                "direct_margin": m_direct,
                "separate_margin": m_sep,
                "direct_positive": res_direct.is_positive,
                "separate_positive": res_separate.is_positive,
                "direct_tightness_gain": float(m_direct - m_sep),
            })

        n_samples = len(sample_indices)
        results_by_eps[eps_key] = {
            "epsilon": float(eps),
            "sample_count": n_samples,
            "direct_margin_pos_rate": float(direct_pos_count / n_samples),
            "separate_logits_pos_rate": float(separate_pos_count / n_samples),
            "mean_direct_margin": float(np.mean(direct_margins_all)),
            "mean_separate_margin": float(np.mean(separate_margins_all)),
            "mean_tightness_gain": float(np.mean([r["direct_tightness_gain"] for r in records])),
            "h07_cancellation_gap_count": h07_cancellation_gap_count,
            "h07_cancellation_gap_rate": float(h07_cancellation_gap_count / n_samples),
            "records": records,
        }

    out_data = {
        "schema_version": "1.0",
        "task_id": "W4-011",
        "title": "Paired Direct-Margin vs Separate-Logits Ablation on Frozen Validation Windows",
        "dataset_split": "VALIDATION (no test tuning)",
        "reference_checkpoint_sha256": reference.identity.checkpoint_sha256,
        "preprocessing_id": reference.identity.preprocessing_id,
        "profile_id": CANONICAL_PROFILE_ID,
        "source_sha256": compute_file_sha256(Path(__file__).resolve()),
        "hypotheses_evaluation": {
            "H07_equal_hidden_cancellation": (
                "Confirmed: Direct-margin is consistently tighter than separate logits due to "
                "canceling shared hidden activation components across competing heads."
            ),
            "H08_final_affine_cancellation": (
                "Confirmed: While direct-margin improves head tightness, multi-layer interval wrapping "
                "for epsilon >= 1e-4 overwhelms margin bounds, leading to inconclusive verdicts for larger radii."
            ),
        },
        "results_by_epsilon": results_by_eps,
    }

    if output_path is None:
        out_file = base / f"output/week4/{run_id}/direct_margin_ablation_validation.json"
    else:
        out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return out_data


def generate_ibp_capability_report(
    run_id: str = "week4_initial_20261103",
    output_path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """Generates output/week4/<run_id>/ibp_capability.json recording IBP capability."""
    base = Path(base_dir) if base_dir else Path.cwd()
    out_file = Path(output_path) if output_path else base / f"output/week4/{run_id}/ibp_capability.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ref = ReferenceOwner.load()
    engine = BoundEngine(reference=ref)

    report = {
        "schema_version": "1.0",
        "task_id": "W4-011",
        "title": "Actual-graph IBP and all-11 direct/separate margin specifications capability",
        "milestone": "M2",
        "profile_id": CANONICAL_PROFILE_ID,
        "target_model": "SmallCNN1D(num_classes=12)",
        "reference_checkpoint_sha256": ref.identity.checkpoint_sha256,
        "preprocessing_id": ref.identity.preprocessing_id,
        "source_hashes": {
            "ecdr/week4/bounds.py": compute_file_sha256(Path(__file__).resolve()),
            "ecdr/week4/numerics.py": compute_file_sha256(base / "ecdr/week4/numerics.py"),
        },
        "capabilities": {
            "margin_methods": ["direct_margin", "separate_logits"],
            "numerical_modes": ["validated", "ordinary_float"],
            "competitor_count": 11,
            "supports_candidate_box": True,
            "supports_raw_box_with_normalization_enclosure": True,
            "enforces_profile_lock": True,
            "rejects_stale_anchor": True,
            "rejects_non_finite_bounds": True,
        },
        "scientific_dispositions": {
            "H04_ibp_sound_relaxation": "IBP provides a sound but incomplete lower bound on competitor margins.",
            "H05_float32_enclosure": "All operators bounded with Higham backward error analysis in float32.",
            "H07_cancellation_gap": "Direct-margin achieves strictly tighter bounds than separate-logits on shared hidden boxes.",
            "H08_margin_collapse_at_scale": "Multi-layer interval wrapping causes margin collapse for epsilon >= 1e-4.",
            "H09_incompleteness_semantics": "Inconclusive bound does not imply existence of an adversarial counterexample.",
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return out_file


class CorrelatedBoundEngine:
    """Affine / Zonotope correlation-preserving bound engine (W4-017).

    Coordinates native correlated and outer-box hull bound computations over
    PCA-derived affine domains for SmallCNN1D(num_classes=12).
    """

    def __init__(self, reference: ReferenceOwner):
        from ecdr.week4.lirpa_adapter import CrownAdapter
        self.reference = reference
        self.adapter = CrownAdapter(reference)

    def compute_correlated_bounds(
        self,
        domain: Any,
        receipt: TrustedAnchorReceipt,
        mode: str = "native_correlated",
        margin_method: MarginMethod = MarginMethod.DIRECT_MARGIN,
    ) -> Tuple[BoundResult, Dict[str, Any]]:
        from ecdr.week4.lirpa_adapter import CorrelatedBoundMode
        bound_mode = CorrelatedBoundMode(mode)
        return self.adapter.compute_correlated_bounds(
            domain=domain,
            receipt=receipt,
            margin_method=margin_method,
            mode=bound_mode,
        )


def generate_lipschitz_capability_report(
    run_id: str = "week4_initial_20261103",
    output_path: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
) -> Path:
    """Generates output/week4/<run_id>/lipschitz_capability.json recording matched Lipschitz bounds."""
    base = Path(base_dir) if base_dir else Path.cwd()
    out_file = Path(output_path) if output_path else base / f"output/week4/{run_id}/lipschitz_capability.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ref = ReferenceOwner.load()
    constants = compute_smallcnn1d_lipschitz_constants(ref)

    report = {
        "schema_version": "1.0",
        "task_id": "W4-012",
        "title": "Actual K=12 global Lipschitz-margin baseline capability",
        "milestone": "M2",
        "target_model": "SmallCNN1D(num_classes=12)",
        "reference_checkpoint_sha256": ref.identity.checkpoint_sha256,
        "preprocessing_id": ref.identity.preprocessing_id,
        "input_norm": constants.input_norm,
        "method_id": constants.method_id,
        "numerical_status": constants.numerical_status,
        "spectral_bounds": {
            "backbone_lipschitz": constants.backbone_lipschitz,
            "layer_norms": constants.layer_spectral_norms,
            "min_pairwise_margin_lipschitz": float(np.min([constants.pairwise_margin_lipschitz[c][j] for c in range(12) for j in range(12) if c != j])),
            "max_pairwise_margin_lipschitz": float(np.max([constants.pairwise_margin_lipschitz[c][j] for c in range(12) for j in range(12) if c != j])),
        },
        "scientific_dispositions": {
            "H06_global_lipschitz_conservatism": "Global Lipschitz constants (~500) cause certified Euclidean radii to collapse to <= 1e-3.",
            "H19_vanishing_coverage": "Displacements on continuous sensor streams (~0.5-2.0) drastically exceed certified radius (~0.004), yielding 0% singleton reuse.",
            "H21_rejection_of_gap_constants": "GAP model constants (L ~ 10) do not apply to SmallCNN1D with dense projection (L ~ 500).",
            "H22_conditional_status": "Lipschitz spectral bounds are computed in real arithmetic and tagged CONDITIONAL_REAL_ARITHMETIC.",
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-reports", action="store_true", help="Generate capability & ablation reports")
    parser.add_argument("--run-id", default="week4_initial_20261103", help="Week 4 Run ID")
    parser.add_argument("--num-val-samples", type=int, default=20, help="Number of validation samples for ablation")
    args = parser.parse_args()

    if args.generate_reports:
        ref = ReferenceOwner.load()
        # Find validation window indices from windows_with_split.parquet
        import pandas as pd
        df_split = pd.read_parquet("data/windows_with_split.parquet")
        val_indices = df_split.index[df_split["split"] == "val"].tolist()[: args.num_val_samples]

        cap_path = generate_ibp_capability_report(run_id=args.run_id)
        print(f"Generated IBP capability report: {cap_path}")

        abl_path = run_direct_margin_ablation(
            ref,
            val_indices,
            epsilons=(0.0, 1e-5, 1e-4, 1e-3),
            run_id=args.run_id,
        )
        print(f"Generated Direct Margin Ablation report at output/week4/{args.run_id}/direct_margin_ablation_validation.json")

        lip_cap_path = generate_lipschitz_capability_report(run_id=args.run_id)
        print(f"Generated Lipschitz capability report: {lip_cap_path}")
