"""Certified adaptive reuse for the K=12 ECDR reference model.

This module is deliberately separate from ``ecdr.replay``.  The existing replay
engine implements exact interval certificates; this module adds a conservative
margin/Lipschitz policy for experiments without changing that contract.

For an anchor ``a`` with winning class ``c`` and competitor ``j``, let
``g_cj(x) = F_c(x) - F_j(x)``.  The global bound used here is

    |g_cj(x) - g_cj(a)| <= L_cj * ||x-a||_1

where ``L_cj`` is obtained from induced L1 operator norms of the three valid
convolutions, GAP, and the logit-margin head.  Since the adaptive distance is
``mean(abs(x-a)) = ||x-a||_1 / 768``, a sufficient certified radius is

    tau_c(a) < min_j g_cj(a) / (768 * L_cj).

The bound is global and intentionally conservative.  It is a formal decision
invariance condition for this frozen NumPy reference model, not an accuracy
claim about the ground-truth labels.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ecdr.model import OperationalSequentialNumpyCNN


INPUT_SIZE = 6 * 128


def conv1d_l1_operator_norm(weights: np.ndarray) -> float:
    """Return a safe induced L1 norm for a valid Conv1D operator.

    A convolution column is one input channel/time coordinate.  Its absolute
    output-column sum is bounded by the sum of absolute taps over all output
    channels.  Taking the maximum over input channels is safe for every valid
    temporal position, including boundary positions.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 3 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite with shape (C_out, C_in, kernel)")
    # Sum over output channels and kernel taps, then maximize the input channel.
    column_sums = np.sum(np.abs(w), axis=(0, 2), dtype=np.float64)
    # The norm is used as an upper bound, so round it outward.
    column_sums = np.nextafter(column_sums, np.inf)
    return float(np.max(column_sums))


def margin_lipschitz_bounds(net: OperationalSequentialNumpyCNN) -> np.ndarray:
    """Return global L1 Lipschitz bounds for every logit margin.

    The returned matrix has shape (K, K).  Entry [c, j] bounds the margin
    ``F_c - F_j``; the diagonal is zero by convention.
    """
    conv_norm = (
        conv1d_l1_operator_norm(net.w1)
        * conv1d_l1_operator_norm(net.w2)
        * conv1d_l1_operator_norm(net.w3)
    )
    head = np.asarray(net.w_head, dtype=np.float64)
    if head.ndim != 2 or head.shape[0] != net.num_classes or head.shape[1] != 64:
        raise ValueError("net.w_head has an incompatible shape")
    bounds = np.zeros((net.num_classes, net.num_classes), dtype=np.float64)
    for c in range(net.num_classes):
        for j in range(net.num_classes):
            if c != j:
                # GAP has induced L1 norm 1.  A scalar linear functional has
                # L1 operator norm ||coefficient||_inf.
                head_diff = np.nextafter(np.abs(head[c] - head[j]), np.inf)
                head_norm = float(np.nextafter(np.max(head_diff), np.inf))
                product = np.float64(conv_norm) * np.float64(head_norm)
                bounds[c, j] = float(np.nextafter(product, np.inf))
    if not np.all(np.isfinite(bounds)):
        raise ValueError("non-finite Lipschitz bound")
    return bounds


def certified_radius_from_logits(
    logits: np.ndarray,
    target_class: int,
    lipschitz_bounds: np.ndarray,
    input_size: int = INPUT_SIZE,
) -> float:
    """Compute a conservative mean-absolute-distance radius for one anchor."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    bounds = np.asarray(lipschitz_bounds, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite vector with at least two classes")
    if bounds.shape != (len(values), len(values)) or not np.all(np.isfinite(bounds)):
        raise ValueError("lipschitz_bounds must be a finite K x K matrix")
    if not (0 <= int(target_class) < len(values)):
        raise ValueError("target_class is outside the logits vector")
    if input_size <= 0:
        raise ValueError("input_size must be positive")

    c = int(target_class)
    # The decision margin is used as a lower bound, hence directed rounding.
    margins = np.nextafter(values[c] - values, -np.inf)
    competitors = np.array([j for j in range(len(values)) if j != c], dtype=np.int64)
    if not np.all(margins[competitors] > 0.0):
        return 0.0
    denominators = float(input_size) * bounds[c, competitors]
    if np.any(denominators <= 0.0):
        # A zero Lipschitz bound means that margin is constant under this
        # global bound.  It imposes no radius restriction.
        restricted = margins[competitors][denominators > 0.0] / denominators[denominators > 0.0]
        if len(restricted) == 0:
            return float("inf")
    else:
        restricted = margins[competitors] / denominators
    radius = float(np.min(restricted))
    # Strict inequality avoids accepting the floating-point boundary itself.
    return float(np.nextafter(radius, 0.0)) if np.isfinite(radius) else radius


@dataclass(frozen=True)
class CertifiedAnchor:
    """Immutable state used by the certified adaptive cache."""

    window: np.ndarray
    target_class: int
    certified_radius: float
    logits: np.ndarray

    def __post_init__(self) -> None:
        window = np.array(self.window, dtype=np.float64, copy=True)
        logits = np.array(self.logits, dtype=np.float64, copy=True)
        window.flags.writeable = False
        logits.flags.writeable = False
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "logits", logits)


class CertifiedAdaptiveCache:
    """Per-recording certified adaptive cache over the frozen reference model."""

    def __init__(self, net: OperationalSequentialNumpyCNN):
        self.net = net
        self.lipschitz_bounds = margin_lipschitz_bounds(net)
        self.anchor: Optional[CertifiedAnchor] = None
        self.current_recording_id: Optional[str] = None
        self.last_step_idx: Optional[int] = None

    def reset(self, recording_id: str) -> None:
        self.anchor = None
        self.current_recording_id = str(recording_id)
        self.last_step_idx = None

    def _ensure_lifecycle(self, recording_id: str, step_idx: int) -> None:
        recording_id = str(recording_id)
        if self.current_recording_id != recording_id:
            self.reset(recording_id)
        elif self.last_step_idx is not None and int(step_idx) != self.last_step_idx + 1:
            self.reset(recording_id)
        self.last_step_idx = int(step_idx)

    def predict(
        self,
        window: np.ndarray,
        recording_id: str,
        step_idx: int,
    ) -> Dict[str, object]:
        """Return a certified reuse or a full reference prediction.

        On a miss, the current reference result becomes the next anchor.  A hit
        never renews the anchor, so lifecycle and proof state remain explicit.
        """
        self._ensure_lifecycle(recording_id, step_idx)
        query = np.asarray(window, dtype=np.float64)
        if query.shape != (6, 128) or not np.all(np.isfinite(query)):
            raise ValueError("window must be finite with shape (6, 128)")

        if self.anchor is not None:
            distance = float(np.mean(np.abs(query - self.anchor.window)))
            if distance <= self.anchor.certified_radius:
                return {
                    "pred_class": self.anchor.target_class,
                    "is_hit": True,
                    "distance": distance,
                    "radius": self.anchor.certified_radius,
                    "reason": "certified_margin_lipschitz_membership",
                }

        result = self.net.forward(query)
        target_class = int(result["pred_class"])
        radius = certified_radius_from_logits(
            logits=result["logits"],
            target_class=target_class,
            lipschitz_bounds=self.lipschitz_bounds,
        )
        self.anchor = CertifiedAnchor(
            window=query,
            target_class=target_class,
            certified_radius=radius,
            logits=result["logits"],
        )
        return {
            "pred_class": target_class,
            "is_hit": False,
            "distance": None,
            "radius": radius,
            "reason": "reference_fallback_and_anchor_refresh",
        }

