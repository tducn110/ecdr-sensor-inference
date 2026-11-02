"""bounds.py — Whole-domain wave and diagonal margin bounds for all K-1 competitors.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 8, 9, 11).

Provides:
1. Operational enclosed interval propagation with per-operation outward rounding.
2. Direct logit bounds: ell_{cj}^{direct} = lower(F_c) - upper(F_j) > 0 for all j != c.
3. Whole-domain verification for both:
   - Diagonal candidate boxes Z in [Z_anchor - b, Z_anchor + b]
   - Wave-domain candidate boxes U in [U_lo, U_hi] on tail s=64 samples.
4. Separate uncertified signed-q GAP delta diagnostic (clearly labeled as non-proof).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ecdr.intervals import Interval
from ecdr.model import OperationalSequentialNumpyCNN


@dataclass(frozen=True)
class MarginBoundsResult:
    target_class: int
    num_classes: int
    direct_lower_margins: np.ndarray  # shape (K,)
    signed_q_diagnostic_margins: Optional[np.ndarray] # shape (K,) diagnostic only
    is_certified: bool
    min_direct_margin: float
    min_signed_q_diagnostic: Optional[float]
    method: str
    diagnostics: Dict[str, Union[float, str, bool]]
    domain_box: Optional[Interval] = None


def interval_conv_relu_vectorized(
    x_int: Interval, w: np.ndarray, b: np.ndarray
) -> Tuple[Interval, Interval]:
    """Interval Conv1D and ReLU vectorized across channels and positions with sequential term reduction.

    Performs per-operation outward rounding via np.nextafter.
    x_int: Interval of shape (C_in, T_in)
    w: (C_out, C_in, k)
    b: (C_out,)
    Returns (preactivation_interval, post_relu_interval) of shape (C_out, T_out).
    """
    c_out, c_in, k = w.shape
    _, t_in = x_int.shape
    t_out = t_in - k + 1

    # Initialize with bias broadcast over time positions with outward rounding
    acc_lo = np.zeros((c_out, t_out), dtype=np.float64)
    acc_hi = np.zeros((c_out, t_out), dtype=np.float64)
    acc_lo = np.nextafter(acc_lo + b[:, None], -np.inf)
    acc_hi = np.nextafter(acc_hi + b[:, None], np.inf)

    # Positive and negative weight slices
    w_pos = np.maximum(0.0, w)
    w_neg = np.minimum(0.0, w)

    # Explicit sequential loop over input channels and kernel taps (no BLAS reduction)
    for c in range(c_in):
        for r in range(k):
            x_lo_slice = x_int.lo[c, r : r + t_out][None, :]  # (1, T_out)
            x_hi_slice = x_int.hi[c, r : r + t_out][None, :]  # (1, T_out)

            w_p = w_pos[:, c, r, None]  # (C_out, 1)
            w_n = w_neg[:, c, r, None]  # (C_out, 1)

            # Product lower bound: w^+ * x_lo + w^- * x_hi
            p1_lo = np.multiply(w_p, x_lo_slice)
            p2_lo = np.multiply(w_n, x_hi_slice)
            prod_lo = np.nextafter(p1_lo + p2_lo, -np.inf)

            # Product upper bound: w^+ * x_hi + w^- * x_lo
            p1_hi = np.multiply(w_p, x_hi_slice)
            p2_hi = np.multiply(w_n, x_lo_slice)
            prod_hi = np.nextafter(p1_hi + p2_hi, np.inf)

            # Outward accumulation
            acc_lo = np.nextafter(acc_lo + prod_lo, -np.inf)
            acc_hi = np.nextafter(acc_hi + prod_hi, np.inf)

    # Abort if non-finite or inverted
    if not (np.all(np.isfinite(acc_lo)) and np.all(np.isfinite(acc_hi)) and np.all(acc_lo <= acc_hi)):
        raise ValueError("Interval overflow, non-finite value, or inversion in Conv layer")

    pre = Interval(lo=acc_lo, hi=acc_hi)
    post = Interval(lo=np.maximum(0.0, acc_lo), hi=np.maximum(0.0, acc_hi))
    return pre, post


def interval_gap_canonical(h_int: Interval) -> Interval:
    """Interval Global Average Pooling reduced sequentially in canonical order."""
    d_h, n_pos = h_int.shape
    lo_acc = np.zeros(d_h, dtype=np.float64)
    hi_acc = np.zeros(d_h, dtype=np.float64)

    for i in range(n_pos):
        lo_acc = np.nextafter(lo_acc + h_int.lo[:, i], -np.inf)
        hi_acc = np.nextafter(hi_acc + h_int.hi[:, i], np.inf)

    gap_lo = np.nextafter(lo_acc / float(n_pos), -np.inf)
    gap_hi = np.nextafter(hi_acc / float(n_pos), np.inf)

    if not (np.all(np.isfinite(gap_lo)) and np.all(np.isfinite(gap_hi)) and np.all(gap_lo <= gap_hi)):
        raise ValueError("Interval non-finite or inverted in GAP layer")

    return Interval(lo=gap_lo, hi=gap_hi)


def interval_affine_head(
    gap_int: Interval, w_head: np.ndarray, b_head: np.ndarray
) -> Interval:
    """Interval affine head: F_j = b_j + sum_a W_{j, a} * m_a with outward rounding."""
    num_classes, d_h = w_head.shape
    lo_acc = np.copy(b_head)
    hi_acc = np.copy(b_head)
    lo_acc = np.nextafter(lo_acc, -np.inf)
    hi_acc = np.nextafter(hi_acc, np.inf)

    w_pos = np.maximum(0.0, w_head)
    w_neg = np.minimum(0.0, w_head)

    for a in range(d_h):
        m_lo = gap_int.lo[a]
        m_hi = gap_int.hi[a]

        p1_lo = np.multiply(w_pos[:, a], m_lo)
        p2_lo = np.multiply(w_neg[:, a], m_hi)
        prod_lo = np.nextafter(p1_lo + p2_lo, -np.inf)

        p1_hi = np.multiply(w_pos[:, a], m_hi)
        p2_hi = np.multiply(w_neg[:, a], m_lo)
        prod_hi = np.nextafter(p1_hi + p2_hi, np.inf)

        lo_acc = np.nextafter(lo_acc + prod_lo, -np.inf)
        hi_acc = np.nextafter(hi_acc + prod_hi, np.inf)

    if not (np.all(np.isfinite(lo_acc)) and np.all(np.isfinite(hi_acc)) and np.all(lo_acc <= hi_acc)):
        raise ValueError("Interval non-finite or inverted in Affine Head")

    return Interval(lo=lo_acc, hi=hi_acc)


def compute_diagonal_box_certificate(
    net: OperationalSequentialNumpyCNN,
    z_anchor: np.ndarray,
    budget_norm: Union[float, np.ndarray],
    target_class: Optional[int] = None,
    margin_threshold: float = 0.0,
) -> MarginBoundsResult:
    """Certify full-window diagonal candidate box: |Z - Z_anchor| <= budget_norm."""
    k = net.num_classes

    # Strict validation of threshold
    if not np.isfinite(margin_threshold) or margin_threshold < 0.0:
        return MarginBoundsResult(
            target_class=target_class if target_class is not None else 0,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="diagonal_box_ibp_invalid_threshold",
            diagnostics={"error": f"margin_threshold must be finite and >= 0, got {margin_threshold}"},
        )

    # Validate target_class if given
    if target_class is not None and not (0 <= target_class < k):
        return MarginBoundsResult(
            target_class=0,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="diagonal_box_ibp_invalid_target",
            diagnostics={"error": f"target_class must be in 0..{k-1}, got {target_class}"},
        )

    try:
        x_int = Interval.from_box(z_anchor, budget_norm)
        _, h1 = interval_conv_relu_vectorized(x_int, net.w1, net.b1)
        _, h2 = interval_conv_relu_vectorized(h1, net.w2, net.b2)
        _, h3 = interval_conv_relu_vectorized(h2, net.w3, net.b3)
        gap_int = interval_gap_canonical(h3)
        logits_int = interval_affine_head(gap_int, net.w_head, net.b_head)

        if target_class is None:
            anchor_out = net.forward(z_anchor)
            target_class = anchor_out["pred_class"]

        c = target_class

        # Direct margin bounds: ell_{cj}^{direct} = lower(F_c) - upper(F_j)
        direct_margins = np.zeros(k, dtype=np.float64)
        for j in range(k):
            if j == c:
                direct_margins[j] = 0.0
            else:
                diff = logits_int.lo[c] - logits_int.hi[j]
                direct_margins[j] = np.nextafter(diff, -np.inf)

        competitors = [j for j in range(k) if j != c]
        min_direct = float(np.min(direct_margins[competitors]))

        # Strictly require min_direct > margin_threshold AND min_direct > 0 AND finite
        is_certified = bool(
            np.isfinite(min_direct)
            and (min_direct > margin_threshold)
            and (min_direct > 0.0)
        )

        return MarginBoundsResult(
            target_class=c,
            num_classes=k,
            direct_lower_margins=direct_margins,
            signed_q_diagnostic_margins=None,
            is_certified=is_certified,
            min_direct_margin=min_direct,
            min_signed_q_diagnostic=None,
            method="diagonal_box_ibp",
            diagnostics={
                "margin_threshold": float(margin_threshold),
                "num_competitors": len(competitors),
                "all_competitors_positive": bool(min_direct > 0.0),
            },
            domain_box=x_int,
        )
    except Exception as e:
        c = target_class if target_class is not None else 0
        return MarginBoundsResult(
            target_class=c,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="diagonal_box_ibp_failed",
            diagnostics={"error": str(e)},
        )


def compute_wave_domain_certificate(
    net: OperationalSequentialNumpyCNN,
    z_anchor: np.ndarray,
    wave_box: Interval,  # shape (6, shift) interval for tail samples
    shift: int = 64,
    target_class: Optional[int] = None,
    margin_threshold: float = 0.0,
) -> MarginBoundsResult:
    """Certify wave domain U in [U_lo, U_hi] with fixed overlap from z_anchor."""
    k = net.num_classes

    # Strict validation of inputs
    if not (1 <= shift <= 120):
        return MarginBoundsResult(
            target_class=target_class if target_class is not None else 0,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="wave_domain_invalid_shift",
            diagnostics={"error": f"shift must be in 1..120, got {shift}"},
        )

    if not np.isfinite(margin_threshold) or margin_threshold < 0.0:
        return MarginBoundsResult(
            target_class=target_class if target_class is not None else 0,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="wave_domain_invalid_threshold",
            diagnostics={"error": f"margin_threshold must be finite and >= 0, got {margin_threshold}"},
        )

    if target_class is not None and not (0 <= target_class < k):
        return MarginBoundsResult(
            target_class=0,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="wave_domain_invalid_target",
            diagnostics={"error": f"target_class must be in 0..{k-1}, got {target_class}"},
        )

    try:
        assert wave_box.shape == (6, shift), f"Expected wave box shape (6, {shift}), got {wave_box.shape}"

        # Evaluate anchor to obtain reference values
        anchor_res = net.forward(z_anchor)
        c = anchor_res["pred_class"] if target_class is None else target_class
        n_pos = 120

        # Construct new window interval:
        # Overlap columns (0..128-s-1): degenerate point interval from z_anchor[:, s:]
        overlap_len = 128 - shift
        overlap_lo = np.copy(z_anchor[:, shift:])
        overlap_hi = np.copy(z_anchor[:, shift:])

        full_lo = np.concatenate([overlap_lo, wave_box.lo], axis=1)
        full_hi = np.concatenate([overlap_hi, wave_box.hi], axis=1)
        window_int = Interval(lo=full_lo, hi=full_hi)

        # Propagate through CNN layers with outward interval arithmetic
        _, h1 = interval_conv_relu_vectorized(window_int, net.w1, net.b1)
        _, h2 = interval_conv_relu_vectorized(h1, net.w2, net.b2)
        _, h3 = interval_conv_relu_vectorized(h2, net.w3, net.b3)
        gap_int = interval_gap_canonical(h3)
        logits_int = interval_affine_head(gap_int, net.w_head, net.b_head)

        # 1. Direct margin bounds (sound proof of operational reference invariance)
        direct_margins = np.zeros(k, dtype=np.float64)
        for j in range(k):
            if j == c:
                direct_margins[j] = 0.0
            else:
                diff = logits_int.lo[c] - logits_int.hi[j]
                direct_margins[j] = np.nextafter(diff, -np.inf)

        competitors = [j for j in range(k) if j != c]
        min_direct = float(np.min(direct_margins[competitors]))

        # Strictly require min_direct > margin_threshold AND min_direct > 0 AND finite
        is_certified = bool(
            np.isfinite(min_direct)
            and (min_direct > margin_threshold)
            and (min_direct > 0.0)
        )

        # 2. Signed q-vector GAP-delta diagnostic (unrounded/approximate diagnostic, NOT certified proof)
        d_drop = np.sum(anchor_res["h3"][:, :shift], axis=1)  # (64,)
        tail_h3_lo = h3.lo[:, (n_pos - shift):]
        tail_h3_hi = h3.hi[:, (n_pos - shift):]

        d_new_lo = np.sum(tail_h3_lo, axis=1)
        d_new_hi = np.sum(tail_h3_hi, axis=1)

        signed_q_diag = np.zeros(k, dtype=np.float64)
        for j in range(k):
            if j == c:
                signed_q_diag[j] = 0.0
            else:
                # Explicit margin for target c: logits[c] - logits[j]
                gamma_anchor_c = anchor_res["logits"][c] - anchor_res["logits"][j]
                q_j = net.w_head[c] - net.w_head[j]
                q_pos = np.maximum(0.0, q_j)
                q_neg = np.minimum(0.0, q_j)

                term1 = np.dot(q_pos, d_new_lo)
                term2 = np.dot(q_neg, d_new_hi)
                term3 = np.dot(q_j, d_drop)

                delta_bound = (term1 + term2 - term3) / float(n_pos)
                signed_q_diag[j] = gamma_anchor_c + delta_bound

        min_signed_q_diag = float(np.min(signed_q_diag[competitors]))

        return MarginBoundsResult(
            target_class=c,
            num_classes=k,
            direct_lower_margins=direct_margins,
            signed_q_diagnostic_margins=signed_q_diag,
            is_certified=is_certified,
            min_direct_margin=min_direct,
            min_signed_q_diagnostic=min_signed_q_diag,
            method="wave_domain_tail_ibp",
            diagnostics={
                "margin_threshold": float(margin_threshold),
                "num_competitors": len(competitors),
                "direct_certified": is_certified,
                "signed_q_is_proof": False,  # Explicitly flagged as diagnostic
            },
        )
    except Exception as e:
        c = target_class if target_class is not None else 0
        return MarginBoundsResult(
            target_class=c,
            num_classes=k,
            direct_lower_margins=np.full(k, -np.inf),
            signed_q_diagnostic_margins=None,
            is_certified=False,
            min_direct_margin=-np.inf,
            min_signed_q_diagnostic=None,
            method="wave_domain_tail_failed",
            diagnostics={"error": str(e)},
        )
