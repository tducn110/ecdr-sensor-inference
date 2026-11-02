"""membership.py — Conservative domain membership checkers.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 9.1, 10.2)
and 07_G3_DOMAIN_CONTRACT_VI.md (Section 3).

Soundness obligations:
1. Rejects non-finite queries (NaN, +/-Inf) immediately.
2. Checks containment directly against the EXACT stored finite interval endpoints [lo, hi]
   used by the certifier, never computing loose/rounded distance |query - anchor| <= budget.
   This guarantees that proof domain and membership domain strictly coincide bitwise.
3. For wave-domain S(Z^r, U): checks both exact bitwise overlap equality
   query[:, :W-s] == anchor[:, s:] AND tail wave containment U_query in [U_lo, U_hi].
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from ecdr.intervals import Interval


@dataclass(frozen=True)
class MembershipResult:
    is_member: bool
    domain_type: str
    reason: str
    max_violation: float = 0.0


def check_diagonal_box_membership(
    z_query: np.ndarray,
    z_anchor: Optional[np.ndarray] = None,
    budget_norm: Optional[Union[float, np.ndarray]] = None,
    domain_box: Optional[Interval] = None,
) -> MembershipResult:
    """Check containment directly against certified interval endpoints [lo, hi].

    Either domain_box (Interval) or (z_anchor, budget_norm) must be provided.
    Direct endpoint comparison prevents false accepts caused by floating point
    roundoff in abs(query - anchor).
    """
    q_arr = np.asarray(z_query, dtype=np.float64)
    if not np.all(np.isfinite(q_arr)):
        return MembershipResult(
            is_member=False,
            domain_type="diagonal_box",
            reason="Non-finite values detected in query",
            max_violation=float("inf"),
        )

    # Resolve certified domain box
    if domain_box is not None:
        box = domain_box
    elif z_anchor is not None and budget_norm is not None:
        anc_arr = np.asarray(z_anchor, dtype=np.float64)
        if not np.all(np.isfinite(anc_arr)):
            return MembershipResult(
                is_member=False,
                domain_type="diagonal_box",
                reason="Non-finite values detected in anchor",
                max_violation=float("inf"),
            )
        try:
            box = Interval.from_box(anc_arr, budget_norm)
        except Exception as e:
            return MembershipResult(
                is_member=False,
                domain_type="diagonal_box",
                reason=f"Failed to construct interval box: {e}",
                max_violation=float("inf"),
            )
    else:
        raise ValueError("Must provide either domain_box or (z_anchor, budget_norm)")

    if q_arr.shape != box.shape:
        return MembershipResult(
            is_member=False,
            domain_type="diagonal_box",
            reason=f"Shape mismatch: query {q_arr.shape} vs domain box {box.shape}",
            max_violation=float("inf"),
        )

    # Strict direct comparison against exact certified interval endpoints
    viol_lo = box.lo - q_arr  # > 0 if q_arr < box.lo
    viol_hi = q_arr - box.hi  # > 0 if q_arr > box.hi

    max_viol = float(max(np.max(viol_lo), np.max(viol_hi)))
    if max_viol <= 0.0:
        return MembershipResult(
            is_member=True,
            domain_type="diagonal_box",
            reason="Contained in certified diagonal interval box",
            max_violation=max_viol,
        )
    else:
        return MembershipResult(
            is_member=False,
            domain_type="diagonal_box",
            reason=f"Query falls outside certified interval endpoints by {max_viol:.6e}",
            max_violation=max_viol,
        )


def check_wave_domain_membership(
    z_query: np.ndarray,
    z_anchor: np.ndarray,
    wave_box: Interval,
    shift: int = 64,
) -> MembershipResult:
    """Check containment in wave domain S(Z_anchor, U).

    Requires:
    1. Exact bitwise equality on overlap: z_query[:, :W-s] == z_anchor[:, s:]
    2. Containment of tail wave directly against wave_box interval endpoints.
    """
    q_arr = np.asarray(z_query, dtype=np.float64)
    anc_arr = np.asarray(z_anchor, dtype=np.float64)

    if not (np.all(np.isfinite(q_arr)) and np.all(np.isfinite(anc_arr))):
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason="Non-finite values detected in query or anchor",
            max_violation=float("inf"),
        )

    if q_arr.shape != anc_arr.shape:
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason=f"Shape mismatch: query {q_arr.shape} vs anchor {anc_arr.shape}",
            max_violation=float("inf"),
        )

    num_channels, window_length = anc_arr.shape
    overlap_len = window_length - shift
    if overlap_len <= 0 or overlap_len >= window_length:
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason=f"Invalid shift {shift} for window length {window_length}",
            max_violation=float("inf"),
        )

    # Check exact overlap equality
    if not np.array_equal(q_arr[:, :overlap_len], anc_arr[:, shift:]):
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason="Overlap mismatch with anchor state",
            max_violation=float("inf"),
        )

    # Check tail wave interval containment
    u_query = q_arr[:, overlap_len:]
    if u_query.shape != wave_box.shape:
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason=f"Tail wave shape {u_query.shape} does not match wave box shape {wave_box.shape}",
            max_violation=float("inf"),
        )

    viol_lo = wave_box.lo - u_query
    viol_hi = u_query - wave_box.hi
    max_viol = float(max(np.max(viol_lo), np.max(viol_hi)))

    if max_viol <= 0.0:
        return MembershipResult(
            is_member=True,
            domain_type="wave_domain",
            reason="Contained in certified wave domain",
            max_violation=max_viol,
        )
    else:
        return MembershipResult(
            is_member=False,
            domain_type="wave_domain",
            reason=f"Tail wave outside certified box by {max_viol:.6e}",
            max_violation=max_viol,
        )
