"""certificate.py — Trusted-anchor lifecycle and certificate contract.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 10)
and 07_G3_DOMAIN_CONTRACT_VI.md (Sections 4, 6).

A certificate binds:
1. model_id: hash of frozen weights & architecture
2. prep_id: hash of frozen normalization statistics and channel specs
3. recording_id: recording identifier (e.g. exp in HAPT)
4. anchor_id: window index / timestamp within recording
5. target_class: certified winning class index
6. domain_type: "diagonal_box" or "wave_domain"
7. domain_params: budget or wave box interval (immutable copies)
8. margins: verified lower bounds for all K-1 competitors (all strictly positive)
9. horizon: number of valid transitions (default 1 for 1-step wave certificate)
10. status: ACTIVE, EXPIRED, or INVALIDATED

Reset/invalidation policies:
- Recording boundary transition ALWAYS invalidates active certificates.
- Discontinuity or overlap mismatch invalidates wave certificate.
- A hit does NOT renew an unverified wave anchor for subsequent shifts.
"""

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ecdr.intervals import Interval


def compute_window_hash(window: np.ndarray) -> str:
    """Compute deterministic cryptographic SHA-256 fingerprint from float64 window shape and bytes."""
    w = np.ascontiguousarray(np.asarray(window, dtype=np.float64))
    hasher = hashlib.sha256()
    hasher.update(f"shape:{w.shape}:".encode("utf-8"))
    hasher.update(w.tobytes())
    return hasher.hexdigest()


@dataclass
class Certificate:
    cert_id: str
    model_id: str
    prep_id: str
    recording_id: str
    anchor_id: int
    target_class: int
    domain_type: str
    anchor_state_hash: str = ""
    domain_box: Optional[Interval] = None
    budget_norm: Optional[np.ndarray] = None
    wave_box: Optional[Interval] = None
    min_lower_margin: float = 0.0
    all_lower_margins: Optional[np.ndarray] = None
    horizon: int = 1
    steps_used: int = 0
    created_at_sec: float = field(default_factory=time.time)
    is_active: bool = True
    integrity_hash: str = field(default="")

    def __post_init__(self):
        # Deep-copy and freeze arrays to prevent external alias mutation
        if self.budget_norm is not None:
            b_arr = np.copy(np.asarray(self.budget_norm, dtype=np.float64))
            b_arr.flags.writeable = False
            self.budget_norm = b_arr

        if self.all_lower_margins is not None:
            m_arr = np.copy(np.asarray(self.all_lower_margins, dtype=np.float64))
            m_arr.flags.writeable = False
            self.all_lower_margins = m_arr

        # Compute initial integrity hash binding all immutable fields and domain endpoints
        if not self.integrity_hash:
            self.integrity_hash = self.compute_integrity_hash()

    def compute_integrity_hash(self) -> str:
        """Compute SHA-256 integrity hash binding model, prep, anchor, domain endpoints, and margins."""
        hasher = hashlib.sha256()
        header = (
            f"model_id:{self.model_id}|prep_id:{self.prep_id}|"
            f"rec:{self.recording_id}|anchor:{self.anchor_id}|"
            f"target:{self.target_class}|domain:{self.domain_type}|"
            f"anchor_hash:{self.anchor_state_hash}|"
            f"horizon:{self.horizon}|min_margin:{float(self.min_lower_margin):.17g}:"
        )
        hasher.update(header.encode("utf-8"))
        if self.domain_box is not None:
            hasher.update(b"dbox_lo:" + self.domain_box.lo.tobytes())
            hasher.update(b"dbox_hi:" + self.domain_box.hi.tobytes())
        if self.budget_norm is not None:
            hasher.update(b"budget:" + self.budget_norm.tobytes())
        if self.wave_box is not None:
            hasher.update(b"wbox_lo:" + self.wave_box.lo.tobytes())
            hasher.update(b"wbox_hi:" + self.wave_box.hi.tobytes())
        if self.all_lower_margins is not None:
            hasher.update(b"margins:" + self.all_lower_margins.tobytes())
        return hasher.hexdigest()

    def validate(
        self,
        expected_k: Optional[int] = None,
        expected_model_id: Optional[str] = None,
        expected_prep_id: Optional[str] = None,
        expected_recording_id: Optional[str] = None,
        expected_anchor_window: Optional[np.ndarray] = None,
        raise_on_error: bool = False,
    ) -> bool:
        """Strict validation of certificate gate requirements.

        Enforces:
        1. Certificate is active and non-expired.
        2. Expected model, preprocessing, and recording identities match (if specified).
        3. Anchor state hash is non-empty and matches expected anchor window content (if specified).
        4. Cryptographic integrity hash matches (detects in-place field/endpoint mutation).
        5. Vector all_lower_margins is finite, 1D array of length K >= 2.
        6. Expected class count matches expected_k (if specified).
        7. target_class is integer in 0 <= target_class < K.
        8. All K-1 competitor margins (j != target_class) are strictly positive (> 0.0) and finite.
        9. Scalar min_lower_margin is finite, strictly positive, and consistent with competitor vector minimum.
        10. Expected domain payloads and valid endpoints exist:
           - diagonal_box: domain_box Interval with lo <= hi finite, and budget_norm non-negative finite.
           - wave_domain: wave_box Interval with lo <= hi finite.
        """
        def _fail(msg: str):
            if raise_on_error:
                raise ValueError(f"Certificate validation failed: {msg}")
            return False

        if not self.is_active:
            return _fail("Certificate is inactive or expired")

        # Identity matches
        if expected_model_id is not None and self.model_id != expected_model_id:
            return _fail(f"Model ID mismatch: {self.model_id} != {expected_model_id}")
        if expected_prep_id is not None and self.prep_id != expected_prep_id:
            return _fail(f"Preprocessing ID mismatch: {self.prep_id} != {expected_prep_id}")
        if expected_recording_id is not None and str(self.recording_id) != str(expected_recording_id):
            return _fail(f"Recording ID mismatch: {self.recording_id} != {expected_recording_id}")

        # Proof-time anchor state hash must be non-empty
        if not self.anchor_state_hash:
            return _fail("Certificate missing required anchor_state_hash")

        # Compare expected anchor window content fingerprint if provided
        if expected_anchor_window is not None:
            w_hash = compute_window_hash(expected_anchor_window)
            if self.anchor_state_hash != w_hash:
                return _fail(
                    f"Anchor window content mismatch: expected {w_hash} != cert {self.anchor_state_hash}"
                )

        # Integrity hash verification
        if self.compute_integrity_hash() != self.integrity_hash:
            return _fail("Integrity hash mismatch: certificate fields or domain endpoints were mutated")

        # Vector margins check
        if self.all_lower_margins is None:
            return _fail("all_lower_margins vector is missing")
        if self.all_lower_margins.ndim != 1:
            return _fail(f"all_lower_margins must be 1D, got ndim={self.all_lower_margins.ndim}")

        k = len(self.all_lower_margins)
        if k < 2:
            return _fail(f"Invalid class count K={k}; minimum 2 required")
        if expected_k is not None and k != expected_k:
            return _fail(f"Expected K={expected_k} classes, but certificate has {k}")
        if not np.all(np.isfinite(self.all_lower_margins)):
            return _fail("all_lower_margins contains non-finite (NaN or Inf) values")

        # Target class check
        if not isinstance(self.target_class, (int, np.integer)):
            return _fail(f"target_class must be int, got {type(self.target_class)}")
        c = int(self.target_class)
        if not (0 <= c < k):
            return _fail(f"target_class {c} out of valid range [0, {k-1}]")

        # All K-1 competitor margins strictly positive
        competitors = [j for j in range(k) if j != c]
        competitor_margins = self.all_lower_margins[competitors]
        if not np.all(competitor_margins > 0.0):
            min_comp = float(np.min(competitor_margins))
            return _fail(f"Not all K-1 competitor margins are strictly positive; min competitor margin is {min_comp}")

        # Scalar min_lower_margin consistency
        if not (np.isfinite(self.min_lower_margin) and self.min_lower_margin > 0.0):
            return _fail(f"min_lower_margin {self.min_lower_margin} must be finite and > 0")
        expected_min = float(np.min(competitor_margins))
        if abs(self.min_lower_margin - expected_min) > 1e-12:
            return _fail(
                f"min_lower_margin {self.min_lower_margin} inconsistent with vector min {expected_min}"
            )

        # Domain payload and endpoints
        if self.domain_type == "diagonal_box":
            if self.domain_box is None:
                return _fail("diagonal_box certificate missing domain_box Interval")
            if not isinstance(self.domain_box, Interval):
                return _fail(f"domain_box must be Interval, got {type(self.domain_box)}")
            if not (np.all(np.isfinite(self.domain_box.lo)) and np.all(np.isfinite(self.domain_box.hi))):
                return _fail("domain_box contains non-finite endpoints")
            if not np.all(self.domain_box.lo <= self.domain_box.hi):
                return _fail("domain_box endpoints inverted: lo > hi")
            if self.budget_norm is None:
                return _fail("diagonal_box certificate missing budget_norm")
            if not np.all(np.isfinite(self.budget_norm)):
                return _fail("budget_norm contains non-finite values")
            if np.any(self.budget_norm < 0.0):
                return _fail("budget_norm contains negative values")

        elif self.domain_type == "wave_domain":
            if self.wave_box is None:
                return _fail("wave_domain certificate missing wave_box Interval")
            if not isinstance(self.wave_box, Interval):
                return _fail(f"wave_box must be Interval, got {type(self.wave_box)}")
            if not (np.all(np.isfinite(self.wave_box.lo)) and np.all(np.isfinite(self.wave_box.hi))):
                return _fail("wave_box contains non-finite endpoints")
            if not np.all(self.wave_box.lo <= self.wave_box.hi):
                return _fail("wave_box endpoints inverted: lo > hi")

        else:
            return _fail(f"Unsupported domain_type: {self.domain_type}")

        return True

    def is_valid_for_recording(self, current_recording_id: str) -> bool:
        """Check if certificate belongs to current recording."""
        return self.is_active and (str(self.recording_id) == str(current_recording_id))

    def is_compatible(self, model_id: str, prep_id: str) -> bool:
        """Check if certificate matches active model and preprocessing identities."""
        return (self.model_id == model_id) and (self.prep_id == prep_id)

    def consume_step(self) -> None:
        """Increment step counter and expire if horizon reached."""
        self.steps_used += 1
        if self.steps_used >= self.horizon:
            self.is_active = False

    def invalidate(self, reason: str = "invalidated") -> None:
        """Deactivate certificate."""
        self.is_active = False


class CertificateManager:
    """Manages trusted-anchor lifecycle, caching, and state transitions."""

    def __init__(self, model_id: str, prep_id: str):
        self.model_id = model_id
        self.prep_id = prep_id
        self.active_cert: Optional[Certificate] = None
        self.anchor_window: Optional[np.ndarray] = None
        self.current_recording_id: Optional[str] = None
        self.anchor_step: int = -1

    def reset_recording(self, new_recording_id: str) -> None:
        """Strict reset at recording boundary: invalidates all certificates and clears anchor."""
        if self.active_cert is not None:
            self.active_cert.invalidate("recording_boundary_reset")
        self.active_cert = None
        self.anchor_window = None
        self.current_recording_id = str(new_recording_id)
        self.anchor_step = -1

    def establish_trusted_anchor(
        self,
        window: np.ndarray,
        step_idx: int,
        recording_id: str,
        cert: Optional[Certificate] = None,
    ) -> None:
        """Set a newly computed reference window as the trusted anchor."""
        if str(recording_id) != self.current_recording_id:
            self.reset_recording(str(recording_id))

        # Enforce that certificate's anchor_id matches this anchor's step_idx and passes gate validation
        if cert is not None:
            if not cert.is_active:
                self.active_cert = None
            else:
                if cert.anchor_id != step_idx:
                    raise ValueError(
                        f"Stale anchor rebind rejected: certificate anchor_id={cert.anchor_id} "
                        f"does not match anchor step_idx={step_idx}"
                    )
                cert.validate(
                    expected_model_id=self.model_id,
                    expected_prep_id=self.prep_id,
                    expected_recording_id=str(recording_id),
                    expected_anchor_window=window,
                    raise_on_error=True,
                )
                self.active_cert = cert
        else:
            self.active_cert = None

        anchor_copy = np.copy(np.asarray(window, dtype=np.float64))
        anchor_copy.flags.writeable = False
        self.anchor_window = anchor_copy
        self.anchor_step = step_idx

    def get_active_certificate(self) -> Optional[Certificate]:
        """Return active certificate if valid and non-expired, else None."""
        if self.active_cert is not None and self.active_cert.is_active:
            return self.active_cert
        return None
