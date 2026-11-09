"""ecdr.week4.types — Immutable public data types and contracts for Week 4.

Defines QueryView (strictly without labels/oracle information),
ReferenceIdentity, CandidateBox, TrustedAnchorReceipt, MembershipResult,
BoundResult, NumericalCapability, Certificate, and RuntimeDecision.
All public dataclasses enforce frozen immutability, schema versioning,
and strict non-NaN serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np

SCHEMA_VERSION: str = "w4.v1"
T = TypeVar("T")


class SerializationError(ValueError):
    """Raised when serialization or deserialization fails strict validation."""


def _digest_json(value: Any) -> str:
    """Deterministic JSON digest rejecting NaN/Inf."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _ensure_finite_float_tuple(values: Any, expected_len: Optional[int] = None) -> Tuple[float, ...]:
    """Convert input sequence into a tuple of strictly finite floats."""
    if isinstance(values, np.ndarray):
        flat = values.ravel()
    elif isinstance(values, (list, tuple)):
        flat = list(values)
    else:
        raise ValueError(f"Expected sequence or ndarray, got {type(values)}")

    if expected_len is not None and len(flat) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(flat)}")

    result = []
    for x in flat:
        if isinstance(x, (bool, np.bool_)):
            raise ValueError("Boolean values are not valid real numbers")
        try:
            val = float(x)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Cannot convert {x!r} to float") from err
        if not np.isfinite(val):
            raise ValueError(f"Value must be finite real number, got {val}")
        result.append(val)
    return tuple(result)


# ==============================================================================
# Disjoint Status Enumerations
# ==============================================================================

class MembershipStatus(str, Enum):
    EXACT_MEMBER = "EXACT_MEMBER"
    NON_MEMBER = "NON_MEMBER"
    VALIDATED_OUTSIDE = "VALIDATED_OUTSIDE"
    CONDITIONAL_MEMBER = "CONDITIONAL_MEMBER"
    UNSUPPORTED_DOMAIN = "UNSUPPORTED_DOMAIN"
    ERROR = "ERROR"


class BoundStatus(str, Enum):
    EXACT_ENCLOSURE = "EXACT_ENCLOSURE"
    CONDITIONAL_POSITIVE = "CONDITIONAL_POSITIVE"
    CONDITIONAL_INCONCLUSIVE = "CONDITIONAL_INCONCLUSIVE"
    VALIDATED_POSITIVE = "VALIDATED_POSITIVE"
    VALIDATED_INCONCLUSIVE = "VALIDATED_INCONCLUSIVE"
    FAILED = "FAILED"
    UNSUPPORTED_METHOD = "UNSUPPORTED_METHOD"


class NumericalStatus(str, Enum):
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    VALIDATED_FLOAT32_ENCLOSURE = "VALIDATED_FLOAT32_ENCLOSURE"
    CONDITIONAL_REAL_ARITHMETIC = "CONDITIONAL_REAL_ARITHMETIC"
    UNSOUND = "UNSOUND"


class DecisionAction(str, Enum):
    REUSE_CERTIFIED = "REUSE_CERTIFIED"
    FALLBACK_REFERENCE = "FALLBACK_REFERENCE"
    FALLBACK_UNSUPPORTED = "FALLBACK_UNSUPPORTED"
    FALLBACK_EXPIRED = "FALLBACK_EXPIRED"
    FALLBACK_IDENTITY_MISMATCH = "FALLBACK_IDENTITY_MISMATCH"
    FALLBACK_RECORDING_RESET = "FALLBACK_RECORDING_RESET"
    FALLBACK_NON_MEMBER = "FALLBACK_NON_MEMBER"
    FALLBACK_NUMERICAL_GATE_CLOSED = "FALLBACK_NUMERICAL_GATE_CLOSED"


class ExecutionMode(str, Enum):
    STRONG_IDENTITY = "STRONG_IDENTITY"
    OWNED_REFERENCE = "OWNED_REFERENCE"


# ==============================================================================
# C01 — QueryView
# ==============================================================================

@dataclass(frozen=True)
class QueryView:
    """Owned raw input window at an explicit stream step.

    STRICT CONTRACT: Contains ONLY raw sensor window (shape 6x128), global row ID,
    recording ID, and step index. Absolutely NO label, ground-truth activity,
    purity, split information, future inputs, or split-wide logits.
    Evaluator holds ground truth labels separately and joins only after decisions.
    """
    global_row_id: int
    recording_id: str
    step_idx: int
    raw_window: Tuple[float, ...] = field(repr=False)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        if isinstance(self.global_row_id, (bool, np.bool_)) or not isinstance(self.global_row_id, (int, np.integer)):
            raise ValueError("global_row_id must be a non-negative integer")
        if int(self.global_row_id) < 0:
            raise ValueError("global_row_id must be >= 0")
        if not isinstance(self.recording_id, str) or not self.recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        if isinstance(self.step_idx, (bool, np.bool_)) or not isinstance(self.step_idx, (int, np.integer)):
            raise ValueError("step_idx must be a non-negative integer")
        if int(self.step_idx) < 0:
            raise ValueError("step_idx must be >= 0")

        # Convert raw_window to strictly finite float tuple of length 768
        tup = _ensure_finite_float_tuple(self.raw_window, expected_len=768)
        object.__setattr__(self, "raw_window", tup)
        object.__setattr__(self, "global_row_id", int(self.global_row_id))
        object.__setattr__(self, "step_idx", int(self.step_idx))

    @classmethod
    def from_array(
        cls,
        raw: np.ndarray,
        global_row_id: int,
        recording_id: str,
        step_idx: int,
    ) -> QueryView:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.shape != (6, 128):
            raise ValueError(f"raw window must have shape (6, 128), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("raw window contains non-finite values")
        return cls(
            global_row_id=global_row_id,
            recording_id=recording_id,
            step_idx=step_idx,
            raw_window=tuple(float(x) for x in arr.ravel()),
        )

    def array(self) -> np.ndarray:
        """Return a copy of the raw window as contiguous float32 ndarray shape (6, 128)."""
        arr = np.asarray(self.raw_window, dtype=np.float32).reshape(6, 128).copy()
        arr.flags.writeable = False
        return arr

    @property
    def raw_hash(self) -> str:
        arr = np.asarray(self.raw_window, dtype=np.float32).reshape(6, 128)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_row_id": self.global_row_id,
            "recording_id": self.recording_id,
            "step_idx": self.step_idx,
            "raw_window": list(self.raw_window),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryView:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            global_row_id=data["global_row_id"],
            recording_id=data["recording_id"],
            step_idx=data["step_idx"],
            raw_window=tuple(data["raw_window"]),
            schema_version=data["schema_version"],
        )


# ==============================================================================
# C02 / C03 — Reference Identity
# ==============================================================================

@dataclass(frozen=True)
class ReferenceIdentity:
    """Authoritative frozen reference model and preprocessing identity."""
    checkpoint_sha256: str
    preprocessing_id: str
    class_order: Tuple[str, ...]
    graph_id: str
    weights_id: str
    bn_id: str
    torch_version: str
    numpy_version: str
    dtype: str = "float32"
    device: str = "cpu"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        if len(self.class_order) != 12:
            raise ValueError(f"class_order must have 12 classes, got {len(self.class_order)}")
        if self.dtype != "float32":
            raise ValueError(f"Only float32 reference supported, got {self.dtype}")
        if self.device != "cpu":
            raise ValueError(f"Only CPU reference supported, got {self.device}")

    @property
    def identity_id(self) -> str:
        data = {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_id": self.preprocessing_id,
            "class_order": list(self.class_order),
            "graph_id": self.graph_id,
            "weights_id": self.weights_id,
            "bn_id": self.bn_id,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "dtype": self.dtype,
            "device": self.device,
        }
        return _digest_json(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_id": self.identity_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_id": self.preprocessing_id,
            "class_order": list(self.class_order),
            "graph_id": self.graph_id,
            "weights_id": self.weights_id,
            "bn_id": self.bn_id,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "dtype": self.dtype,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReferenceIdentity:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            checkpoint_sha256=data["checkpoint_sha256"],
            preprocessing_id=data["preprocessing_id"],
            class_order=tuple(data["class_order"]),
            graph_id=data["graph_id"],
            weights_id=data["weights_id"],
            bn_id=data["bn_id"],
            torch_version=data["torch_version"],
            numpy_version=data["numpy_version"],
            dtype=data.get("dtype", "float32"),
            device=data.get("device", "cpu"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# CandidateBox Domain
# ==============================================================================

@dataclass(frozen=True)
class CandidateBox:
    """Stored float64 endpoint box; no implicit comparison tolerance.

    Endpoints define the domain. Stored as tuples of 768 floats (shape 6, 128).
    """
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    kind: str
    free_dimensions: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        lo = _ensure_finite_float_tuple(self.lower, expected_len=768)
        hi = _ensure_finite_float_tuple(self.upper, expected_len=768)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

        arr_lo = np.asarray(lo).reshape(6, 128)
        arr_hi = np.asarray(hi).reshape(6, 128)
        if np.any(arr_lo > arr_hi):
            raise ValueError("domain lower bounds must be <= upper bounds")

        if self.kind not in ("diagonal_box", "fixed_overlap_wave"):
            raise ValueError(f"unsupported candidate domain kind: {self.kind}")

        nonzeros = int(np.count_nonzero(arr_lo != arr_hi))
        if self.free_dimensions != nonzeros:
            raise ValueError(f"free_dimensions mismatch: declared {self.free_dimensions}, actual {nonzeros}")

        if self.kind == "fixed_overlap_wave":
            if not np.array_equal(arr_lo[:, :64], arr_hi[:, :64]):
                raise ValueError("fixed_overlap_wave requires exact equality in first 64 steps")

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(self.lower, dtype=np.float64).reshape(6, 128)
        hi = np.asarray(self.upper, dtype=np.float64).reshape(6, 128)
        return lo, hi

    @property
    def domain_id(self) -> str:
        return _digest_json({
            "schema_version": self.schema_version,
            "kind": self.kind,
            "free_dimensions": self.free_dimensions,
            "lower": list(self.lower),
            "upper": list(self.upper),
        })

    def contains(self, normalized_query: Any) -> bool:
        """Check whether normalized_query is inside [lower, upper] in float64."""
        arr = np.asarray(normalized_query, dtype=np.float64)
        if arr.shape != (6, 128) or not np.all(np.isfinite(arr)):
            return False
        lo, hi = self.arrays()
        return bool(np.all(arr >= lo) and np.all(arr <= hi))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain_id": self.domain_id,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "kind": self.kind,
            "free_dimensions": self.free_dimensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateBox:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            lower=tuple(data["lower"]),
            upper=tuple(data["upper"]),
            kind=data["kind"],
            free_dimensions=data["free_dimensions"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# C05 — TrustedAnchorReceipt
# ==============================================================================

@dataclass(frozen=True)
class TrustedAnchorReceipt:
    """Cryptographic proof of reference inference executed by ReferenceOwner.

    A caller cannot fabricate a valid receipt without the ReferenceOwner's
    internal signature token and generation identifier.
    """
    reference_identity_id: str
    checkpoint_sha256: str
    preprocessing_id: str
    graph_id: str
    weights_id: str
    query_row_id: int
    recording_id: str
    step_idx: int
    raw_hash: str
    normalized_hash: str
    logits: Tuple[float, ...]
    winner: int
    margin_to_runner_up: float
    elapsed_ms: float
    issuer_signature: str
    generation_id: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        tup_logits = _ensure_finite_float_tuple(self.logits, expected_len=12)
        object.__setattr__(self, "logits", tup_logits)
        if isinstance(self.winner, (bool, np.bool_)) or not isinstance(self.winner, (int, np.integer)):
            raise ValueError("winner must be an integer in 0..11")
        if self.winner not in range(12):
            raise ValueError(f"winner must be in 0..11, got {self.winner}")
        if not np.isfinite(self.margin_to_runner_up):
            raise ValueError("margin_to_runner_up must be finite")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        if not self.issuer_signature:
            raise ValueError("issuer_signature must be non-empty")
        object.__setattr__(self, "winner", int(self.winner))
        object.__setattr__(self, "query_row_id", int(self.query_row_id))
        object.__setattr__(self, "step_idx", int(self.step_idx))
        object.__setattr__(self, "generation_id", int(self.generation_id))

    @property
    def receipt_id(self) -> str:
        return _digest_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_identity_id": self.reference_identity_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_id": self.preprocessing_id,
            "graph_id": self.graph_id,
            "weights_id": self.weights_id,
            "query_row_id": self.query_row_id,
            "recording_id": self.recording_id,
            "step_idx": self.step_idx,
            "raw_hash": self.raw_hash,
            "normalized_hash": self.normalized_hash,
            "logits": list(self.logits),
            "winner": self.winner,
            "margin_to_runner_up": self.margin_to_runner_up,
            "elapsed_ms": self.elapsed_ms,
            "issuer_signature": self.issuer_signature,
            "generation_id": self.generation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrustedAnchorReceipt:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            reference_identity_id=data["reference_identity_id"],
            checkpoint_sha256=data["checkpoint_sha256"],
            preprocessing_id=data["preprocessing_id"],
            graph_id=data["graph_id"],
            weights_id=data["weights_id"],
            query_row_id=data["query_row_id"],
            recording_id=data["recording_id"],
            step_idx=data["step_idx"],
            raw_hash=data["raw_hash"],
            normalized_hash=data["normalized_hash"],
            logits=tuple(data["logits"]),
            winner=data["winner"],
            margin_to_runner_up=data["margin_to_runner_up"],
            elapsed_ms=data["elapsed_ms"],
            issuer_signature=data["issuer_signature"],
            generation_id=data["generation_id"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# MembershipResult
# ==============================================================================

@dataclass(frozen=True)
class MembershipResult:
    """Result of testing whether a query belongs to a candidate domain."""
    domain_id: str
    status: str
    is_member: bool
    reason: str
    method_id: str
    elapsed_ms: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        valid_statuses = {s.value for s in MembershipStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Unknown status {self.status}, must be one of {valid_statuses}")
        if not isinstance(self.is_member, bool):
            raise ValueError("is_member must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain_id": self.domain_id,
            "status": self.status,
            "is_member": self.is_member,
            "reason": self.reason,
            "method_id": self.method_id,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MembershipResult:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            domain_id=data["domain_id"],
            status=data["status"],
            is_member=data["is_member"],
            reason=data["reason"],
            method_id=data["method_id"],
            elapsed_ms=data["elapsed_ms"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# BoundResult
# ==============================================================================

@dataclass(frozen=True)
class BoundResult:
    """Formal or conditional lower margin bounds for a candidate domain."""
    method_id: str
    domain_id: str
    reference_identity_id: str
    target_class: int
    competitor_classes: Tuple[int, ...]
    lower_margins: Tuple[float, ...]
    status: str
    numerical_status: str
    is_positive: bool
    failure_reason: str = ""
    bound_ms: float = 0.0
    total_ms: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        valid_b_statuses = {s.value for s in BoundStatus}
        if self.status not in valid_b_statuses:
            raise ValueError(f"Unknown BoundStatus {self.status}, must be in {valid_b_statuses}")
        valid_n_statuses = {s.value for s in NumericalStatus}
        if self.numerical_status not in valid_n_statuses:
            raise ValueError(f"Unknown NumericalStatus {self.numerical_status}, must be in {valid_n_statuses}")
        if isinstance(self.target_class, (bool, np.bool_)) or self.target_class not in range(12):
            raise ValueError("target_class must be in 0..11")
        if len(self.competitor_classes) != 11:
            raise ValueError(f"competitor_classes must have exactly 11 items, got {len(self.competitor_classes)}")
        if self.target_class in self.competitor_classes:
            raise ValueError("target_class cannot be in competitor_classes")
        if len(set(self.competitor_classes)) != 11:
            raise ValueError("competitor_classes contains duplicates")

        margins = _ensure_finite_float_tuple(self.lower_margins, expected_len=11)
        object.__setattr__(self, "lower_margins", margins)
        object.__setattr__(self, "target_class", int(self.target_class))
        object.__setattr__(self, "competitor_classes", tuple(int(x) for x in self.competitor_classes))

    @property
    def min_lower_margin(self) -> float:
        return min(self.lower_margins)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "domain_id": self.domain_id,
            "reference_identity_id": self.reference_identity_id,
            "target_class": self.target_class,
            "competitor_classes": list(self.competitor_classes),
            "lower_margins": list(self.lower_margins),
            "status": self.status,
            "numerical_status": self.numerical_status,
            "is_positive": self.is_positive,
            "failure_reason": self.failure_reason,
            "bound_ms": self.bound_ms,
            "total_ms": self.total_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundResult:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            method_id=data["method_id"],
            domain_id=data["domain_id"],
            reference_identity_id=data["reference_identity_id"],
            target_class=data["target_class"],
            competitor_classes=tuple(data["competitor_classes"]),
            lower_margins=tuple(data["lower_margins"]),
            status=data["status"],
            numerical_status=data["numerical_status"],
            is_positive=data["is_positive"],
            failure_reason=data.get("failure_reason", ""),
            bound_ms=data.get("bound_ms", 0.0),
            total_ms=data.get("total_ms", 0.0),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# NumericalCapability
# ==============================================================================

@dataclass(frozen=True)
class NumericalCapability:
    """Status of the numerical rounding bridge and soundness proof."""
    capability_id: str
    status: str
    directed_rounding_verified: bool
    soundness_proof_reference: str
    formal_reuse_eligible: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        valid_statuses = {s.value for s in NumericalStatus}
        if self.status not in valid_statuses:
            raise ValueError(f"Unknown status {self.status}, must be in {valid_statuses}")
        if self.status != NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value and self.formal_reuse_eligible:
            raise ValueError("formal_reuse_eligible cannot be True without VALIDATED_FLOAT32_ENCLOSURE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "status": self.status,
            "directed_rounding_verified": self.directed_rounding_verified,
            "soundness_proof_reference": self.soundness_proof_reference,
            "formal_reuse_eligible": self.formal_reuse_eligible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NumericalCapability:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            capability_id=data["capability_id"],
            status=data["status"],
            directed_rounding_verified=data["directed_rounding_verified"],
            soundness_proof_reference=data["soundness_proof_reference"],
            formal_reuse_eligible=data["formal_reuse_eligible"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# Certificate
# ==============================================================================

@dataclass(frozen=True)
class Certificate:
    """Formal or conditional invariance certificate over a candidate domain."""
    reference_identity_id: str
    checkpoint_sha256: str
    preprocessing_id: str
    graph_id: str
    anchor_raw_hash: str
    anchor_normalized_hash: str
    anchor_step: int
    recording_id: str
    target_class: int
    competitor_classes: Tuple[int, ...]
    domain: CandidateBox
    lower_margins: Tuple[float, ...]
    max_age: int
    method_id: str
    numerical_status: str
    lifecycle: str
    formal_reuse_enabled: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if isinstance(self.target_class, (bool, np.bool_)) or self.target_class not in range(12):
            raise ValueError("target_class must be in 0..11")
        if len(self.competitor_classes) != 11:
            raise ValueError("competitor_classes must have 11 items")
        margins = _ensure_finite_float_tuple(self.lower_margins, expected_len=11)
        object.__setattr__(self, "lower_margins", margins)
        object.__setattr__(self, "target_class", int(self.target_class))
        object.__setattr__(self, "anchor_step", int(self.anchor_step))

    @property
    def certificate_id(self) -> str:
        data = self.to_dict()
        return _digest_json(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_identity_id": self.reference_identity_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing_id": self.preprocessing_id,
            "graph_id": self.graph_id,
            "anchor_raw_hash": self.anchor_raw_hash,
            "anchor_normalized_hash": self.anchor_normalized_hash,
            "anchor_step": self.anchor_step,
            "recording_id": self.recording_id,
            "target_class": self.target_class,
            "competitor_classes": list(self.competitor_classes),
            "domain": self.domain.to_dict(),
            "lower_margins": list(self.lower_margins),
            "max_age": self.max_age,
            "method_id": self.method_id,
            "numerical_status": self.numerical_status,
            "lifecycle": self.lifecycle,
            "formal_reuse_enabled": self.formal_reuse_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Certificate:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            reference_identity_id=data["reference_identity_id"],
            checkpoint_sha256=data["checkpoint_sha256"],
            preprocessing_id=data["preprocessing_id"],
            graph_id=data["graph_id"],
            anchor_raw_hash=data["anchor_raw_hash"],
            anchor_normalized_hash=data["anchor_normalized_hash"],
            anchor_step=data["anchor_step"],
            recording_id=data["recording_id"],
            target_class=data["target_class"],
            competitor_classes=tuple(data["competitor_classes"]),
            domain=CandidateBox.from_dict(data["domain"]),
            lower_margins=tuple(data["lower_margins"]),
            max_age=data["max_age"],
            method_id=data["method_id"],
            numerical_status=data["numerical_status"],
            lifecycle=data["lifecycle"],
            formal_reuse_enabled=data["formal_reuse_enabled"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# RuntimeDecision
# ==============================================================================

@dataclass(frozen=True)
class RuntimeDecision:
    """Decision made at runtime for a given streaming query."""
    query_row_id: int
    recording_id: str
    step_idx: int
    action: str
    predicted_class: int
    certificate_id: Optional[str]
    elapsed_ms: float
    reason: str
    is_reuse: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {self.schema_version}")
        valid_actions = {a.value for a in DecisionAction}
        if self.action not in valid_actions:
            raise ValueError(f"Unknown action {self.action}, must be in {valid_actions}")
        if isinstance(self.predicted_class, (bool, np.bool_)) or self.predicted_class not in range(12):
            raise ValueError(f"predicted_class must be in 0..11, got {self.predicted_class}")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        if not isinstance(self.is_reuse, bool):
            raise ValueError("is_reuse must be boolean")
        if self.is_reuse and self.action != DecisionAction.REUSE_CERTIFIED.value:
            raise ValueError("is_reuse can only be True when action is REUSE_CERTIFIED")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_row_id": self.query_row_id,
            "recording_id": self.recording_id,
            "step_idx": self.step_idx,
            "action": self.action,
            "predicted_class": self.predicted_class,
            "certificate_id": self.certificate_id,
            "elapsed_ms": self.elapsed_ms,
            "reason": self.reason,
            "is_reuse": self.is_reuse,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeDecision:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SerializationError(f"Unsupported schema_version {data.get('schema_version')}")
        return cls(
            query_row_id=data["query_row_id"],
            recording_id=data["recording_id"],
            step_idx=data["step_idx"],
            action=data["action"],
            predicted_class=data["predicted_class"],
            certificate_id=data.get("certificate_id"),
            elapsed_ms=data["elapsed_ms"],
            reason=data["reason"],
            is_reuse=data["is_reuse"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


# ==============================================================================
# Strict Serialization Utilities
# ==============================================================================

def to_json_str(obj: Any) -> str:
    """Serialize dataclass to JSON string rejecting NaN and Inf."""
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif isinstance(obj, dict):
        data = obj
    else:
        data = asdict(obj)
    return json.dumps(data, sort_keys=True, allow_nan=False, indent=2)


def from_json_str(cls: Type[T], s: str) -> T:
    """Deserialize JSON string into cls enforcing schema_version check."""
    data = json.loads(s)
    if not isinstance(data, dict):
        raise SerializationError(f"Expected JSON object, got {type(data)}")
    if hasattr(cls, "from_dict"):
        return cls.from_dict(data)
    return cls(**data)
