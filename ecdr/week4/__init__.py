"""ecdr.week4 — Week 4 formal certification, numerics, and runtime reuse package.

Frozen reference smallcnn1d (K=12), formal interval contracts, and execution types.
"""

from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    Certificate,
    DecisionAction,
    ExecutionMode,
    MembershipResult,
    MembershipStatus,
    NumericalCapability,
    NumericalStatus,
    QueryView,
    ReferenceIdentity,
    RuntimeDecision,
    SCHEMA_VERSION,
    TrustedAnchorReceipt,
)
from ecdr.week4.reference import (
    FrozenReferenceAdapter,
    IdentityError,
    ReferenceOwner,
    UnsupportedGraphError,
)

__all__ = [
    "SCHEMA_VERSION",
    "QueryView",
    "ReferenceIdentity",
    "TrustedAnchorReceipt",
    "MembershipResult",
    "MembershipStatus",
    "BoundResult",
    "BoundStatus",
    "NumericalCapability",
    "NumericalStatus",
    "Certificate",
    "RuntimeDecision",
    "DecisionAction",
    "ExecutionMode",
    "ReferenceOwner",
    "FrozenReferenceAdapter",
    "IdentityError",
    "UnsupportedGraphError",
    "CandidateBox",
]
