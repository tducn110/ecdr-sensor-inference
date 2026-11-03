"""Week 3 actual-graph K=12 box diagnostics and a fail-closed runtime gate.

The interval formulas are justified in real arithmetic.  Their implementation
uses ordinary float64 PyTorch operations, *not* directed-rounding arithmetic or
a proved enclosure of the frozen float32 backend.  Consequently a positive
result is CONDITIONAL evidence only.  It can never authorize runtime reuse.
No tolerance/epsilon is used as a substitute for a numerical soundness proof.

This module deliberately does not use the historical valid-convolution/GAP
``ecdr.model`` or ``ecdr.bounds`` implementation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from ecdr.schema import CLASS_NAMES_12
from scripts.backbone import SmallCNN1D


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "output/week1_closeout_20261102/reference"
CHECKPOINT_SHA256 = "8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1"
PREPROCESSING_ID = "aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e"
BOUND_METHOD = "actual-smallcnn1d-direct-margin-ibp-float64-v1"
NUMERICAL_STATUS = "NOT ESTABLISHED"
NUMERICAL_ASSUMPTIONS = (
    "real-arithmetic interval formulas evaluated exactly or with validated containment",
    "bounds also enclose frozen float32 normalization and reference-backend evaluation",
)


class IdentityError(ValueError):
    """The executable reference no longer matches the frozen identity."""


class UnsupportedGraphError(ValueError):
    """A graph/operator is outside this verifier's explicitly supported scope."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def preprocessing_id(mean: np.ndarray, std: np.ndarray) -> str:
    """Exactly the Week 1 ID algorithm, including float32 byte representation."""
    return hashlib.sha256(
        b"norm:train_only:per_channel:axes_0_2:clip_1e-6:float32:"
        + mean.tobytes() + std.tobytes()
    ).hexdigest()


def _window(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (6, 128) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError("window must have shape (6,128) and a real numeric dtype")
    if np.iscomplexobj(arr) or not np.all(np.isfinite(arr)):
        raise ValueError("window must contain finite real values")
    # A finite float64 value can overflow when converted to the frozen dtype.
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(arr, dtype=dtype)
    if not np.all(np.isfinite(converted)):
        raise ValueError("window is not finite after input dtype conversion")
    return converted


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def _graph_signature(model: nn.Module) -> str:
    if type(model) is not SmallCNN1D:
        raise UnsupportedGraphError("only the actual SmallCNN1D forward graph is supported")
    if getattr(model.forward, "__func__", None) is not SmallCNN1D.forward:
        raise UnsupportedGraphError("SmallCNN1D.forward has been replaced")
    # Output-changing hooks and instance forward overrides are executable graph
    # changes even when parameter bytes and nn.Module repr stay unchanged.
    global_module = torch.nn.modules.module
    if (global_module._global_forward_hooks or global_module._global_forward_pre_hooks):
        raise UnsupportedGraphError("global forward hooks invalidate frozen graph identity")
    forwards = []
    for name, module in model.named_modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise UnsupportedGraphError("forward hooks invalidate frozen graph identity")
        method = getattr(module.forward, "__func__", None)
        if method is not type(module).forward or "_call_impl" in module.__dict__:
            raise UnsupportedGraphError("module forward/call implementation has been replaced")
        code = getattr(method, "__code__", None)
        if code is None:
            raise UnsupportedGraphError("unrecognized module forward implementation")
        forwards.append((name, hashlib.sha256(marshal.dumps(code)).hexdigest()))
    return _digest_json({
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "forward_source": inspect.getsource(SmallCNN1D.forward),
        "modules": [(name, repr(module)) for name, module in model.named_modules()],
        "forward_implementations": forwards,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "torch.float32",
    })


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceIdentity:
    checkpoint_sha256: str
    preprocessing_id: str
    class_order: tuple[str, ...]
    graph_id: str
    state_id: str
    torch_version: str
    numpy_version: str
    dtype: str = "float32"
    device: str = "cpu"

    @property
    def identity_id(self) -> str:
        return _digest_json(asdict(self))


class FrozenK12Reference:
    """Load the exact selected checkpoint and reproduce frozen numpy float32 N.

    In-memory state, buffers, preprocessing, class order, eval state, graph and
    device are revalidated at inference/build boundaries.  A changed executable
    is rejected: it cannot serve as its own supposedly frozen fallback.
    """

    @classmethod
    def load(cls, reference_dir: Path | str = REFERENCE_DIR) -> "FrozenK12Reference":
        directory = Path(reference_dir)
        checkpoint_path = directory / "selected_best.pt"
        if _sha_file(checkpoint_path) != CHECKPOINT_SHA256:
            raise IdentityError("selected checkpoint SHA-256 mismatch")
        config = json.loads((directory / "config.json").read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        with np.load(directory / "normalization.npz") as data:
            mean, std = data["mean"].copy(), data["std"].copy()
        if mean.dtype != np.float32 or std.dtype != np.float32:
            raise IdentityError("frozen normalization must be float32")
        if mean.shape != (6,) or std.shape != (6,) or not np.all(np.isfinite(mean)):
            raise IdentityError("invalid normalization dimensions or mean")
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise IdentityError("normalization std must be finite and positive")
        prep_id = preprocessing_id(mean, std)
        if prep_id != PREPROCESSING_ID or checkpoint["preprocessing_id"] != prep_id:
            raise IdentityError("frozen preprocessing ID mismatch")
        if not np.array_equal(mean, checkpoint["norm_mean"]) or not np.array_equal(std, checkpoint["norm_std"]):
            raise IdentityError("checkpoint and normalization arrays disagree")
        if _sha_file(directory / "normalization.npz") != checkpoint["norm_sha256"]:
            raise IdentityError("normalization artifact SHA-256 mismatch")
        classes = tuple(checkpoint["class_names"])
        if classes != tuple(CLASS_NAMES_12) or classes != tuple(config["class_order"]):
            raise IdentityError("full K=12 class order mismatch")
        if checkpoint["num_classes"] != 12 or checkpoint["model_cfg"] != config["model_cfg"]:
            raise IdentityError("K=12 model configuration mismatch")
        model = SmallCNN1D(num_classes=12, cfg=checkpoint["model_cfg"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        if sum(p.numel() for p in model.parameters()) != 283692:
            raise IdentityError("actual frozen K=12 parameter count mismatch")
        result = cls.__new__(cls)
        result.model = model
        result.mean, result.std = mean, std
        result.class_order = classes
        result.identity = ReferenceIdentity(
            CHECKPOINT_SHA256, prep_id, classes, _graph_signature(model),
            _state_digest(model), str(torch.__version__), np.__version__,
        )
        result.reference_dir = directory.resolve()
        result.assert_identity()
        return result

    def assert_identity(self) -> None:
        if tuple(self.class_order) != self.identity.class_order:
            raise IdentityError("runtime class order changed")
        if self.mean.dtype != np.float32 or self.std.dtype != np.float32:
            raise IdentityError("runtime preprocessing dtype changed")
        if preprocessing_id(self.mean, self.std) != self.identity.preprocessing_id:
            raise IdentityError("runtime preprocessing changed")
        if any(m.training for m in self.model.modules()):
            raise IdentityError("reference must remain in evaluation mode")
        if any(t.device.type != "cpu" for t in self.model.state_dict().values()):
            raise IdentityError("reference device changed")
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise IdentityError("reference model dtype changed")
        try:
            graph_id = _graph_signature(self.model)
        except UnsupportedGraphError as error:
            raise IdentityError(str(error)) from error
        if graph_id != self.identity.graph_id or _state_digest(self.model) != self.identity.state_id:
            raise IdentityError("reference graph, weights, or buffers changed")

    def normalize(self, raw: Any) -> np.ndarray:
        x = _window(raw)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("frozen normalization produced non-finite input")
        return np.ascontiguousarray(normalized)

    def infer(self, raw: Any) -> tuple[np.ndarray, float]:
        """Return reference logits and complete identity+normalization+forward ms."""
        started = time.perf_counter()
        self.assert_identity()
        normalized = self.normalize(raw)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(normalized[None])).numpy()[0].copy()
        if logits.shape != (12,) or not np.all(np.isfinite(logits)):
            raise ValueError("reference produced non-finite or malformed logits")
        return logits, (time.perf_counter() - started) * 1000.0


def _radius(radius: Any, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(radius, dtype=np.float64)
    if value.shape == (6,):
        value = value[:, None]
    try:
        value = np.array(np.broadcast_to(value, shape), copy=True)
    except ValueError as error:
        raise ValueError(f"radius must broadcast to {shape}") from error
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("radius must be finite and non-negative")
    return value


@dataclass(frozen=True)
class CandidateBox:
    """A stored float64 endpoint box; no implicit comparison tolerance.

    Endpoints (not an unrounded center/radius expression) define the domain.
    Tuple storage prevents array aliases from mutating issued certificates.
    The wave domain fixes the new window's first half to the trusted anchor's
    second half and varies only its 384-coordinate suffix.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    kind: str
    free_dimensions: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", tuple(float(x) for x in self.lower))
        object.__setattr__(self, "upper", tuple(float(x) for x in self.upper))
        lo, hi = self.arrays()
        if self.kind not in ("diagonal_box", "fixed_overlap_wave"):
            raise ValueError("unsupported candidate domain kind")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo > hi):
            raise ValueError("domain endpoints must be finite and ordered")
        if self.free_dimensions != int(np.count_nonzero(lo != hi)):
            raise ValueError("free-dimension count does not match endpoints")
        if self.kind == "fixed_overlap_wave" and not np.array_equal(lo[:, :64], hi[:, :64]):
            raise ValueError("wave overlap coordinates must be fixed exactly")

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.lower) != 768 or len(self.upper) != 768:
            raise ValueError("candidate must have 768 coordinates")
        return np.asarray(self.lower).reshape(6, 128), np.asarray(self.upper).reshape(6, 128)

    @property
    def domain_id(self) -> str:
        return _digest_json(asdict(self))

    def contains(self, normalized_query: Any) -> bool:
        """Compare exactly represented finite float32 queries to stored endpoints."""
        z = _window(normalized_query, dtype=np.float64)
        lo, hi = self.arrays()
        return bool(np.all(z >= lo) and np.all(z <= hi))


def diagonal_box(normalized_anchor: Any, radius: Any) -> CandidateBox:
    z = _window(normalized_anchor, dtype=np.float64)
    budget = _radius(radius, (6, 128))
    lo, hi = z - budget, z + budget
    return CandidateBox(tuple(lo.ravel()), tuple(hi.ravel()), "diagonal_box", int(np.count_nonzero(lo != hi)))


def fixed_overlap_wave(normalized_anchor: Any, radius: Any) -> CandidateBox:
    z = _window(normalized_anchor, dtype=np.float64)
    # Constant continuation is a declared candidate selector, not a future oracle.
    center = np.concatenate((z[:, 64:], z[:, 64:]), axis=1)
    budget = _radius(radius, (6, 64))
    lo, hi = center.copy(), center.copy()
    lo[:, 64:] -= budget
    hi[:, 64:] += budget
    return CandidateBox(tuple(lo.ravel()), tuple(hi.ravel()), "fixed_overlap_wave", int(np.count_nonzero(lo != hi)))


def interval_layer(layer: nn.Module, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional real-arithmetic IBP transfer for explicitly supported ops.

    The code performs finite float64 evaluation without directed rounding.
    BatchNorm folding is likewise a finite computation, not exact coefficients.
    """
    if lower.dtype != torch.float64 or upper.dtype != torch.float64:
        raise ValueError("bound tensors must be float64")
    if lower.shape != upper.shape or not bool(torch.all(lower <= upper)):
        raise ValueError("malformed input interval")
    if not bool(torch.all(torch.isfinite(lower))) or not bool(torch.all(torch.isfinite(upper))):
        raise ValueError("non-finite input interval")
    if type(layer) is nn.Conv1d:
        if layer.padding_mode != "zeros":
            raise UnsupportedGraphError("only zero-padded Conv1d is supported")
        pos, neg = layer.weight.clamp(min=0), layer.weight.clamp(max=0)
        kwargs = {"stride": layer.stride, "padding": layer.padding, "dilation": layer.dilation, "groups": layer.groups}
        lo = functional.conv1d(lower, pos, layer.bias, **kwargs) + functional.conv1d(upper, neg, None, **kwargs)
        hi = functional.conv1d(upper, pos, layer.bias, **kwargs) + functional.conv1d(lower, neg, None, **kwargs)
    elif type(layer) is nn.BatchNorm1d:
        if layer.training or not layer.track_running_stats or layer.running_mean is None or layer.running_var is None:
            raise UnsupportedGraphError("BatchNorm requires frozen evaluation running statistics")
        if bool(torch.any(layer.running_var < 0)) or layer.eps <= 0:
            raise ValueError("invalid BatchNorm variance/epsilon")
        scale = (layer.weight if layer.affine else torch.ones_like(layer.running_mean)) / torch.sqrt(layer.running_var + layer.eps)
        bias = (layer.bias if layer.affine else torch.zeros_like(layer.running_mean)) - scale * layer.running_mean
        scale, bias = scale[None, :, None], bias[None, :, None]
        a, b = scale * lower + bias, scale * upper + bias
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    elif type(layer) is nn.ReLU:
        lo, hi = lower.clamp(min=0), upper.clamp(min=0)
    elif type(layer) is nn.MaxPool1d:
        if layer.return_indices:
            raise UnsupportedGraphError("MaxPool index outputs are unsupported")
        kwargs = {"kernel_size": layer.kernel_size, "stride": layer.stride, "padding": layer.padding,
                  "dilation": layer.dilation, "ceil_mode": layer.ceil_mode}
        lo, hi = functional.max_pool1d(lower, **kwargs), functional.max_pool1d(upper, **kwargs)
    elif type(layer) is nn.Flatten:
        lo, hi = layer(lower), layer(upper)
    elif type(layer) is nn.Linear:
        pos, neg = layer.weight.clamp(min=0), layer.weight.clamp(max=0)
        lo = functional.linear(lower, pos, layer.bias) + functional.linear(upper, neg)
        hi = functional.linear(upper, pos, layer.bias) + functional.linear(lower, neg)
    else:
        raise UnsupportedGraphError(f"unsupported operator: {type(layer).__name__}")
    if not bool(torch.all(torch.isfinite(lo))) or not bool(torch.all(torch.isfinite(hi))) or not bool(torch.all(lo <= hi)):
        raise ValueError("non-finite or inverted propagated interval")
    return lo, hi


@dataclass(frozen=True)
class ConditionalCertificate:
    reference_identity_id: str
    checkpoint_sha256: str
    preprocessing_id: str
    class_order: tuple[str, ...]
    graph_id: str
    anchor_raw_id: str
    anchor_normalized_id: str
    anchor_logits: tuple[float, ...]
    anchor_step: int
    recording_id: str
    target_class: int
    competitors: tuple[int, ...]
    domain: CandidateBox
    lower_margins: tuple[float, ...]
    max_age: int
    method: str = BOUND_METHOD
    numerical_status: str = NUMERICAL_STATUS
    numerical_assumptions: tuple[str, ...] = NUMERICAL_ASSUMPTIONS
    lifecycle: str = "CONDITIONAL_ONLY"

    @property
    def certificate_id(self) -> str:
        return _digest_json(asdict(self))

    @property
    def formal_reuse_enabled(self) -> bool:
        # No constructor flag or receipt can bypass the absent numerical proof.
        return False


@dataclass(frozen=True)
class BuildResult:
    status: str
    reason: str
    conditional_positive: bool
    target_class: int
    competitors: tuple[int, ...]
    lower_margins: tuple[float, ...]
    separate_lower_margins: tuple[float, ...]
    certificate: ConditionalCertificate | None
    domain: CandidateBox | None
    reference_ms: float
    bound_ms: float
    total_ms: float
    numerical_status: str = NUMERICAL_STATUS
    formal_reuse_enabled: bool = False

    @property
    def min_lower_margin(self) -> float | None:
        return min(self.lower_margins) if self.lower_margins else None


class ActualGraphIBP:
    """Direct target-minus-all-eleven-final-layer-specifications IBP."""

    def __init__(self, reference: FrozenK12Reference):
        reference.assert_identity()
        self.reference = reference
        self._model = copy.deepcopy(reference.model).double().eval()
        self._snapshot_id = _state_digest(self._model)
        self._graph_id = _graph_signature(self._model)

    def bounds(self, domain: CandidateBox, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Return direct margin estimates and separate-logit baseline estimates.

        This method does not declare either vector a validated enclosure.
        """
        if (not isinstance(target_class, (int, np.integer)) or
            isinstance(target_class, (bool, np.bool_)) or not 0 <= target_class < 12):
            raise ValueError("target class must be in 0..11")
        if _state_digest(self._model) != self._snapshot_id or _graph_signature(self._model) != self._graph_id:
            raise IdentityError("bound-model snapshot changed")
        if any(m.training for m in self._model.modules()):
            raise IdentityError("bound model must remain in evaluation mode")
        lo, hi = domain.arrays()
        lower, upper = torch.from_numpy(lo[None].copy()), torch.from_numpy(hi[None].copy())
        competitors = [j for j in range(12) if j != target_class]
        with torch.inference_mode():
            for layer in list(self._model.conv) + list(self._model.head)[:-1]:
                lower, upper = interval_layer(layer, lower, upper)
            final = self._model.head[-1]
            if type(final) is not nn.Linear or final.out_features != 12:
                raise UnsupportedGraphError("final head must be Linear(128,12)")
            weights = final.weight[target_class][None] - final.weight[competitors]
            bias = final.bias[target_class] - final.bias[competitors]
            direct = functional.linear(lower, weights.clamp(min=0), bias) + functional.linear(upper, weights.clamp(max=0))
            logits_lo, logits_hi = interval_layer(final, lower, upper)
            separate = logits_lo[:, target_class, None] - logits_hi[:, competitors]
        direct_array, separate_array = direct.numpy()[0].copy(), separate.numpy()[0].copy()
        if direct_array.shape != (11,) or not np.all(np.isfinite(direct_array)):
            raise ValueError("invalid direct eleven-margin estimate")
        return direct_array, separate_array

    def build(self, anchor_raw: Any, radius: Any, *, recording_id: str, step_idx: int,
              max_age: int = 128, domain_kind: str = "diagonal_box") -> BuildResult:
        started = time.perf_counter()
        _validate_location(recording_id, step_idx)
        step_idx = int(step_idx)
        if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age < 1:
            raise ValueError("max_age must be a positive integer")
        raw = _window(anchor_raw)
        logits, reference_ms = self.reference.infer(raw)
        normalized = self.reference.normalize(raw)
        winner = int(np.argmax(logits))  # exactly reference tie-breaking
        competitors = tuple(j for j in range(12) if j != winner)
        if domain_kind == "diagonal_box":
            domain = diagonal_box(normalized, radius)
        elif domain_kind == "fixed_overlap_wave":
            domain = fixed_overlap_wave(normalized, radius)
        else:
            raise ValueError("unsupported domain kind")
        bound_started = time.perf_counter()
        try:
            direct, separate = self.bounds(domain, winner)
        except (ValueError, RuntimeError) as error:
            return BuildResult("FAILED", str(error), False, winner, competitors, (), (), None,
                               domain, reference_ms, (time.perf_counter()-bound_started)*1000,
                               (time.perf_counter()-started)*1000)
        bound_ms = (time.perf_counter() - bound_started) * 1000
        # Strict anchor margin also excludes ties, including degenerate boxes.
        strict_anchor = bool(np.all(logits[winner] > logits[list(competitors)]))
        positive = strict_anchor and bool(np.all(direct > 0))
        certificate = None
        if positive:
            identity = self.reference.identity
            certificate = ConditionalCertificate(
                identity.identity_id, identity.checkpoint_sha256, identity.preprocessing_id,
                identity.class_order, identity.graph_id, _array_hash(raw),
                _array_hash(normalized), tuple(float(x) for x in logits), step_idx,
                recording_id, winner, competitors, domain, tuple(float(x) for x in direct), max_age,
            )
        reason = "numerical_guarantee_unavailable" if positive else "anchor_tie" if not strict_anchor else "nonpositive_margin_estimate"
        return BuildResult("CONDITIONAL_POSITIVE" if positive else "INCONCLUSIVE", reason,
                           positive, winner, competitors, tuple(direct), tuple(separate),
                           certificate, domain, reference_ms, bound_ms,
                           (time.perf_counter()-started)*1000)


def _validate_location(recording_id: str, step_idx: int) -> None:
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("recording_id must be a nonempty string")
    if not isinstance(step_idx, (int, np.integer)) or isinstance(step_idx, (bool, np.bool_)) or step_idx < 0:
        raise ValueError("step_idx must be a nonnegative integer")


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    is_member: bool
    reason: str
    certificate_age: int | None
    elapsed_ms: float


def check_certificate(reference: FrozenK12Reference, certificate: ConditionalCertificate,
                      raw_query: Any, *, recording_id: str, step_idx: int,
                      expected_anchor_raw: Any | None = None) -> GateResult:
    """Lifecycle+identity+membership gate; final numerical gate always fails shut."""
    started = time.perf_counter()
    _validate_location(recording_id, step_idx)
    step_idx = int(step_idx)
    reference.assert_identity()
    normalized = reference.normalize(raw_query)
    age = step_idx - certificate.anchor_step
    reason = "numerical_guarantee_unavailable"
    member = False
    identity = reference.identity
    if (certificate.reference_identity_id != identity.identity_id or
        certificate.checkpoint_sha256 != identity.checkpoint_sha256 or
        certificate.preprocessing_id != identity.preprocessing_id or
        certificate.class_order != identity.class_order or certificate.graph_id != identity.graph_id):
        reason = "identity_mismatch"
    elif certificate.lifecycle != "CONDITIONAL_ONLY" or certificate.method != BOUND_METHOD:
        reason = "invalid_certificate_state_or_method"
    elif certificate.recording_id != recording_id:
        reason = "recording_reset"
    elif certificate.max_age < 1 or age <= 0 or age > certificate.max_age:
        reason = "stale_or_expired_anchor"
    elif (isinstance(certificate.target_class, (bool, np.bool_)) or
          certificate.target_class not in range(12) or
          certificate.competitors != tuple(j for j in range(12) if j != certificate.target_class) or
          len(certificate.lower_margins) != 11 or
          not np.all(np.isfinite(certificate.lower_margins)) or
          not np.all(np.asarray(certificate.lower_margins) > 0)):
        reason = "invalid_margin_specification"
    elif expected_anchor_raw is not None and _array_hash(_window(expected_anchor_raw)) != certificate.anchor_raw_id:
        reason = "anchor_mismatch"
    else:
        member = certificate.domain.contains(normalized)
        if not member:
            reason = "outside_domain"
    return GateResult(False, member, reason, age, (time.perf_counter()-started)*1000)


@dataclass(frozen=True)
class RuntimeDecision:
    pred_class: int
    logits: tuple[float, ...]
    is_reuse: bool
    reason: str
    is_member: bool
    certificate_age: int | None
    reference_ms: float
    membership_ms: float
    build_ms: float
    total_ms: float
    build_result: BuildResult | None


class RuntimeGate:
    """Recording-aware lifecycle retaining a fixed trusted conditional anchor.

    Reused queries are never anchors.  In this implementation no query can be
    formally reused.  A successful diagnostic build stays bound to its original
    reference anchor until expiry/reset; inconclusive builds are not cached.
    All valid queries execute the frozen reference; invalid input is rejected.
    """

    def __init__(self, reference: FrozenK12Reference, verifier: ActualGraphIBP | None = None):
        self.reference = reference
        self.verifier = verifier if verifier is not None else ActualGraphIBP(reference)
        if self.verifier.reference is not reference:
            raise IdentityError("runtime verifier belongs to a different reference instance")
        self.reset()

    def reset(self) -> None:
        self.recording_id: str | None = None
        self.last_step: int | None = None
        self.certificate: ConditionalCertificate | None = None

    def predict(self, raw: Any, *, recording_id: str, step_idx: int,
                radius: Any | None = None, domain_kind: str = "diagonal_box",
                max_age: int = 128) -> RuntimeDecision:
        started = time.perf_counter()
        _validate_location(recording_id, step_idx)
        step_idx = int(step_idx)
        # Reject conversion/normalization overflow before changing lifecycle state.
        self.reference.normalize(raw)
        reset_reason = None
        if self.recording_id is not None and recording_id != self.recording_id:
            reset_reason = "recording_reset"
        elif self.last_step is not None and step_idx != self.last_step + 1:
            reset_reason = "discontinuity_reset"
        if reset_reason:
            self.certificate = None
        reason, member, age, membership_ms = reset_reason or "no_certificate", False, None, 0.0
        if self.certificate is not None:
            gate = check_certificate(self.reference, self.certificate, raw,
                                     recording_id=recording_id, step_idx=step_idx)
            reason, member, age, membership_ms = gate.reason, gate.is_member, gate.certificate_age, gate.elapsed_ms
            if reason in ("stale_or_expired_anchor", "identity_mismatch", "recording_reset"):
                self.certificate = None
        logits, reference_ms = self.reference.infer(raw)
        build = None
        if radius is not None and self.certificate is None:
            build = self.verifier.build(raw, radius, recording_id=recording_id, step_idx=step_idx,
                                        max_age=max_age, domain_kind=domain_kind)
            self.certificate = build.certificate
        self.recording_id, self.last_step = recording_id, int(step_idx)
        return RuntimeDecision(int(np.argmax(logits)), tuple(float(x) for x in logits), False,
                               reason, member, age, reference_ms, membership_ms,
                               build.total_ms if build else 0.0,
                               (time.perf_counter()-started)*1000, build)
