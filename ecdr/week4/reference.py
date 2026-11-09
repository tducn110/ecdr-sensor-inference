"""ecdr.week4.reference — Authoritative frozen reference owner and adapter.

Enforces strict K=12 SmallCNN1D reference loading, deep identity assertion
(graph signature, weights, BatchNorm buffers, evaluation mode, CPU float32),
deterministic NumPy normalization, and unforgeable TrustedAnchorReceipt generation.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import hashlib
import hmac
import inspect
import json
import marshal
from pathlib import Path
import secrets
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from ecdr.schema import CLASS_NAMES_12
from ecdr.week4.types import (
    QueryView,
    ReferenceIdentity,
    SCHEMA_VERSION,
    TrustedAnchorReceipt,
)
from scripts.backbone import SmallCNN1D

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "output/week1_closeout_20261102/reference"
CHECKPOINT_SHA256 = "8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1"
PREPROCESSING_ID = "aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e"


class IdentityError(ValueError):
    """The executable reference no longer matches the frozen identity."""


class UnsupportedGraphError(ValueError):
    """A graph or operator is outside the strictly frozen reference scope."""


class IdentityCheckMode(str, Enum):
    """Identity checking policy for ReferenceOwner.

    STRONG_PER_QUERY:
        Deep verification on every single query inference. Re-hashes all 283,692
        parameters, BatchNorm running buffers, and Python bytecode AST graph.
        Provides uncompromised zero-trust guarantees at ~1.28 ms/query cost.
        Baseline: M01 strong wrapper.

    OWNED_GENERATION:
        Full verification at initialization, checkpoint loading, and receipt
        generation salt invalidation. Between generation updates, performs fast
        shallow safety checks (eval mode, CPU float32 tensors, requires_grad=False,
        forward hooks presence) in < 0.01 ms. Relies on Python in-process memory
        encapsulation and exclusive model ownership by ReferenceOwner.
        Baseline: M16 owned-reference.

    CORE_INFERENCE:
        Direct PyTorch model inference without runtime assertion checks or wrapping.
        Baseline: M00 core inference.
    """
    STRONG_PER_QUERY = "strong_per_query"
    OWNED_GENERATION = "owned_generation"
    CORE_INFERENCE = "core_inference"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _preprocessing_id(mean: np.ndarray, std: np.ndarray) -> str:
    """Exactly the Week 1 ID algorithm, including float32 byte representation."""
    return hashlib.sha256(
        b"norm:train_only:per_channel:axes_0_2:clip_1e-6:float32:"
        + mean.tobytes()
        + std.tobytes()
    ).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(str((arr.shape, arr.dtype.str)).encode() + arr.tobytes()).hexdigest()


class NormalizationConstants:
    """Owned, read-only copy of the frozen float32 per-channel mean/std.

    The constants are verified against the Week 1 preprocessing ID at
    construction. Arrays are private copies; accessors hand out fresh read-only
    copies so callers cannot alias or mutate the frozen values.
    """

    __slots__ = ("_mean", "_std", "_preprocessing_id")

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        m = np.array(mean, dtype=np.float32, copy=True)
        s = np.array(std, dtype=np.float32, copy=True)
        if m.shape != (6,) or s.shape != (6,):
            raise ValueError(f"mean/std must have shape (6,), got {m.shape}, {s.shape}")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s))):
            raise ValueError("mean/std must be finite")
        if np.any(s <= 0.0):
            raise ValueError("std must be strictly positive (required for monotone division)")
        m.flags.writeable = False
        s.flags.writeable = False
        self._mean = m
        self._std = s
        self._preprocessing_id = _preprocessing_id(m, s)

    @property
    def preprocessing_id(self) -> str:
        return self._preprocessing_id

    @property
    def mean(self) -> np.ndarray:
        out = self._mean.copy()
        out.flags.writeable = False
        return out

    @property
    def std(self) -> np.ndarray:
        out = self._std.copy()
        out.flags.writeable = False
        return out


def execute_normalization(x32: np.ndarray, constants: NormalizationConstants) -> np.ndarray:
    """The frozen executable normalization: float32 subtract, then float32 divide.

    Single source of truth shared by ``ReferenceOwner.normalize`` and the
    enclosure below. Each step is one elementwise, correctly rounded IEEE-754
    float32 operation (no FMA, no reduction) with the same per-channel constants.
    Raises ValueError (before returning anything) on bad shape/dtype or if the
    result is non-finite (normalization overflow).
    """
    x = np.asarray(x32)
    if x.dtype != np.float32:
        raise ValueError(f"executable normalization requires float32 input, got {x.dtype}")
    if x.shape != (6, 128):
        raise ValueError(f"raw window must have shape (6, 128), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("raw window contains non-finite values")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norm = ((x - constants._mean[:, None]) / constants._std[:, None]).astype(np.float32)
    if not np.all(np.isfinite(norm)):
        raise ValueError("normalization produced non-finite values")
    out = np.ascontiguousarray(norm)
    out.flags.writeable = False
    return out


def _raw_endpoints_to_float32(name: str, values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return (original float64 view, float32 round-to-nearest conversion); reject overflow."""
    a = np.asarray(values)
    if a.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point array, got dtype {a.dtype}")
    if a.shape != (6, 128):
        raise ValueError(f"{name} must have shape (6, 128), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    a64 = a.astype(np.float64)
    with np.errstate(over="ignore"):
        a32 = a64.astype(np.float32)
    if not np.all(np.isfinite(a32)):
        raise ValueError(f"{name}: float32 conversion overflow")
    return a64, a32


def enclose_normalization(
    raw_lower: Any,
    raw_upper: Any,
    constants: NormalizationConstants,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact enclosure of the executable normalization over a raw coordinate box.

    Claim (for every raw real x with lower <= x <= upper, coordinatewise):
        N_hat(fl32(x)) is in [N_hat(fl32(lower)), N_hat(fl32(upper))]
    where N_hat(t) = fl32(fl32(t - mean_c) / std_c) is the frozen executable.

    Proof sketch: fl32 (round-to-nearest) is monotone non-decreasing; for fixed
    mean_c, t -> t - mean_c is increasing and its correctly rounded float32 value
    is monotone non-decreasing; for fixed std_c > 0, u -> u / std_c likewise.
    A composition of monotone non-decreasing maps is monotone, so the image of a
    coordinate interval lies between the images of its endpoints. No rounding
    error term is needed because the endpoints are evaluated by the *same*
    executable sequence.

    Premises (not proved here, listed in the receipt): the executing NumPy build
    implements float32 subtract/divide as correctly rounded elementwise IEEE-754
    operations (no fusion, no flush-to-zero change of ordering); constants are
    the verified frozen values; std > 0. Raw-conversion overflow, inverted
    endpoints, bad shape/dtype and non-finite normalized endpoints raise
    ValueError (fail closed). The returned arrays are read-only float32.
    """
    lo64, lo32 = _raw_endpoints_to_float32("raw_lower", raw_lower)
    hi64, hi32 = _raw_endpoints_to_float32("raw_upper", raw_upper)
    if np.any(lo64 > hi64):
        raise ValueError("raw_lower must be <= raw_upper coordinatewise")
    n_lo = execute_normalization(lo32, constants)
    n_hi = execute_normalization(hi32, constants)
    if np.any(n_lo > n_hi):  # impossible if the monotonicity premise holds
        raise ValueError("monotone normalization premise violated by this backend")
    return n_lo, n_hi


def _graph_signature(model: nn.Module) -> str:
    if type(model) is not SmallCNN1D:
        raise UnsupportedGraphError("only the actual SmallCNN1D forward graph is supported")
    if getattr(model.forward, "__func__", None) is not SmallCNN1D.forward:
        raise UnsupportedGraphError("SmallCNN1D.forward has been replaced")

    # Global hooks on torch.nn.modules.module invalidate graph identity
    global_module = torch.nn.modules.module
    if getattr(global_module, "_global_forward_hooks", None) or getattr(
        global_module, "_global_forward_pre_hooks", None
    ):
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
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "torch.float32",
    })


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bn_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.running_mean is not None:
                digest.update(f"{name}.running_mean".encode() + module.running_mean.detach().cpu().numpy().tobytes())
            if module.running_var is not None:
                digest.update(f"{name}.running_var".encode() + module.running_var.detach().cpu().numpy().tobytes())
            digest.update(f"{name}.eps:{module.eps}".encode())
    return digest.hexdigest()


class ReferenceOwner:
    """Exclusive owner of the executable frozen K=12 SmallCNN1D reference model.

    Responsibilities:
    - Verifies byte hashes of checkpoint and normalization before instantiation.
    - Locks model into eval mode with CPU float32 tensors and zero gradients.
    - Asserts graph, weight, BN buffers, and runtime environment identity.
    - Normalizes raw input windows deterministically.
    - Issues cryptographically authenticated TrustedAnchorReceipt instances.
    - Does NOT expose mutable aliases of its internal state.
    """

    def __init__(
        self,
        model: SmallCNN1D,
        mean: np.ndarray,
        std: np.ndarray,
        class_order: Tuple[str, ...],
        reference_dir: Path,
    ) -> None:
        self._model = model
        self._mean = np.ascontiguousarray(mean, dtype=np.float32)
        self._std = np.ascontiguousarray(std, dtype=np.float32)
        self._class_order = tuple(class_order)
        self._reference_dir = reference_dir.resolve()

        # Secret session salt for cryptographic anchor receipts
        self._secret_salt = secrets.token_bytes(32)
        self._generation_id = 1

        # Execution identity check mode (W4-024)
        self._identity_mode: IdentityCheckMode = IdentityCheckMode.STRONG_PER_QUERY
        self._forward_calls_count: int = 0
        self._cached_norm_constants: Optional[NormalizationConstants] = None

        # Capture frozen identity
        self._identity = ReferenceIdentity(
            checkpoint_sha256=CHECKPOINT_SHA256,
            preprocessing_id=_preprocessing_id(self._mean, self._std),
            class_order=self._class_order,
            graph_id=_graph_signature(self._model),
            weights_id=_state_digest(self._model),
            bn_id=_bn_digest(self._model),
            torch_version=str(torch.__version__),
            numpy_version=str(np.__version__),
            dtype="float32",
            device="cpu",
            schema_version=SCHEMA_VERSION,
        )
        self.assert_identity()

    @classmethod
    def load(cls, reference_dir: Union[Path, str] = REFERENCE_DIR) -> ReferenceOwner:
        """Load frozen reference with full SHA-256 and parameter checks."""
        dir_path = Path(reference_dir)
        checkpoint_path = dir_path / "selected_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        if _sha_file(checkpoint_path) != CHECKPOINT_SHA256:
            raise IdentityError(f"Checkpoint SHA-256 mismatch for {checkpoint_path}")

        config_path = dir_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        norm_path = dir_path / "normalization.npz"
        if _sha_file(norm_path) != checkpoint.get("norm_sha256"):
            raise IdentityError("Normalization artifact SHA-256 does not match checkpoint recorded sha256")

        with np.load(norm_path) as data:
            mean = data["mean"].copy()
            std = data["std"].copy()

        if mean.dtype != np.float32 or std.dtype != np.float32:
            raise IdentityError("Normalization mean/std must be float32")
        if mean.shape != (6,) or std.shape != (6,):
            raise IdentityError(f"Normalization shape mismatch: expected (6,), got mean={mean.shape}, std={std.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise IdentityError("Normalization parameters must be finite with strictly positive std")

        prep_id = _preprocessing_id(mean, std)
        if prep_id != PREPROCESSING_ID or checkpoint.get("preprocessing_id") != prep_id:
            raise IdentityError("Preprocessing ID mismatch")

        if not np.array_equal(mean, checkpoint["norm_mean"]) or not np.array_equal(std, checkpoint["norm_std"]):
            raise IdentityError("Normalization arrays disagree with checkpoint embedded arrays")

        classes = tuple(checkpoint["class_names"])
        if classes != tuple(CLASS_NAMES_12) or classes != tuple(config["class_order"]):
            raise IdentityError("Class order mismatch against canonical 12-class schema")

        if checkpoint.get("num_classes") != 12:
            raise IdentityError(f"Expected 12 classes, got {checkpoint.get('num_classes')}")

        model_cfg = checkpoint.get("model_cfg", config.get("model_cfg"))
        model = SmallCNN1D(num_classes=12, cfg=model_cfg)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        model.eval().requires_grad_(False)

        param_count = sum(p.numel() for p in model.parameters())
        if param_count != 283692:
            raise IdentityError(f"Parameter count mismatch: expected 283,692, got {param_count}")

        return cls(
            model=model,
            mean=mean,
            std=std,
            class_order=classes,
            reference_dir=dir_path,
        )

    @property
    def identity(self) -> ReferenceIdentity:
        return self._identity

    @property
    def generation_id(self) -> int:
        return self._generation_id

    @property
    def identity_mode(self) -> IdentityCheckMode:
        return self._identity_mode

    def set_identity_mode(self, mode: Union[IdentityCheckMode, str]) -> None:
        """Configure runtime identity assertion policy (W4-024-C03)."""
        if isinstance(mode, str):
            try:
                mode = IdentityCheckMode(mode)
            except ValueError:
                raise ValueError(
                    f"Invalid identity check mode: {mode}. Must be one of {[m.value for m in IdentityCheckMode]}"
                )
        self._identity_mode = mode

    @property
    def forward_calls_count(self) -> int:
        """Total actual forward inferences executed by this reference owner (W4-024-C01)."""
        return self._forward_calls_count

    def reset_forward_calls_count(self) -> None:
        self._forward_calls_count = 0

    def _fast_identity_check(self) -> None:
        """Fast shallow identity verification (< 0.01 ms) for OWNED_GENERATION mode (W4-024-C03).

        Verifies operational runtime invariants without computing expensive
        full-parameter and bytecode SHA-256 digests:
        - Class order tuple immutability
        - Preprocessing arrays dtypes and shapes
        - Model evaluation mode (m.training == False)
        - Tensors remain on CPU with float32 dtype and requires_grad=False
        - Absence of injected forward / pre-forward hooks
        """
        if self._class_order != self._identity.class_order:
            raise IdentityError("Class order changed in memory")
        if self._mean.dtype != np.float32 or self._std.dtype != np.float32:
            raise IdentityError("Preprocessing dtype changed")
        if self._mean.shape != (6,) or self._std.shape != (6,):
            raise IdentityError("Preprocessing shape changed")
        if any(m.training for m in self._model.modules()):
            raise IdentityError("Model must remain in eval mode")
        if any(p.requires_grad for p in self._model.parameters()):
            raise IdentityError("Model parameters must have requires_grad=False")
        if any(t.device.type != "cpu" for t in self._model.state_dict().values()):
            raise IdentityError("Model tensors moved off CPU")
        if any(p.dtype != torch.float32 for p in self._model.parameters()):
            raise IdentityError("Model parameter dtype changed from float32")

        # Global hooks on torch.nn.modules.module
        global_module = torch.nn.modules.module
        if getattr(global_module, "_global_forward_hooks", None) or getattr(
            global_module, "_global_forward_pre_hooks", None
        ):
            raise UnsupportedGraphError("global forward hooks invalidate frozen graph identity")

        for name, module in self._model.named_modules():
            if module._forward_hooks or module._forward_pre_hooks:
                raise UnsupportedGraphError("forward hooks invalidate frozen graph identity")
            if getattr(module.forward, "__func__", None) is not type(module).forward:
                raise UnsupportedGraphError("module forward implementation has been replaced")

    def assert_identity(self) -> None:
        """Thorough verification of execution mode, weights, graph, and buffers."""
        if self._class_order != self._identity.class_order:
            raise IdentityError("Class order changed in memory")
        if self._mean.dtype != np.float32 or self._std.dtype != np.float32:
            raise IdentityError("Preprocessing dtype changed")
        if _preprocessing_id(self._mean, self._std) != self._identity.preprocessing_id:
            raise IdentityError("Preprocessing arrays mutated")
        if any(m.training for m in self._model.modules()):
            raise IdentityError("Model must remain in eval mode")
        if any(p.requires_grad for p in self._model.parameters()):
            raise IdentityError("Model parameters must have requires_grad=False")
        if any(t.device.type != "cpu" for t in self._model.state_dict().values()):
            raise IdentityError("Model tensors moved off CPU")
        if any(p.dtype != torch.float32 for p in self._model.parameters()):
            raise IdentityError("Model parameter dtype changed from float32")

        try:
            graph_id = _graph_signature(self._model)
        except UnsupportedGraphError as error:
            raise IdentityError(str(error)) from error
        if graph_id != self._identity.graph_id:
            raise IdentityError("Model execution graph or forward hook signature changed")

        bn_id = _bn_digest(self._model)
        if bn_id != self._identity.bn_id:
            raise IdentityError("Model BatchNorm running statistics mutated")

        weights_id = _state_digest(self._model)
        if weights_id != self._identity.weights_id:
            raise IdentityError("Model parameter weights mutated")

    def normalize(self, raw: Union[QueryView, np.ndarray, Any]) -> np.ndarray:
        """Normalize raw input window using frozen mean and std."""
        if isinstance(raw, QueryView):
            x = raw.array()
        else:
            x = np.asarray(raw, dtype=np.float32)

        if x.shape != (6, 128):
            raise ValueError(f"raw window must have shape (6, 128), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("raw window contains non-finite values")

        return execute_normalization(x, self.normalization_constants())

    def normalization_constants(self) -> NormalizationConstants:
        """Owned copy of the frozen mean/std, re-verified against the frozen preprocessing ID."""
        if (
            self._mean.dtype != np.float32
            or self._std.dtype != np.float32
            or _preprocessing_id(self._mean, self._std) != self._identity.preprocessing_id
        ):
            raise IdentityError("Preprocessing arrays mutated or changed dtype")
        if self._cached_norm_constants is None:
            self._cached_norm_constants = NormalizationConstants(self._mean, self._std)
        return self._cached_norm_constants

    def enclose_normalization(self, raw_lower: Any, raw_upper: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Enclose the frozen executable normalization over a raw coordinate box."""
        return enclose_normalization(raw_lower, raw_upper, self.normalization_constants())

    def infer(
        self,
        query: Union[QueryView, np.ndarray, Any],
        *,
        global_row_id: int = 0,
        recording_id: str = "test",
        step_idx: int = 0,
    ) -> TrustedAnchorReceipt:
        """Run complete inference and issue an authenticated TrustedAnchorReceipt."""
        started = time.perf_counter()
        self._forward_calls_count += 1

        if self._identity_mode == IdentityCheckMode.STRONG_PER_QUERY:
            self.assert_identity()
        elif self._identity_mode == IdentityCheckMode.OWNED_GENERATION:
            self._fast_identity_check()
        elif self._identity_mode == IdentityCheckMode.CORE_INFERENCE:
            pass

        if isinstance(query, QueryView):
            q_row_id = query.global_row_id
            q_rec_id = query.recording_id
            q_step = query.step_idx
            raw_hash = query.raw_hash
            raw_arr = query.array()
        else:
            raw_arr = np.asarray(query, dtype=np.float32)
            if raw_arr.shape != (6, 128) or not np.all(np.isfinite(raw_arr)):
                raise ValueError("query array must be shape (6, 128) and finite float32")
            q_row_id = int(global_row_id)
            q_rec_id = str(recording_id)
            q_step = int(step_idx)
            raw_hash = hashlib.sha256(raw_arr.tobytes()).hexdigest()

        normalized = self.normalize(raw_arr)
        norm_hash = hashlib.sha256(normalized.tobytes()).hexdigest()

        # Inference on CPU in float32
        with torch.inference_mode():
            tensor_in = torch.from_numpy(normalized[None].copy()).to(dtype=torch.float32, device="cpu")
            logits_tensor = self._model(tensor_in)[0]
            logits_arr = logits_tensor.detach().cpu().numpy().astype(np.float32)

        if logits_arr.shape != (12,) or not np.all(np.isfinite(logits_arr)):
            raise ValueError("Model produced malformed or non-finite logits")

        # NumPy argmax is exact reference tie-breaker (lowest index on equality)
        winner = int(np.argmax(logits_arr))
        competitors = [j for j in range(12) if j != winner]
        margin_to_runner_up = float(logits_arr[winner] - max(logits_arr[competitors]))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        logits_tuple = tuple(float(x) for x in logits_arr)
        logits_hash = hashlib.sha256(logits_arr.tobytes()).hexdigest()

        # Sign receipt with secret session key
        sig_payload = (
            f"{self._identity.identity_id}:{self._generation_id}:{q_row_id}:"
            f"{q_rec_id}:{q_step}:{raw_hash}:{norm_hash}:{logits_hash}:{winner}"
        ).encode("utf-8")
        signature = hmac.new(self._secret_salt, sig_payload, hashlib.sha256).hexdigest()

        return TrustedAnchorReceipt(
            reference_identity_id=self._identity.identity_id,
            checkpoint_sha256=self._identity.checkpoint_sha256,
            preprocessing_id=self._identity.preprocessing_id,
            graph_id=self._identity.graph_id,
            weights_id=self._identity.weights_id,
            query_row_id=q_row_id,
            recording_id=q_rec_id,
            step_idx=q_step,
            raw_hash=raw_hash,
            normalized_hash=norm_hash,
            logits=logits_tuple,
            winner=winner,
            margin_to_runner_up=margin_to_runner_up,
            elapsed_ms=elapsed_ms,
            issuer_signature=signature,
            generation_id=self._generation_id,
            schema_version=SCHEMA_VERSION,
        )

    def validate_receipt(
        self,
        receipt: TrustedAnchorReceipt,
        query: Union[QueryView, np.ndarray, Any],
    ) -> bool:
        """Validate provenance, query binding, and signature of a TrustedAnchorReceipt."""
        if not isinstance(receipt, TrustedAnchorReceipt):
            return False
        if receipt.schema_version != SCHEMA_VERSION:
            return False
        if receipt.reference_identity_id != self._identity.identity_id:
            return False
        if receipt.checkpoint_sha256 != self._identity.checkpoint_sha256:
            return False
        if receipt.preprocessing_id != self._identity.preprocessing_id:
            return False
        if receipt.graph_id != self._identity.graph_id:
            return False
        if receipt.weights_id != self._identity.weights_id:
            return False
        if receipt.generation_id != self._generation_id:
            return False

        # Extract query hashes
        if isinstance(query, QueryView):
            if query.global_row_id != receipt.query_row_id:
                return False
            if query.recording_id != receipt.recording_id:
                return False
            if query.step_idx != receipt.step_idx:
                return False
            raw_hash = query.raw_hash
            raw_arr = query.array()
        else:
            raw_arr = np.asarray(query, dtype=np.float32)
            if raw_arr.shape != (6, 128) or not np.all(np.isfinite(raw_arr)):
                return False
            raw_hash = hashlib.sha256(raw_arr.tobytes()).hexdigest()

        if raw_hash != receipt.raw_hash:
            return False

        try:
            norm_arr = self.normalize(raw_arr)
        except Exception:
            return False
        norm_hash = hashlib.sha256(norm_arr.tobytes()).hexdigest()
        if norm_hash != receipt.normalized_hash:
            return False

        logits_arr = np.asarray(receipt.logits, dtype=np.float32)
        logits_hash = hashlib.sha256(logits_arr.tobytes()).hexdigest()

        sig_payload = (
            f"{self._identity.identity_id}:{receipt.generation_id}:{receipt.query_row_id}:"
            f"{receipt.recording_id}:{receipt.step_idx}:{raw_hash}:{norm_hash}:{logits_hash}:{receipt.winner}"
        ).encode("utf-8")
        expected_sig = hmac.new(self._secret_salt, sig_payload, hashlib.sha256).hexdigest()

        return hmac.compare_digest(receipt.issuer_signature, expected_sig)

    def invalidate(self, reason: str = "") -> None:
        """Invalidate all previously issued receipts by rotating the session salt."""
        self._generation_id += 1
        self._secret_salt = secrets.token_bytes(32)
        # Re-assert identity on generation rotation boundary
        self.assert_identity()


# Alias for backward-compatible reference
FrozenReferenceAdapter = ReferenceOwner
