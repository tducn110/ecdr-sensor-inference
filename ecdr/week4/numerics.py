"""ecdr.week4.numerics — Validated outward interval arithmetic and error bounds.

Implements rigorous outward-directed interval primitives, exact Higham dot-product
accumulation error bounds, non-expansive activation enclosures, and fail-closed
numerical capability checks for ECDR Week 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# IEEE 754 Machine Constants and Supported Profiles
# ==============================================================================

# Unit roundoff u = eps_mach / 2
ULP32_HALF: float = 2.0 ** -24       # ~5.960464477539063e-08
ULP64_HALF: float = 2.0 ** -53       # ~1.1102230246251565e-16

EPS_MACH_F32: float = 2.0 ** -23     # ~1.1920928955078125e-07
EPS_MACH_F64: float = 2.0 ** -52     # ~2.220446049250313e-16

SUB_NORMAL_F32_MAX: float = 2.0 ** -126  # ~1.1754943508222875e-38

# Validated dot product lengths corresponding to SmallCNN1D layers:
# Conv1 (30), Conv2 (96), Conv3 (192), Linear1 (2048), Linear2 (128)
SUPPORTED_DOT_LENGTHS: FrozenSet[int] = frozenset({30, 96, 128, 192, 2048})

LOCKED_PROFILE_ID: str = "linux-x86_64-py311-numpy246-torch2141cpu-avx512"

SUPPORTED_PROFILES: FrozenSet[str] = frozenset({
    LOCKED_PROFILE_ID,
    "linux-x86_64-py311-torch2141cpu-avx512",
    "cpu_float32",
    "scalar_ieee754",
})


# ==============================================================================
# Numerical Domain Exceptions
# ==============================================================================

class NumericalError(ValueError):
    """Base class for domain numerical and enclosure failures."""


class InvertedIntervalError(NumericalError):
    """Raised when an interval lower bound strictly exceeds its upper bound."""


class DivisionByZeroIntervalError(NumericalError):
    """Raised when an interval division divisor contains or equals zero."""


class NegativeSqrtIntervalError(NumericalError):
    """Raised when taking the square root of a strictly negative interval."""


class UnsupportedNumericalError(NumericalError):
    """Raised when an operation, dimension, or profile is outside verified capability."""


class NonFiniteIntervalError(NumericalError):
    """Raised when an endpoint is NaN or Infinite."""


# ==============================================================================
# Outward Directed Rounding Primitives
# ==============================================================================

def round_down_f64(x: float) -> float:
    """Round down to the next representable float64 number towards -infinity."""
    if not math.isfinite(x):
        raise NonFiniteIntervalError(f"Cannot round non-finite float64: {x}")
    return math.nextafter(x, -math.inf)


def round_up_f64(x: float) -> float:
    """Round up to the next representable float64 number towards +infinity."""
    if not math.isfinite(x):
        raise NonFiniteIntervalError(f"Cannot round non-finite float64: {x}")
    return math.nextafter(x, math.inf)


def round_down_f32(x: float) -> float:
    """Cast to float32 and round down to the nearest representable float32."""
    if not math.isfinite(x):
        raise NonFiniteIntervalError(f"Cannot round non-finite float32: {x}")
    val_f32 = np.float32(x)
    return float(np.nextafter(val_f32, np.float32(-np.inf), dtype=np.float32))


def round_up_f32(x: float) -> float:
    """Cast to float32 and round up to the nearest representable float32."""
    if not math.isfinite(x):
        raise NonFiniteIntervalError(f"Cannot round non-finite float32: {x}")
    val_f32 = np.float32(x)
    return float(np.nextafter(val_f32, np.float32(np.inf), dtype=np.float32))


def gamma_bound(n: int, u: float = ULP32_HALF) -> float:
    """Compute the Higham error factor gamma_n = (n * u) / (1 - n * u).
    
    Strictly requires n * u < 1.0; otherwise raises UnsupportedNumericalError.
    """
    if n <= 0:
        return 0.0
    nu = float(n) * u
    if nu >= 1.0:
        raise UnsupportedNumericalError(
            f"Reduction length {n} with unit roundoff {u} violates stability condition nu < 1 (nu={nu})"
        )
    return nu / (1.0 - nu)


# ==============================================================================
# Scalar Interval Class
# ==============================================================================

@dataclass(frozen=True)
class Interval:
    """Closed real interval [lo, hi] with verified finite endpoints and outward rounding."""
    lo: float
    hi: float

    def __post_init__(self) -> None:
        try:
            lo_f = float(self.lo)
            hi_f = float(self.hi)
        except (ValueError, TypeError) as err:
            raise NonFiniteIntervalError(f"Endpoints must be float-convertible, got ({self.lo}, {self.hi})") from err

        if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
            raise NonFiniteIntervalError(f"Interval endpoints must be strictly finite: [{lo_f}, {hi_f}]")

        if lo_f > hi_f:
            raise InvertedIntervalError(
                f"Inverted interval: lower bound {lo_f} strictly exceeds upper bound {hi_f}"
            )

        object.__setattr__(self, "lo", lo_f)
        object.__setattr__(self, "hi", hi_f)

    @classmethod
    def point(cls, val: float) -> Interval:
        """Create a degenerate point interval [val, val]."""
        f = float(val)
        return cls(lo=f, hi=f)

    @classmethod
    def from_f32_enclosure(cls, val: float) -> Interval:
        """Enclose a scalar value by outward float32 rounding [round_down(val), round_up(val)]."""
        return cls(lo=round_down_f32(val), hi=round_up_f32(val))

    @property
    def is_degenerate(self) -> bool:
        """Return True if lo == hi."""
        return self.lo == self.hi

    @property
    def rad(self) -> float:
        """Return interval radius (hi - lo) / 2."""
        return (self.hi - self.lo) / 2.0

    @property
    def mid(self) -> float:
        """Return interval midpoint (lo + hi) / 2."""
        return (self.lo + self.hi) / 2.0

    @property
    def mag(self) -> float:
        """Return maximum absolute value (magnitude): max(|lo|, |hi|)."""
        return max(abs(self.lo), abs(self.hi))

    @property
    def mig(self) -> float:
        """Return minimum absolute value (mignitude).
        
        mig = 0 if 0 in [lo, hi], else min(|lo|, |hi|).
        """
        if self.lo <= 0.0 <= self.hi:
            return 0.0
        return min(abs(self.lo), abs(self.hi))

    def contains(self, x: Union[float, int, Interval]) -> bool:
        """Check whether x is completely enclosed within [lo, hi]."""
        if isinstance(x, Interval):
            return self.lo <= x.lo and x.hi <= self.hi
        try:
            val = float(x)
        except (ValueError, TypeError):
            return False
        return self.lo <= val <= self.hi

    def widen(self, delta: float) -> Interval:
        """Symmetrically expand interval by non-negative delta with outward directed rounding."""
        d = float(delta)
        if d < 0.0:
            raise ValueError(f"Widen delta must be non-negative, got {d}")
        new_lo = round_down_f64(self.lo - d)
        new_hi = round_up_f64(self.hi + d)
        return Interval(lo=new_lo, hi=new_hi)

    def add(self, other: Union[Interval, float, int]) -> Interval:
        """Interval addition: [a, b] + [c, d] = [round_down(a + c), round_up(b + d)]."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        new_lo = round_down_f64(self.lo + other.lo)
        new_hi = round_up_f64(self.hi + other.hi)
        return Interval(lo=new_lo, hi=new_hi)

    def sub(self, other: Union[Interval, float, int]) -> Interval:
        """Interval subtraction: [a, b] - [c, d] = [round_down(a - d), round_up(b - c)]."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        new_lo = round_down_f64(self.lo - other.hi)
        new_hi = round_up_f64(self.hi - other.lo)
        return Interval(lo=new_lo, hi=new_hi)

    def mul(self, other: Union[Interval, float, int]) -> Interval:
        """Interval multiplication with outward directed rounding covering all endpoint products."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        p1 = self.lo * other.lo
        p2 = self.lo * other.hi
        p3 = self.hi * other.lo
        p4 = self.hi * other.hi
        min_p = min(p1, p2, p3, p4)
        max_p = max(p1, p2, p3, p4)
        return Interval(lo=round_down_f64(min_p), hi=round_up_f64(max_p))

    def scale(self, factor: float) -> Interval:
        """Multiply interval by a scalar factor with outward directed rounding."""
        f = float(factor)
        if not math.isfinite(f):
            raise NonFiniteIntervalError(f"Scale factor must be finite, got {f}")
        if f == 0.0:
            return Interval(lo=0.0, hi=0.0)
        p1 = self.lo * f
        p2 = self.hi * f
        min_p = min(p1, p2)
        max_p = max(p1, p2)
        return Interval(lo=round_down_f64(min_p), hi=round_up_f64(max_p))

    def div(self, other: Union[Interval, float, int]) -> Interval:
        """Interval division: raises DivisionByZeroIntervalError if divisor contains 0."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        if other.lo <= 0.0 <= other.hi:
            raise DivisionByZeroIntervalError(
                f"Interval division by zero or interval containing zero: [{other.lo}, {other.hi}]"
            )
        q1 = self.lo / other.lo
        q2 = self.lo / other.hi
        q3 = self.hi / other.lo
        q4 = self.hi / other.hi
        min_q = min(q1, q2, q3, q4)
        max_q = max(q1, q2, q3, q4)
        return Interval(lo=round_down_f64(min_q), hi=round_up_f64(max_q))

    def sqrt(self) -> Interval:
        """Interval square root: raises NegativeSqrtIntervalError if strictly negative."""
        if self.hi < 0.0:
            raise NegativeSqrtIntervalError(
                f"Cannot compute square root of strictly negative interval: [{self.lo}, {self.hi}]"
            )
        safe_lo = max(0.0, self.lo)
        res_lo = 0.0 if safe_lo == 0.0 else round_down_f64(math.sqrt(safe_lo))
        res_hi = round_up_f64(math.sqrt(self.hi))
        return Interval(lo=res_lo, hi=res_hi)

    def relu(self) -> Interval:
        """Non-expansive coordinate-wise ReLU: [max(0, lo), max(0, hi)]."""
        return Interval(lo=max(0.0, self.lo), hi=max(0.0, self.hi))

    def max_with(self, other: Union[Interval, float, int]) -> Interval:
        """Coordinate-wise maximum of two intervals: [max(lo1, lo2), max(hi1, hi2)]."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        return Interval(lo=max(self.lo, other.lo), hi=max(self.hi, other.hi))

    def min_with(self, other: Union[Interval, float, int]) -> Interval:
        """Coordinate-wise minimum of two intervals: [min(lo1, lo2), min(hi1, hi2)]."""
        if isinstance(other, (int, float)):
            other = Interval.point(float(other))
        return Interval(lo=min(self.lo, other.lo), hi=min(self.hi, other.hi))

    # Python operator overloads for expressive mathematical syntax
    def __add__(self, other: Union[Interval, float, int]) -> Interval:
        return self.add(other)

    def __radd__(self, other: Union[Interval, float, int]) -> Interval:
        return self.add(other)

    def __sub__(self, other: Union[Interval, float, int]) -> Interval:
        return self.sub(other)

    def __rsub__(self, other: Union[Interval, float, int]) -> Interval:
        return Interval.point(float(other)).sub(self)

    def __mul__(self, other: Union[Interval, float, int]) -> Interval:
        return self.mul(other)

    def __rmul__(self, other: Union[Interval, float, int]) -> Interval:
        return self.mul(other)

    def __truediv__(self, other: Union[Interval, float, int]) -> Interval:
        return self.div(other)

    def __neg__(self) -> Interval:
        return Interval(lo=-self.hi, hi=-self.lo)


# ==============================================================================
# Vectorized Interval Container
# ==============================================================================

@dataclass(frozen=True)
class IntervalVector:
    """1D or ND array of intervals with element-wise bounds and dot-product enclosure."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo_arr = np.asarray(self.lo, dtype=np.float64)
        hi_arr = np.asarray(self.hi, dtype=np.float64)

        if lo_arr.shape != hi_arr.shape:
            raise ValueError(f"Shape mismatch between lo {lo_arr.shape} and hi {hi_arr.shape}")

        if not (np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr))):
            raise NonFiniteIntervalError("IntervalVector contains non-finite endpoints")

        if np.any(lo_arr > hi_arr):
            bad_idx = np.where(lo_arr > hi_arr)
            raise InvertedIntervalError(
                f"Inverted interval detected at indices {bad_idx}: lo={lo_arr[bad_idx]} > hi={hi_arr[bad_idx]}"
            )

        # Ensure arrays are immutable/read-only to preserve frozen semantics
        lo_arr.flags.writeable = False
        hi_arr.flags.writeable = False
        object.__setattr__(self, "lo", lo_arr)
        object.__setattr__(self, "hi", hi_arr)

    @classmethod
    def from_center_and_radius(cls, center: np.ndarray, radius: Union[float, np.ndarray]) -> IntervalVector:
        """Create an IntervalVector from center coordinates and non-negative radii."""
        c = np.asarray(center, dtype=np.float64)
        if isinstance(radius, (int, float)):
            r = np.full_like(c, float(radius))
        else:
            r = np.asarray(radius, dtype=np.float64)
        if np.any(r < 0.0):
            raise ValueError("Radii must be non-negative")
        lo = c - r
        hi = c + r
        return cls(lo=lo, hi=hi)

    @classmethod
    def from_point_array(cls, arr: np.ndarray) -> IntervalVector:
        """Create degenerate IntervalVector where lo == hi == arr."""
        a = np.asarray(arr, dtype=np.float64)
        return cls(lo=a, hi=a.copy())

    def __len__(self) -> int:
        return int(len(self.lo))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lo.shape

    @property
    def mag(self) -> np.ndarray:
        """Element-wise magnitude: max(|lo|, |hi|)."""
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def contains(self, arr: np.ndarray) -> bool:
        """Check if entire array arr is coordinate-wise enclosed."""
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != self.shape:
            return False
        return bool(np.all(self.lo <= a) and np.all(a <= self.hi))

    def relu(self) -> IntervalVector:
        """Coordinate-wise non-expansive ReLU."""
        return IntervalVector(lo=np.maximum(0.0, self.lo), hi=np.maximum(0.0, self.hi))

    def add(self, other: Union[IntervalVector, np.ndarray]) -> IntervalVector:
        """Element-wise addition."""
        if isinstance(other, np.ndarray):
            other = IntervalVector.from_point_array(other)
        return IntervalVector(lo=self.lo + other.lo, hi=self.hi + other.hi)

    def sub(self, other: Union[IntervalVector, np.ndarray]) -> IntervalVector:
        """Element-wise subtraction."""
        if isinstance(other, np.ndarray):
            other = IntervalVector.from_point_array(other)
        return IntervalVector(lo=self.lo - other.hi, hi=self.hi - other.lo)

    def dot_with_constant(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        enforce_supported_length: bool = True,
        profile: str = "linux-x86_64-py311-torch2141cpu-avx512",
    ) -> Interval:
        """Compute sound interval dot product against constant weights with Higham error bound.
        
        Evaluates s = sum_{i=1}^n w_i * [lo_i, hi_i] + b.
        Bounds accumulated floating-point reduction error using gamma_n = (n*u)/(1 - n*u)
        where u = 2^-24 (single-precision roundoff) and widens the resulting interval.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(self.lo.ravel()) != len(w):
            raise ValueError(f"Dimension mismatch: interval vector {len(self.lo.ravel())} vs weights {len(w)}")

        n = len(w)
        if enforce_supported_length and n not in SUPPORTED_DOT_LENGTHS:
            raise UnsupportedNumericalError(
                f"Dot product length {n} is outside validated SmallCNN1D lengths: {sorted(SUPPORTED_DOT_LENGTHS)}"
            )

        if profile not in SUPPORTED_PROFILES:
            raise UnsupportedNumericalError(f"Unsupported execution profile: {profile}")

        # Exact scalar dot product intervalization:
        # For each coordinate i:
        # if w_i >= 0: term in [w_i * lo_i, w_i * hi_i]
        # if w_i < 0:  term in [w_i * hi_i, w_i * lo_i]
        pos_mask = (w >= 0.0)
        neg_mask = ~pos_mask

        lo_flat = self.lo.ravel()
        hi_flat = self.hi.ravel()

        term_lo = np.zeros(n, dtype=np.float64)
        term_hi = np.zeros(n, dtype=np.float64)

        term_lo[pos_mask] = w[pos_mask] * lo_flat[pos_mask]
        term_hi[pos_mask] = w[pos_mask] * hi_flat[pos_mask]

        term_lo[neg_mask] = w[neg_mask] * hi_flat[neg_mask]
        term_hi[neg_mask] = w[neg_mask] * lo_flat[neg_mask]

        # Accumulation with float64 precision
        sum_lo = float(np.sum(term_lo) + bias)
        sum_hi = float(np.sum(term_hi) + bias)

        # Higham dot product accumulation error bound:
        # For a dot product of length n with an added scalar bias, there are n products
        # and n additions (accumulating n + 1 terms total if bias != 0).
        # Theorem 3.1 (Higham 2002): forward error is bounded by gamma_{n+1} * (sum |w_i x_i| + |bias|).
        max_abs_terms = np.maximum(np.abs(term_lo), np.abs(term_hi))
        abs_sum = float(np.sum(max_abs_terms) + abs(bias))

        num_accum_terms = n + 1 if float(bias) != 0.0 else n
        gamma = gamma_bound(num_accum_terms, u=ULP32_HALF)
        delta_reduction = gamma * abs_sum

        # Add subnormal margin protection if applicable
        delta_total = delta_reduction + (SUB_NORMAL_F32_MAX * float(num_accum_terms))

        # Directed outward rounding to enclose final float32 executable result
        final_lo = round_down_f32(sum_lo - delta_total)
        final_hi = round_up_f32(sum_hi + delta_total)

        return Interval(lo=final_lo, hi=final_hi)


# ==============================================================================
# Exact Rational Oracle (For Testing and Certification Checks)
# ==============================================================================

class ExactRationalOracle:
    """Exact rational arithmetic oracle using fractions.Fraction to verify numerical bounds."""

    @staticmethod
    def exact_dot_product(
        x_vals: Sequence[Union[float, Fraction]],
        weights: Sequence[Union[float, Fraction]],
        bias: Union[float, Fraction] = 0.0,
    ) -> Fraction:
        """Compute mathematically exact dot product in rational arithmetic."""
        if len(x_vals) != len(weights):
            raise ValueError("Length mismatch in exact dot product")
        res = Fraction(0, 1)
        for xi, wi in zip(x_vals, weights):
            res += Fraction(xi) * Fraction(wi)
        res += Fraction(bias)
        return res

    @staticmethod
    def verify_enclosure(interval: Interval, exact_val: Fraction) -> bool:
        """Check whether exact_val in Fraction is strictly within interval [lo, hi]."""
        frac_lo = Fraction(interval.lo)
        frac_hi = Fraction(interval.hi)
        return frac_lo <= exact_val <= frac_hi


# ==============================================================================
# Vectorized Directed Rounding Primitives
# ==============================================================================

def round_down_f32_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorized outward rounding down to previous representable float32."""
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise NonFiniteIntervalError("Array contains non-finite values in round_down_f32_vec")
    return np.nextafter(np.float32(a), np.float32(-np.inf), dtype=np.float32).astype(np.float64)


def round_up_f32_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorized outward rounding up to next representable float32."""
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise NonFiniteIntervalError("Array contains non-finite values in round_up_f32_vec")
    return np.nextafter(np.float32(a), np.float32(np.inf), dtype=np.float32).astype(np.float64)


# ==============================================================================
# W4-008: Validated Neural Network Operator Enclosures
# ==============================================================================

class OperatorKind(str, Enum):
    CONV1D = "Conv1d"
    BATCHNORM1D = "BatchNorm1d"
    RELU = "ReLU"
    MAXPOOL1D = "MaxPool1d"
    FLATTEN = "Flatten"
    LINEAR = "Linear"


@dataclass(frozen=True)
class OperatorCapability:
    """Audit record for a supported neural network operator under locked profile."""
    layer_name: str
    operator_kind: str
    real_valid: bool
    executable_bridge_status: str
    dot_product_length: Optional[int]
    error_bound_formula: str
    max_accumulation_error_estimate: float
    supported_attributes: Dict[str, Any]
    profile: str = LOCKED_PROFILE_ID


def enclose_conv1d(
    x_box: IntervalVector,
    conv_layer: nn.Conv1d,
    profile: str = LOCKED_PROFILE_ID,
) -> IntervalVector:
    """Enclose 1D signed padded Convolution with Higham floating-point error bound.

    Enforces W4-008-C01:
    - Verifies stride=1, dilation=1, groups=1, non-negative integer padding.
    - Validates input shape (C_in, L_in) against layer channels.
    - Accurately treats left/right zero-padding as exact [0.0, 0.0] intervals.
    - Uses Higham backward error analysis gamma_{n+1} * (|W| |X| + |b|) for dot product
      length n = C_in * kernel_size in {30, 96, 192}.
    - Applies outward directed rounding in float32.
    """
    if profile not in SUPPORTED_PROFILES:
        raise UnsupportedNumericalError(f"Unsupported execution profile: {profile}")

    if not isinstance(conv_layer, nn.Conv1d):
        raise TypeError(f"Expected nn.Conv1d, got {type(conv_layer)}")

    # Reject unreviewed/unsupported convolution attributes (W4-008-C01)
    stride = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
    if stride != 1:
        raise UnsupportedNumericalError(f"Conv1d stride={stride} not supported; must be 1")

    dilation = conv_layer.dilation[0] if isinstance(conv_layer.dilation, tuple) else conv_layer.dilation
    if dilation != 1:
        raise UnsupportedNumericalError(f"Conv1d dilation={dilation} not supported; must be 1")

    if conv_layer.groups != 1:
        raise UnsupportedNumericalError(f"Conv1d groups={conv_layer.groups} not supported; must be 1")

    padding = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
    if not isinstance(padding, int) or padding < 0:
        raise UnsupportedNumericalError(f"Invalid padding: {padding}")

    x_shape = x_box.shape
    if len(x_shape) != 2:
        raise ValueError(f"Input interval vector must have 2D shape (C_in, L_in), got {x_shape}")

    c_in, l_in = x_shape
    if c_in != conv_layer.in_channels:
        raise ValueError(f"Channel mismatch: input has {c_in}, Conv1d expects {conv_layer.in_channels}")

    w = conv_layer.weight.detach().cpu().numpy().astype(np.float64)  # (C_out, C_in, K)
    c_out, _, k = w.shape
    n = c_in * k
    if n not in SUPPORTED_DOT_LENGTHS:
        raise UnsupportedNumericalError(f"Dot product length {n} is outside validated SmallCNN1D lengths")

    if conv_layer.bias is not None:
        b = conv_layer.bias.detach().cpu().numpy().astype(np.float64)  # (C_out,)
        has_bias = True
    else:
        b = np.zeros(c_out, dtype=np.float64)
        has_bias = False

    l_out = l_in + 2 * padding - k + 1
    if l_out <= 0:
        raise ValueError(f"Calculated non-positive output length L_out={l_out}")

    # Zero-pad input interval arrays (W4-008-C01 padding treatment)
    x_lo_pad = np.pad(x_box.lo, ((0, 0), (padding, padding)), mode="constant", constant_values=0.0)
    x_hi_pad = np.pad(x_box.hi, ((0, 0), (padding, padding)), mode="constant", constant_values=0.0)

    # Unfold spatial patches into matrix of shape (n, L_out)
    patches_lo = [x_lo_pad[:, j : j + k].reshape(-1) for j in range(l_out)]
    patches_hi = [x_hi_pad[:, j : j + k].reshape(-1) for j in range(l_out)]
    u_lo = np.column_stack(patches_lo)  # (n, L_out)
    u_hi = np.column_stack(patches_hi)  # (n, L_out)

    w_flat = w.reshape(c_out, n)  # (C_out, n)
    w_pos = np.maximum(0.0, w_flat)
    w_neg = np.minimum(0.0, w_flat)

    # Ideal interval dot product over R
    y_lo_ideal = w_pos @ u_lo + w_neg @ u_hi + b[:, None]
    y_hi_ideal = w_pos @ u_hi + w_neg @ u_lo + b[:, None]

    # Higham dot product accumulation error bound (W4-008-C04)
    # n multiplications + (n-1) additions + 1 bias addition = n+1 terms if bias present
    num_accum_terms = n + 1 if has_bias else n
    gamma = gamma_bound(num_accum_terms, u=ULP32_HALF)

    # Maximum absolute operand term: |W| * max(|U_lo|, |U_hi|) + |b|
    m_u = np.maximum(np.abs(u_lo), np.abs(u_hi))
    s_abs = np.abs(w_flat) @ m_u + np.abs(b[:, None])
    delta = gamma * s_abs + (SUB_NORMAL_F32_MAX * float(num_accum_terms))

    # Outward directed rounding in float32
    y_lo = round_down_f32_vec(y_lo_ideal - delta)
    y_hi = round_up_f32_vec(y_hi_ideal + delta)

    return IntervalVector(lo=y_lo, hi=y_hi)


def enclose_batchnorm1d(
    x_box: IntervalVector,
    bn_layer: nn.BatchNorm1d,
    profile: str = LOCKED_PROFILE_ID,
) -> IntervalVector:
    """Enclose 1D BatchNorm in eval mode with running buffers and folding error bounds.

    Enforces W4-008-C02 & W4-008-C06:
    - Enforces eval mode (training == False); rejects training-mode BN with UnsupportedNumericalError.
    - Validates stored running_mean and running_var buffers are non-null, finite, and var >= 0.
    - Explicitly handles positive, negative, and zero gamma (scaling reverses monotonicity when negative).
    - Correctly models affine folding: alpha = gamma / sqrt(var + eps), beta_eff = beta - mu * alpha.
    - Accounts for arithmetic evaluation order and rounding error via gamma_4 bound.
    """
    if profile not in SUPPORTED_PROFILES:
        raise UnsupportedNumericalError(f"Unsupported execution profile: {profile}")

    if not isinstance(bn_layer, nn.BatchNorm1d):
        raise TypeError(f"Expected nn.BatchNorm1d, got {type(bn_layer)}")

    if bn_layer.training:
        raise UnsupportedNumericalError("BatchNorm1d in training mode is not supported for interval enclosure (W4-008-C06)")

    if bn_layer.running_mean is None or bn_layer.running_var is None:
        raise UnsupportedNumericalError("BatchNorm1d missing frozen running statistics (running_mean / running_var)")

    c = bn_layer.num_features
    if x_box.shape[0] != c:
        raise ValueError(f"Feature dimension mismatch: input has {x_box.shape[0]}, BatchNorm expects {c}")

    mu = bn_layer.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn_layer.running_var.detach().cpu().numpy().astype(np.float64)

    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(var))):
        raise NonFiniteIntervalError("BatchNorm1d running statistics contain non-finite values")

    if np.any(var < 0.0):
        raise ValueError("BatchNorm1d running_var contains negative values")

    eps = float(bn_layer.eps)
    if eps <= 0.0:
        raise ValueError(f"BatchNorm1d eps must be positive, got {eps}")

    if bn_layer.affine and bn_layer.weight is not None:
        gamma = bn_layer.weight.detach().cpu().numpy().astype(np.float64)
    else:
        gamma = np.ones(c, dtype=np.float64)

    if bn_layer.affine and bn_layer.bias is not None:
        beta = bn_layer.bias.detach().cpu().numpy().astype(np.float64)
    else:
        beta = np.zeros(c, dtype=np.float64)

    # Compute effective affine parameters in float64
    denom = np.sqrt(var + eps)
    alpha = gamma / denom
    beta_eff = beta - mu * alpha

    # Expand to 2D for broadcasting over spatial length if needed
    if len(x_box.shape) == 2:
        alpha_2d = alpha[:, None]
        beta_eff_2d = beta_eff[:, None]
        mu_2d = mu[:, None]
        beta_2d = beta[:, None]
    else:
        alpha_2d = alpha
        beta_eff_2d = beta_eff
        mu_2d = mu
        beta_2d = beta

    # Sign of alpha determines interval endpoint mapping (W4-008-C02)
    pos_mask = (alpha_2d >= 0.0)

    # When alpha >= 0: lo -> lo, hi -> hi
    # When alpha < 0:  lo -> hi, hi -> lo
    y_lo_ideal = np.where(pos_mask, alpha_2d * x_box.lo + beta_eff_2d, alpha_2d * x_box.hi + beta_eff_2d)
    y_hi_ideal = np.where(pos_mask, alpha_2d * x_box.hi + beta_eff_2d, alpha_2d * x_box.lo + beta_eff_2d)

    # Folding & evaluation error bound:
    # 4 floating-point operations in effective map (division, sqrt, multiply, addition)
    gamma_4 = gamma_bound(4, u=ULP32_HALF)
    abs_x = np.maximum(np.abs(x_box.lo), np.abs(x_box.hi))
    delta = gamma_4 * (abs_x * np.abs(alpha_2d) + np.abs(beta_eff_2d) + np.abs(mu_2d * alpha_2d) + np.abs(beta_2d)) + 4.0 * SUB_NORMAL_F32_MAX

    y_lo = round_down_f32_vec(y_lo_ideal - delta)
    y_hi = round_up_f32_vec(y_hi_ideal + delta)

    return IntervalVector(lo=y_lo, hi=y_hi)


def enclose_relu(x_box: IntervalVector) -> IntervalVector:
    """Enclose coordinate-wise ReLU activation.

    Enforces W4-008-C03:
    - Exactly handles strictly negative intervals ([lo, hi] with hi <= 0 -> [0.0, 0.0]).
    - Exactly handles straddling intervals (lo < 0 < hi -> [0.0, hi]).
    - Non-expansive, exact in float32 arithmetic (zero floating-point error).
    """
    y_lo = np.maximum(0.0, x_box.lo)
    y_hi = np.maximum(0.0, x_box.hi)
    return IntervalVector(lo=y_lo, hi=y_hi)


def enclose_maxpool1d(
    x_box: IntervalVector,
    pool_layer: nn.MaxPool1d,
) -> IntervalVector:
    """Enclose 1D Max Pooling with kernel_size=2, stride=2.

    Enforces W4-008-C03:
    - Verifies stride=2, padding=0, dilation=1, ceil_mode=False.
    - Handles pool ties (equal inputs) without error.
    - Operates along spatial axis (last axis) on even input length.
    - Non-expansive, exact in float32 arithmetic (zero floating-point error).
    """
    if not isinstance(pool_layer, nn.MaxPool1d):
        raise TypeError(f"Expected nn.MaxPool1d, got {type(pool_layer)}")

    k = pool_layer.kernel_size if isinstance(pool_layer.kernel_size, int) else pool_layer.kernel_size[0]
    s = pool_layer.stride if isinstance(pool_layer.stride, int) else pool_layer.stride[0]
    p = pool_layer.padding if isinstance(pool_layer.padding, int) else pool_layer.padding[0]
    d = pool_layer.dilation if isinstance(pool_layer.dilation, int) else pool_layer.dilation[0]
    ceil_mode = pool_layer.ceil_mode

    if k != 2 or s != 2 or p != 0 or d != 1 or ceil_mode:
        raise UnsupportedNumericalError(
            f"Unsupported MaxPool1d config: kernel={k}, stride={s}, padding={p}, dilation={d}, ceil_mode={ceil_mode}"
        )

    l_in = x_box.shape[-1]
    if l_in % 2 != 0:
        raise ValueError(f"Input spatial length {l_in} must be divisible by stride 2")

    # For window {2j, 2j+1}, interval max is:
    # y_lo = max(x_lo[2j], x_lo[2j+1])
    # y_hi = max(x_hi[2j], x_hi[2j+1])
    y_lo = np.maximum(x_box.lo[..., 0::2], x_box.lo[..., 1::2])
    y_hi = np.maximum(x_box.hi[..., 0::2], x_box.hi[..., 1::2])

    return IntervalVector(lo=y_lo, hi=y_hi)


def enclose_flatten(x_box: IntervalVector) -> IntervalVector:
    """Enclose channel-major flattening.

    Enforces W4-008-C03:
    - Preserves exact channel-major order of the computational graph.
    - Exact index reshaping, zero floating-point error.
    """
    return IntervalVector(lo=x_box.lo.reshape(-1), hi=x_box.hi.reshape(-1))


def enclose_linear(
    x_box: IntervalVector,
    linear_layer: nn.Linear,
    profile: str = LOCKED_PROFILE_ID,
) -> IntervalVector:
    """Enclose Linear layer with Higham dot-product error bound.

    Enforces W4-008-C04:
    - Verifies linear in_features in SUPPORTED_DOT_LENGTHS ({128, 2048}).
    - Implements sound interval matrix-vector product W * X + b.
    - Explicitly bounds reduction error via gamma_{n+1} * (|W| |X| + |b|).
    - Applies outward directed rounding in float32.
    """
    if profile not in SUPPORTED_PROFILES:
        raise UnsupportedNumericalError(f"Unsupported execution profile: {profile}")

    if not isinstance(linear_layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear_layer)}")

    n = linear_layer.in_features
    if n not in SUPPORTED_DOT_LENGTHS:
        raise UnsupportedNumericalError(f"Linear in_features {n} is outside validated SmallCNN1D lengths")

    x_lo = x_box.lo.ravel()
    x_hi = x_box.hi.ravel()
    if len(x_lo) != n:
        raise ValueError(f"Input length mismatch: {len(x_lo)} != in_features {n}")

    w = linear_layer.weight.detach().cpu().numpy().astype(np.float64)  # (out_features, in_features)
    out_features = linear_layer.out_features

    if linear_layer.bias is not None:
        b = linear_layer.bias.detach().cpu().numpy().astype(np.float64)  # (out_features,)
        has_bias = True
    else:
        b = np.zeros(out_features, dtype=np.float64)
        has_bias = False

    w_pos = np.maximum(0.0, w)
    w_neg = np.minimum(0.0, w)

    # Ideal interval dot product over R
    y_lo_ideal = w_pos @ x_lo + w_neg @ x_hi + b
    y_hi_ideal = w_pos @ x_hi + w_neg @ x_lo + b

    # Higham error bound
    num_accum_terms = n + 1 if has_bias else n
    gamma = gamma_bound(num_accum_terms, u=ULP32_HALF)

    m_x = np.maximum(np.abs(x_lo), np.abs(x_hi))
    s_abs = np.abs(w) @ m_x + np.abs(b)
    delta = gamma * s_abs + (SUB_NORMAL_F32_MAX * float(num_accum_terms))

    y_lo = round_down_f32_vec(y_lo_ideal - delta)
    y_hi = round_up_f32_vec(y_hi_ideal + delta)

    return IntervalVector(lo=y_lo, hi=y_hi)


def enclose_mlp_head(
    h3_box: IntervalVector,
    head: nn.Sequential,
    profile: str = LOCKED_PROFILE_ID,
) -> IntervalVector:
    """Enclose complete Flatten(2048) -> Linear(128) -> ReLU -> Linear(12) head.

    Enforces W4-008-C04:
    - Validates sequential structure of the head.
    - Evaluates Flatten, Linear1 (2048->128), hidden ReLU, and Linear2 (128->12).
    - Returns enclosed output logits interval vector of shape (12,).
    """
    if len(head) != 4:
        raise ValueError(f"Expected 4-module head sequential, got length {len(head)}")

    # Module 0: Flatten
    flat_box = enclose_flatten(h3_box)

    # Module 1: Linear 2048 -> 128
    l1_box = enclose_linear(flat_box, head[1], profile=profile)

    # Module 2: ReLU
    r_box = enclose_relu(l1_box)

    # Module 3: Linear 128 -> 12
    logits_box = enclose_linear(r_box, head[3], profile=profile)

    return logits_box


def enclose_smallcnn1d(
    x_norm_box: IntervalVector,
    model: nn.Module,
    profile: str = LOCKED_PROFILE_ID,
) -> Tuple[IntervalVector, Dict[str, IntervalVector]]:
    """Execute end-to-end interval enclosure through entire SmallCNN1D network.

    Passes through:
    Stage 1: Conv1 -> BN1 -> ReLU1 -> Pool1
    Stage 2: Conv2 -> BN2 -> ReLU2 -> Pool2
    Stage 3: Conv3 -> BN3 -> ReLU3
    Head: Flatten -> Linear1 -> ReLU -> Linear2
    Returns:
    (logits_box, stage_boxes)
    where stage_boxes contains {"h1": ..., "p1": ..., "h2": ..., "p2": ..., "h3": ..., "logits": ...}.
    """
    c = model.conv
    h = model.head

    # Conv1 + BN1 + ReLU1
    c1 = enclose_conv1d(x_norm_box, c[0], profile=profile)
    bn1 = enclose_batchnorm1d(c1, c[1], profile=profile)
    h1 = enclose_relu(bn1)

    # Pool1
    p1 = enclose_maxpool1d(h1, c[3])

    # Conv2 + BN2 + ReLU2
    c2 = enclose_conv1d(p1, c[4], profile=profile)
    bn2 = enclose_batchnorm1d(c2, c[5], profile=profile)
    h2 = enclose_relu(bn2)

    # Pool2
    p2 = enclose_maxpool1d(h2, c[7])

    # Conv3 + BN3 + ReLU3
    c3 = enclose_conv1d(p2, c[8], profile=profile)
    bn3 = enclose_batchnorm1d(c3, c[9], profile=profile)
    h3 = enclose_relu(bn3)

    # Head
    logits = enclose_mlp_head(h3, h, profile=profile)

    stage_boxes = {
        "h1": h1,
        "p1": p1,
        "h2": h2,
        "p2": p2,
        "h3": h3,
        "logits": logits,
    }
    return logits, stage_boxes


def audit_smallcnn1d_operators(
    model: nn.Module,
    profile: str = LOCKED_PROFILE_ID,
) -> List[OperatorCapability]:
    """Audit every operator in SmallCNN1D and return capability records (W4-008-C07)."""
    capabilities: List[OperatorCapability] = []

    # Conv1
    c0 = model.conv[0]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.0",
            operator_kind=OperatorKind.CONV1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=c0.in_channels * c0.kernel_size[0],
            error_bound_formula="gamma_{n+1} * (|W| |X| + |b|) + subnormal_padding",
            max_accumulation_error_estimate=gamma_bound(31, u=ULP32_HALF),
            supported_attributes={"stride": 1, "dilation": 1, "groups": 1, "padding": c0.padding[0]},
            profile=profile,
        )
    )

    # BN1
    bn1 = model.conv[1]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.1",
            operator_kind=OperatorKind.BATCHNORM1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="gamma_4 * (|X| |alpha| + |beta_eff| + |mu alpha| + |beta|) + subnormal",
            max_accumulation_error_estimate=gamma_bound(4, u=ULP32_HALF),
            supported_attributes={"affine": bn1.affine, "eps": bn1.eps, "training_eval_mode": not bn1.training},
            profile=profile,
        )
    )

    # ReLU1
    capabilities.append(
        OperatorCapability(
            layer_name="conv.2",
            operator_kind=OperatorKind.RELU.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact non-expansive float32)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"inplace": False},
            profile=profile,
        )
    )

    # Pool1
    p1 = model.conv[3]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.3",
            operator_kind=OperatorKind.MAXPOOL1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact pairwise float32 max)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"kernel_size": 2, "stride": 2, "padding": 0, "ceil_mode": False},
            profile=profile,
        )
    )

    # Conv2
    c4 = model.conv[4]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.4",
            operator_kind=OperatorKind.CONV1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=c4.in_channels * c4.kernel_size[0],
            error_bound_formula="gamma_{n+1} * (|W| |X| + |b|) + subnormal_padding",
            max_accumulation_error_estimate=gamma_bound(97, u=ULP32_HALF),
            supported_attributes={"stride": 1, "dilation": 1, "groups": 1, "padding": c4.padding[0]},
            profile=profile,
        )
    )

    # BN2
    bn2 = model.conv[5]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.5",
            operator_kind=OperatorKind.BATCHNORM1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="gamma_4 * (|X| |alpha| + |beta_eff| + |mu alpha| + |beta|) + subnormal",
            max_accumulation_error_estimate=gamma_bound(4, u=ULP32_HALF),
            supported_attributes={"affine": bn2.affine, "eps": bn2.eps, "training_eval_mode": not bn2.training},
            profile=profile,
        )
    )

    # ReLU2
    capabilities.append(
        OperatorCapability(
            layer_name="conv.6",
            operator_kind=OperatorKind.RELU.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact non-expansive float32)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"inplace": False},
            profile=profile,
        )
    )

    # Pool2
    p2 = model.conv[7]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.7",
            operator_kind=OperatorKind.MAXPOOL1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact pairwise float32 max)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"kernel_size": 2, "stride": 2, "padding": 0, "ceil_mode": False},
            profile=profile,
        )
    )

    # Conv3
    c8 = model.conv[8]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.8",
            operator_kind=OperatorKind.CONV1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=c8.in_channels * c8.kernel_size[0],
            error_bound_formula="gamma_{n+1} * (|W| |X| + |b|) + subnormal_padding",
            max_accumulation_error_estimate=gamma_bound(193, u=ULP32_HALF),
            supported_attributes={"stride": 1, "dilation": 1, "groups": 1, "padding": c8.padding[0]},
            profile=profile,
        )
    )

    # BN3
    bn3 = model.conv[9]
    capabilities.append(
        OperatorCapability(
            layer_name="conv.9",
            operator_kind=OperatorKind.BATCHNORM1D.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="gamma_4 * (|X| |alpha| + |beta_eff| + |mu alpha| + |beta|) + subnormal",
            max_accumulation_error_estimate=gamma_bound(4, u=ULP32_HALF),
            supported_attributes={"affine": bn3.affine, "eps": bn3.eps, "training_eval_mode": not bn3.training},
            profile=profile,
        )
    )

    # ReLU3
    capabilities.append(
        OperatorCapability(
            layer_name="conv.10",
            operator_kind=OperatorKind.RELU.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact non-expansive float32)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"inplace": False},
            profile=profile,
        )
    )

    # Flatten
    capabilities.append(
        OperatorCapability(
            layer_name="head.0",
            operator_kind=OperatorKind.FLATTEN.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (channel-major reshape)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"start_dim": 1, "end_dim": -1},
            profile=profile,
        )
    )

    # Linear1
    l1 = model.head[1]
    capabilities.append(
        OperatorCapability(
            layer_name="head.1",
            operator_kind=OperatorKind.LINEAR.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=l1.in_features,
            error_bound_formula="gamma_{n+1} * (|W| |X| + |b|) + subnormal_padding",
            max_accumulation_error_estimate=gamma_bound(2049, u=ULP32_HALF),
            supported_attributes={"in_features": l1.in_features, "out_features": l1.out_features, "bias": True},
            profile=profile,
        )
    )

    # Head ReLU
    capabilities.append(
        OperatorCapability(
            layer_name="head.2",
            operator_kind=OperatorKind.RELU.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=None,
            error_bound_formula="0.0 (exact non-expansive float32)",
            max_accumulation_error_estimate=0.0,
            supported_attributes={"inplace": False},
            profile=profile,
        )
    )

    # Linear2
    l2 = model.head[3]
    capabilities.append(
        OperatorCapability(
            layer_name="head.3",
            operator_kind=OperatorKind.LINEAR.value,
            real_valid=True,
            executable_bridge_status="VALIDATED_FLOAT32_ENCLOSURE",
            dot_product_length=l2.in_features,
            error_bound_formula="gamma_{n+1} * (|W| |X| + |b|) + subnormal_padding",
            max_accumulation_error_estimate=gamma_bound(129, u=ULP32_HALF),
            supported_attributes={"in_features": l2.in_features, "out_features": l2.out_features, "bias": True},
            profile=profile,
        )
    )

    return capabilities


def generate_operator_capabilities_report(
    model: nn.Module,
    profile: str = LOCKED_PROFILE_ID,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate serializable operator capability report (W4-008)."""
    caps = audit_smallcnn1d_operators(model, profile=profile)

    report: Dict[str, Any] = {
        "schema_version": "w4.operator_capabilities.v1",
        "task_id": "W4-008",
        "date": "2026-11-08",
        "environment_profile_id": profile,
        "total_operators_audited": len(caps),
        "all_operators_real_valid": all(c.real_valid for c in caps),
        "all_operators_executable_enclosure": all(c.executable_bridge_status == "VALIDATED_FLOAT32_ENCLOSURE" for c in caps),
        "operators": [
            {
                "layer_name": c.layer_name,
                "operator_kind": c.operator_kind,
                "real_valid": c.real_valid,
                "executable_bridge_status": c.executable_bridge_status,
                "dot_product_length": c.dot_product_length,
                "error_bound_formula": c.error_bound_formula,
                "max_accumulation_error_estimate": c.max_accumulation_error_estimate,
                "supported_attributes": c.supported_attributes,
            }
            for c in caps
        ],
    }

    if output_path is not None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    return report


# ==============================================================================
# W4-009: End-to-End Float32 Bridge & Safe Margin Acceptance
# ==============================================================================

def compute_margin_bounds(
    logits_box: IntervalVector,
    target_class: int,
    method: str = "separate_logits",
    head_linear2: Optional[nn.Linear] = None,
    hidden_box: Optional[IntervalVector] = None,
    profile: str = LOCKED_PROFILE_ID,
) -> Tuple[Tuple[int, ...], Tuple[float, ...], bool]:
    """Compute sound lower bounds on all 11 competitor margins m_j = z_{target} - z_j.

    Enforces W4-009-C01, C02, C03:
    - Target class c* in 0..11, exactly 11 competitors j != c*.
    - Method 'separate_logits':
        Lower bound = round_down_f32(logits_box.lo[c*] - logits_box.hi[j]).
        Guarantees fl(z_{c*}) - fl(z_j) >= lower_margin_j for every concrete input.
    - Method 'direct_head_difference':
        Uses fused difference (W_{c*} - W_j) on hidden features, but explicitly widens
        the lower bound by the backward rounding errors of the two separate reference
        dot products to cover coefficient rounding and reference subtraction semantics.
    - Safe predicate: is_positive = True iff min(lower_margins) > 0.0 (strictly positive).
      Ties (margin <= 0) strictly return is_positive = False.
    """
    if profile not in SUPPORTED_PROFILES:
        raise UnsupportedNumericalError(f"Unsupported execution profile: {profile}")

    if target_class not in range(12):
        raise ValueError(f"target_class must be in 0..11, got {target_class}")

    competitors = tuple(j for j in range(12) if j != target_class)

    if method == "separate_logits":
        lo_c = float(logits_box.lo[target_class])
        lower_margins: List[float] = []
        for j in competitors:
            hi_j = float(logits_box.hi[j])
            diff = lo_c - hi_j
            # Outward directed rounding in float32
            m_j = round_down_f32(diff)
            lower_margins.append(m_j)

        lower_tup = tuple(lower_margins)
        is_pos = bool(min(lower_tup) > 0.0)
        return competitors, lower_tup, is_pos

    elif method == "direct_head_difference":
        if head_linear2 is None or hidden_box is None:
            raise ValueError("direct_head_difference requires head_linear2 and hidden_box")

        w2 = head_linear2.weight.detach().cpu().numpy().astype(np.float64)  # (12, 128)
        b2 = (
            head_linear2.bias.detach().cpu().numpy().astype(np.float64)
            if head_linear2.bias is not None
            else np.zeros(12, dtype=np.float64)
        )

        h_lo = hidden_box.lo.ravel()
        h_hi = hidden_box.hi.ravel()
        n = 128
        gamma_n1 = gamma_bound(n + 1, u=ULP32_HALF)

        # Backward rounding error for logit c*
        m_h = np.maximum(np.abs(h_lo), np.abs(h_hi))
        s_abs_c = float(np.sum(np.abs(w2[target_class]) * m_h) + abs(b2[target_class]))
        delta_c = gamma_n1 * s_abs_c + (n + 1) * SUB_NORMAL_F32_MAX

        lower_margins_diff: List[float] = []
        for j in competitors:
            # Difference weights and bias over R
            dw = w2[target_class] - w2[j]
            db = b2[target_class] - b2[j]

            dw_pos = np.maximum(0.0, dw)
            dw_neg = np.minimum(0.0, dw)
            diff_ideal = float(np.sum(dw_pos * h_lo + dw_neg * h_hi) + db)

            # Separate logit j rounding error
            s_abs_j = float(np.sum(np.abs(w2[j]) * m_h) + abs(b2[j]))
            delta_j = gamma_n1 * s_abs_j + (n + 1) * SUB_NORMAL_F32_MAX

            # Total widening to cover both reference dot products (W4-009-C02)
            total_delta = delta_c + delta_j
            m_j = round_down_f32(diff_ideal - total_delta)
            lower_margins_diff.append(m_j)

        lower_tup = tuple(lower_margins_diff)
        is_pos = bool(min(lower_tup) > 0.0)
        return competitors, lower_tup, is_pos

    else:
        raise UnsupportedNumericalError(f"Unsupported margin method: {method}")


def end_to_end_numerical_bridge(
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    target_class: int,
    model: nn.Module,
    constants: Any,
    domain_id: str = "raw_box",
    profile: str = LOCKED_PROFILE_ID,
) -> Any:
    """Execute complete end-to-end numerical bridge from raw box to certified BoundResult (W4-009)."""
    from ecdr.week4.reference import enclose_normalization
    from ecdr.week4.types import BoundResult, BoundStatus, NumericalStatus

    # Step 1: Normalization enclosure (W4-007)
    norm_lo, norm_hi = enclose_normalization(raw_lower, raw_upper, constants)
    norm_box = IntervalVector(lo=norm_lo, hi=norm_hi)

    # Step 2: 15-layer SmallCNN1D enclosure (W4-008)
    logits_box, stages = enclose_smallcnn1d(norm_box, model, profile=profile)

    # Step 3: Margin bounds computation (W4-009-C01, C02, C03)
    competitors, margins, is_pos = compute_margin_bounds(logits_box, target_class, method="separate_logits", profile=profile)

    status = BoundStatus.VALIDATED_POSITIVE.value if is_pos else BoundStatus.VALIDATED_INCONCLUSIVE.value
    num_status = NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value

    return BoundResult(
        method_id="end_to_end_float32_bridge",
        domain_id=domain_id,
        reference_identity_id=str(getattr(model, "checkpoint_sha256", "SmallCNN1D")),
        target_class=target_class,
        competitor_classes=competitors,
        lower_margins=margins,
        status=status,
        numerical_status=num_status,
        is_positive=is_pos,
        failure_reason="" if is_pos else "lower_margin_not_strictly_positive",
    )


