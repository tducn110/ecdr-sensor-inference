"""intervals.py — Sound float64 interval arithmetic core with per-operation outward rounding.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 11)
and independent review numerical recommendations.

IEEE-754 Directed Outward Rounding:
Every elementary operation (+, -, *, /) performs outward rounding via np.nextafter:
  lower = np.nextafter(computed_lower, -np.inf)
  upper = np.nextafter(computed_upper,  np.inf)

Inclusion property:
If x in [x_lo, x_hi] and y in [y_lo, y_hi], then x (op) y in [res_lo, res_hi].
Non-finite values (NaN, +/-Inf) or inverted intervals (lo > hi) trigger immediate
rejection to ensure strict mathematical conservatism.
All bounds are deep-copied into detached read-only arrays to prevent alias mutation.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class Interval:
    """Interval [lower, upper] stored as detached immutable float64 NumPy arrays."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        # Create detached copies to decouple from caller views and base buffers
        lo_arr = np.copy(np.asarray(self.lo, dtype=np.float64))
        hi_arr = np.copy(np.asarray(self.hi, dtype=np.float64))

        if lo_arr.shape != hi_arr.shape:
            raise ValueError(f"Shape mismatch: {lo_arr.shape} vs {hi_arr.shape}")

        if not (np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr))):
            raise ValueError("Interval bounds must be finite (cannot contain NaN or Inf)")

        if np.any(lo_arr > hi_arr):
            raise ValueError("Interval lower bound must be <= upper bound (no inverted intervals)")

        # Freeze arrays to prevent in-place mutation
        lo_arr.flags.writeable = False
        hi_arr.flags.writeable = False
        object.__setattr__(self, "lo", lo_arr)
        object.__setattr__(self, "hi", hi_arr)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lo.shape

    def is_finite(self) -> bool:
        """Check if all bounds are finite and lo <= hi."""
        return bool(
            np.all(np.isfinite(self.lo))
            and np.all(np.isfinite(self.hi))
            and np.all(self.lo <= self.hi)
        )

    @classmethod
    def from_point(cls, x: np.ndarray) -> "Interval":
        """Degenerate point interval [x, x]."""
        x_f64 = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x_f64)):
            raise ValueError("Point interval input must be finite")
        return cls(lo=np.copy(x_f64), hi=np.copy(x_f64))

    @classmethod
    def from_box(cls, center: np.ndarray, radius: Union[float, np.ndarray]) -> "Interval":
        """Construct interval [center - radius, center + radius] with outward rounding."""
        c = np.asarray(center, dtype=np.float64)
        r = np.asarray(radius, dtype=np.float64)
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
            raise ValueError("Center and radius must be finite")
        if np.any(r < 0):
            raise ValueError("Radius must be non-negative")
        if r.ndim == 1 and c.ndim == 2 and len(r) == c.shape[0]:
            r = r[:, None]
        lo = np.nextafter(c - r, -np.inf)
        hi = np.nextafter(c + r, np.inf)
        return cls(lo=lo, hi=hi)


def interval_add_outward(a: Interval, b: Interval) -> Interval:
    """Interval addition with outward rounding: [a_lo + b_lo, a_hi + b_hi]."""
    lo = np.nextafter(a.lo + b.lo, -np.inf)
    hi = np.nextafter(a.hi + b.hi, np.inf)
    return Interval(lo=lo, hi=hi)


def interval_sub_outward(a: Interval, b: Interval) -> Interval:
    """Interval subtraction with outward rounding: [a_lo - b_hi, a_hi - b_lo]."""
    lo = np.nextafter(a.lo - b.hi, -np.inf)
    hi = np.nextafter(a.hi - b.lo, np.inf)
    return Interval(lo=lo, hi=hi)


def interval_mul_scalar_outward(a: Interval, w: float) -> Interval:
    """Interval multiplied by constant scalar w with outward rounding."""
    w_f64 = float(w)
    if not np.isfinite(w_f64):
        raise ValueError("Scalar multiplier must be finite")
    if w_f64 >= 0.0:
        lo = np.nextafter(w_f64 * a.lo, -np.inf)
        hi = np.nextafter(w_f64 * a.hi, np.inf)
    else:
        lo = np.nextafter(w_f64 * a.hi, -np.inf)
        hi = np.nextafter(w_f64 * a.lo, np.inf)
    return Interval(lo=lo, hi=hi)


def interval_div_scalar_pos_outward(a: Interval, d: float) -> Interval:
    """Interval divided by positive scalar d > 0 with outward rounding."""
    d_f64 = float(d)
    if not np.isfinite(d_f64) or d_f64 <= 0.0:
        raise ValueError(f"Divisor must be finite and strictly positive, got {d_f64}")
    lo = np.nextafter(a.lo / d_f64, -np.inf)
    hi = np.nextafter(a.hi / d_f64, np.inf)
    return Interval(lo=lo, hi=hi)


def interval_relu(a: Interval) -> Interval:
    """ReLU on interval: [max(0, a_lo), max(0, a_hi)]. Exact in IEEE-754."""
    lo = np.maximum(0.0, a.lo)
    hi = np.maximum(0.0, a.hi)
    return Interval(lo=lo, hi=hi)


@dataclass(frozen=True)
class RationalInterval:
    """Exact-rational interval [lo, hi] using Python Fraction for analytical verification."""
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        assert self.lo <= self.hi, f"Inverted rational interval: {self.lo} > {self.hi}"

    @classmethod
    def from_scalar(cls, x: Union[int, float, Fraction, str]) -> "RationalInterval":
        frac = Fraction(x)
        return cls(lo=frac, hi=frac)

    @classmethod
    def from_bounds(
        cls, lo: Union[int, float, Fraction, str], hi: Union[int, float, Fraction, str]
    ) -> "RationalInterval":
        return cls(lo=Fraction(lo), hi=Fraction(hi))

    def add(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(lo=self.lo + other.lo, hi=self.hi + other.hi)

    def sub(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(lo=self.lo - other.hi, hi=self.hi - other.lo)

    def mul_scalar(self, w: Union[int, float, Fraction, str]) -> "RationalInterval":
        w_frac = Fraction(w)
        if w_frac >= 0:
            return RationalInterval(lo=self.lo * w_frac, hi=self.hi * w_frac)
        else:
            return RationalInterval(lo=self.hi * w_frac, hi=self.lo * w_frac)

    def div_scalar(self, d: Union[int, float, Fraction, str]) -> "RationalInterval":
        d_frac = Fraction(d)
        assert d_frac > 0, "Divisor must be positive"
        return RationalInterval(lo=self.lo / d_frac, hi=self.hi / d_frac)

    def relu(self) -> "RationalInterval":
        return RationalInterval(lo=max(Fraction(0), self.lo), hi=max(Fraction(0), self.hi))
