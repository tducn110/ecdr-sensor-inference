"""preprocessing.py — Normalization and Preprocessing Contract.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 2.2).

Equation (2.2):
  Z_{c, i} = P(X)_{c, i} = (X_{c, i} - mu_c) / sigma_c

Key contract rules:
1. Normalization statistics (mu, sigma) are fit exclusively on the TRAIN split.
2. sigma_c is strictly positive, clamped with sigma_min > 0.
3. Affine coordinate transform preserves window overlap:
   If X'_{c, i} = X_{c, i+s}, then Z'_{c, i} = Z_{c, i+s}.
4. Raw box radius b_raw transforms to normalized box radius b_norm = b_raw / sigma.
5. Invariant hash binds the exact preprocessing pipeline identity.
6. Strict validation on finiteness, dimensions, and non-negative budgets.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
from ecdr.schema import CHANNEL_NAMES


@dataclass(frozen=True)
class PreprocessingContract:
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    sigma_min: float
    num_channels: int
    window_length: int
    prep_id: str

    @classmethod
    def create(
        cls,
        mean: Union[np.ndarray, list, tuple],
        std: Union[np.ndarray, list, tuple],
        sigma_min: float = 1e-5,
        num_channels: int = 6,
        window_length: int = 128,
    ) -> "PreprocessingContract":
        mean_arr = np.asarray(mean, dtype=np.float64).flatten()
        std_arr = np.asarray(std, dtype=np.float64).flatten()

        # Strict validation of declared dimensions
        if len(mean_arr) != num_channels:
            raise ValueError(f"Expected {num_channels} mean values, got {len(mean_arr)}")
        if len(std_arr) != num_channels:
            raise ValueError(f"Expected {num_channels} std values, got {len(std_arr)}")

        # Strict finiteness validation
        if not np.all(np.isfinite(mean_arr)):
            raise ValueError("Mean contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(std_arr)):
            raise ValueError("Std contains non-finite values (NaN or Inf)")
        if np.any(std_arr < 0):
            raise ValueError("Std values cannot be negative")

        # Validate sigma_min
        if not np.isfinite(sigma_min) or sigma_min <= 0.0:
            raise ValueError(f"sigma_min must be finite and strictly positive, got {sigma_min}")

        # Clamp scale with sigma_min
        clamped_std = np.maximum(std_arr, sigma_min)
        if not np.all(np.isfinite(clamped_std)) or np.any(clamped_std <= 0):
            raise ValueError("Clamped scale contains non-positive or non-finite values")

        # Compute deterministic SHA-256 identity
        ident_dict = {
            "mean": [float(x) for x in mean_arr],
            "std": [float(x) for x in clamped_std],
            "sigma_min": float(sigma_min),
            "num_channels": int(num_channels),
            "window_length": int(window_length),
            "channel_names": list(CHANNEL_NAMES),
        }
        ident_str = json.dumps(ident_dict, sort_keys=True)
        prep_id = hashlib.sha256(ident_str.encode("utf-8")).hexdigest()[:16]

        return cls(
            mean=tuple(float(x) for x in mean_arr),
            std=tuple(float(x) for x in clamped_std),
            sigma_min=float(sigma_min),
            num_channels=num_channels,
            window_length=window_length,
            prep_id=prep_id,
        )

    def _validate_input_shape(self, shape: Tuple[int, ...]) -> None:
        """Validate that input matches declared num_channels and window_length."""
        if len(shape) == 2:
            if shape != (self.num_channels, self.window_length):
                raise ValueError(
                    f"Expected shape ({self.num_channels}, {self.window_length}), got {shape}"
                )
        elif len(shape) == 3:
            if shape[1:] != (self.num_channels, self.window_length):
                raise ValueError(
                    f"Expected shape (B, {self.num_channels}, {self.window_length}), got {shape}"
                )
        else:
            raise ValueError(
                f"Input must be 2D (C, W) or 3D (B, C, W), got {len(shape)}D array"
            )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply P(X) = (X - mu) / sigma to numpy array with strict shape and finiteness checks."""
        x_arr = np.asarray(x, dtype=np.float64)
        self._validate_input_shape(x_arr.shape)

        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        mu = np.array(self.mean, dtype=np.float64)[:, None]
        sigma = np.array(self.std, dtype=np.float64)[:, None]
        normed = (x_arr - mu) / sigma

        if not np.all(np.isfinite(normed)):
            raise ValueError("Normalized output contains non-finite values")

        return normed

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P(X) = (X - mu) / sigma to PyTorch tensor with strict shape and finiteness checks."""
        self._validate_input_shape(tuple(x.shape))

        if not torch.all(torch.isfinite(x)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        mu = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1)
        sigma = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1)
        normed = (x - mu) / sigma

        if not torch.all(torch.isfinite(normed)):
            raise ValueError("Normalized output contains non-finite values")

        return normed

    def raw_to_norm_budget(self, b_raw: np.ndarray) -> np.ndarray:
        """Convert raw perturbation budget to normalized budget: b_norm = b_raw / sigma."""
        b_raw_arr = np.asarray(b_raw, dtype=np.float64)
        if not np.all(np.isfinite(b_raw_arr)):
            raise ValueError("Raw budget must be finite")
        if np.any(b_raw_arr < 0):
            raise ValueError("Raw budget must be non-negative")

        sigma = np.array(self.std, dtype=np.float64)
        if b_raw_arr.ndim == 1 and len(b_raw_arr) == self.num_channels:
            res = b_raw_arr / sigma
        elif b_raw_arr.ndim == 2 and b_raw_arr.shape == (self.num_channels, self.window_length):
            res = b_raw_arr / sigma[:, None]
        else:
            raise ValueError(
                f"Incompatible budget shape {b_raw_arr.shape}; expected ({self.num_channels},) or ({self.num_channels}, {self.window_length})"
            )

        if not np.all(np.isfinite(res)):
            raise ValueError("Normalized budget contains non-finite values")
        return res

    def norm_to_raw_budget(self, b_norm: np.ndarray) -> np.ndarray:
        """Convert normalized perturbation budget to raw budget: b_raw = b_norm * sigma."""
        b_norm_arr = np.asarray(b_norm, dtype=np.float64)
        if not np.all(np.isfinite(b_norm_arr)):
            raise ValueError("Normalized budget must be finite")
        if np.any(b_norm_arr < 0):
            raise ValueError("Normalized budget must be non-negative")

        sigma = np.array(self.std, dtype=np.float64)
        if b_norm_arr.ndim == 1 and len(b_norm_arr) == self.num_channels:
            res = b_norm_arr * sigma
        elif b_norm_arr.ndim == 2 and b_norm_arr.shape == (self.num_channels, self.window_length):
            res = b_norm_arr * sigma[:, None]
        else:
            raise ValueError(
                f"Incompatible budget shape {b_norm_arr.shape}; expected ({self.num_channels},) or ({self.num_channels}, {self.window_length})"
            )

        if not np.all(np.isfinite(res)):
            raise ValueError("Raw budget contains non-finite values")
        return res


def fit_normalization_stats(
    x_train: np.ndarray, sigma_min: float = 1e-5
) -> PreprocessingContract:
    """Fit channel mean and std on train windows only.

    x_train: (N_train, 6, 128)
    Computes per-channel mean and std over axes (0, 2) in float64.
    """
    if x_train.size == 0 or x_train.ndim != 3 or x_train.shape[1] != 6 or x_train.shape[2] != 128:
        raise ValueError(f"Expected nonempty x_train with shape (N, 6, 128), got {x_train.shape}")
    if not np.all(np.isfinite(x_train)):
        raise ValueError("x_train contains non-finite values")

    x_f64 = x_train.astype(np.float64)
    mean = np.mean(x_f64, axis=(0, 2))
    std = np.std(x_f64, axis=(0, 2))
    return PreprocessingContract.create(
        mean=mean,
        std=std,
        sigma_min=sigma_min,
        num_channels=6,
        window_length=x_train.shape[2],
    )
