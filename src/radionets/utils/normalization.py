from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = [
    "_normalize",
    "_denormalize",
    "clamp_outliers",
]


def _normalize(
    x: np.ndarray | torch.Tensor,
    method: str = "minmax",
    epsilon: float = 1e-7,
    clamp_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize an input array or tensor.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Array-like input data.
    method : str, optional
        Normalization method. ``"zscore"`` centers and scales by global
        mean/std. ``"minmax"`` scales to ``[0, 1]`` via min/max.
        Default: ``"minmax"``.
    epsilon : float, optional
        Small constant added to the denominator to avoid division by zero.
        Default: ``1e-7``.
    clamp_sigma
        If given, cap outlier values to ``[mean - clamp_sigma * std,
        mean + clamp_sigma * std]`` before normalization. If ``None``,
        skip clamping. Default: ``None``

    Returns
    -------
    tuple[np.ndarray | torch.Tensor, dict]
        A pair of ``(normalized, params)`` where ``params`` is a dict
        containing at least ``"method"``, ``"mean"``, ``"std"``,
        ``"min"``, and ``"max"``.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"zscore"`` or ``"minmax"``.
    """
    if method not in ("zscore", "minmax"):
        raise ValueError(f"Unknown normalization method {method!r}.")

    orig_dtype: Any | None = None
    device: str | torch.device | int | None = None

    if isinstance(x, torch.Tensor):
        orig_dtype = x.dtype
        device = x.device
        x_np = x.cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float64)
        if isinstance(x, np.ndarray):
            orig_dtype = x.dtype

    has_inf = np.isinf(x_np)

    mean_val = float(x_np.mean())
    std_val = float(x_np.std())
    min_val = float(x_np.min())
    max_val = float(x_np.max())

    if clamp_sigma is not None and std_val > 0 and np.isfinite(std_val):
        lo = mean_val - clamp_sigma * std_val
        hi = mean_val + clamp_sigma * std_val
        x_np = np.clip(x_np, lo, hi)
        # Recompute stats after clamping
        mean_val = float(x_np.mean())
        std_val = float(x_np.std())
        min_val = float(x_np.min())
        max_val = float(x_np.max())

    params: dict[str, Any] = {
        "method": method,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "epsilon": epsilon,
    }

    if std_val == 0.0 or not np.isfinite(std_val):
        normalized = x_np.copy().astype(np.float64)
    elif method == "zscore":
        normalized = (x_np - mean_val) / (std_val + epsilon)
        normalized = normalized.astype(np.float64)
    else:
        # minmax
        range_val = max_val - min_val
        if range_val == 0.0 or not np.isfinite(range_val):
            normalized = np.zeros_like(x_np, dtype=np.float64)
        else:
            normalized = (x_np - min_val) / (range_val + epsilon)
            normalized = normalized.astype(np.float64)

    # Preserve inf values through normalization (keep their original sign)
    normalized = np.where(has_inf, np.sign(x_np) * np.inf, normalized)

    if isinstance(x, torch.Tensor):
        normalized = torch.from_numpy(normalized).to(device)
        normalized = normalized.to(orig_dtype)  # ty:ignore[no-matching-overload]
    else:
        normalized = normalized.astype(orig_dtype)  # ty:ignore[no-matching-overload]

    return torch.as_tensor(normalized), params


def _denormalize(
    x: np.ndarray | torch.Tensor,
    params: dict[str, Any],
) -> torch.Tensor:
    """Reverse a prior :func:`_normalize` call.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Array-like normalized data.
    params : dict
        The params dict returned by :func:`_normalize`. Must contain
        ``"method"``, ``"mean"``, and ``"std"`` keys.

    Returns
    -------
    np.ndarray or torch.Tensor
        Data restored to the original scale.

    Raises
    ------
    KeyError
        If ``params`` is missing required keys.
    ValueError
        If ``method`` is not one of ``"zscore"`` or ``"minmax"``.
    """
    method = params["method"]
    mean_val = params["mean"]
    std_val = params["std"]
    epsilon = params.get("epsilon", 1e-7)
    min_val = params.get("min")
    max_val = params.get("max")

    if method not in ("zscore", "minmax"):
        raise ValueError(f"Unknown normalization method {method!r}.")

    if isinstance(x, torch.Tensor):
        x_np = x.cpu().detach().numpy()
        device = x.device
    else:
        x_np = np.asarray(x)
        device = None

    if method == "zscore":
        reconstructed = x_np * (std_val + epsilon) + mean_val
    elif method == "minmax":
        if min_val is None or max_val is None:
            raise KeyError("minmax denormalization requires 'min' and 'max' in params.")
        range_val = max_val - min_val
        if range_val == 0.0:
            # All original values were identical (== min == max).
            # Restore to max_val (the original constant value).
            reconstructed = np.full_like(x_np, max_val)
        else:
            reconstructed = x_np * (range_val + epsilon) + min_val
    else:
        raise ValueError(f"Unknown normalization method {method!r}.")

    if device is not None:
        assert isinstance(x, torch.Tensor)
        reconstructed = torch.from_numpy(reconstructed).to(device)
        reconstructed = reconstructed.to(x.dtype)
    elif isinstance(x, np.ndarray):
        reconstructed = reconstructed.astype(x.dtype)

    return torch.as_tensor(reconstructed)


class Normalize(nn.Module):
    """Bundles :func:`_normalize` and :func:`_denormalize` as a ``nn.Module``.

    Parameters
    ----------
    method : str, optional
        Normalization method. ``"zscore"`` centers and scales by global
        mean/std. ``"minmax"`` scales to ``[0, 1]`` via min/max.
        Default: ``"minmax"``.
    epsilon : float, optional
        Small constant added to the denominator to avoid division by zero.
        Default: ``1e-7``.
    clamp_sigma : float or None
        If given, cap outlier values to ``[mean - clamp_sigma * std,
        mean + clamp_sigma * std]`` before normalization. If ``None``,
        skip clamping. Default: ``None``
    """

    _params: torch.Tensor
    _method: str

    def __init__(
        self,
        method: str = "zscore",
        epsilon: float = 1e-7,
        clamp_sigma: float | None = None,
    ) -> None:
        """Initialize the normalization module."""
        super().__init__()
        self.method = method
        self.epsilon = epsilon
        self.clamp_sigma = clamp_sigma
        self.register_buffer("_params", torch.zeros(6, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize *x* and store params for later denormalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        normalized : torch.Tensor
            Normalized tensor.
        """
        normalized, params = _normalize(
            x,
            method=self.method,
            epsilon=self.epsilon,
            clamp_sigma=self.clamp_sigma,
        )
        # Store params as a 1-d float tensor for state_dict compat
        self._params = torch.tensor(
            [
                params["mean"],
                params["std"],
                params["min"],
                params["max"],
                params["epsilon"],
                1.0 if params["method"] == "zscore" else 2.0,
            ],
            dtype=torch.float32,
            device=x.device,
        )
        self._method = params["method"]
        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize `x` using stored parameters.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor.

        Returns
        -------
        torch.Tensor
            Restored tensor.
        """
        params = {
            "method": self._method,
            "mean": float(self._params[0]),
            "std": float(self._params[1]),
            "min": float(self._params[2]),
            "max": float(self._params[3]),
            "epsilon": float(self._params[4]),
        }
        return _denormalize(x, params)


def clamp_outliers(
    x: np.ndarray | torch.Tensor,
    sigma: float = 3.0,
) -> np.ndarray | torch.Tensor:
    """Clip outlier values to ``[median - sigma*MAD, median + sigma*MAD]``,
    where MAD is the median absolute deviation.


    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input array or tensor.
    sigma : float
        Number of MADs for clipping. Default: ``3.0``.

    Returns
    -------
    np.ndarray or torch.Tensor
        Clipped data.
    """
    if isinstance(x, torch.Tensor):
        median_val = float(x.median())
        mad = float((x - median_val).abs().median())
    else:
        median_val = float(np.median(x))
        mad = float(np.median(np.abs(x - median_val)))

    scale = 1.4826  # MAD to std conversion for Gaussian data
    std_val = mad * scale

    if std_val == 0.0:
        return x.copy() if isinstance(x, np.ndarray) else x.clone()

    lo = median_val - sigma * std_val
    hi = median_val + sigma * std_val

    if isinstance(x, torch.Tensor):
        return x.clamp(lo, hi)
    else:
        result = np.clip(x, lo, hi)
        assert isinstance(result, np.ndarray)
        return result
