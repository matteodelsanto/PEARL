from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike1D = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class HistogramConfig:
    """Configuration for histogram embedding."""
    bins: int = 20
    # Choose ONE strategy:
    # - fixed_range: (min, max) used as-is
    # - percentile_range: (low_pct, high_pct) computed from training distribution
    fixed_range: Optional[Tuple[float, float]] = None
    percentile_range: Optional[Tuple[float, float]] = (1.0, 99.0)
    # Small constant for numerical stability where needed
    eps: float = 1e-12


def _to_clean_1d_array(x: ArrayLike1D) -> np.ndarray:
    """
    Convert input to a clean 1D numpy array:
    - float dtype
    - removes NaN/inf
    """
    arr = np.asarray(list(x), dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    return arr[mask]


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(arr: np.ndarray) -> float:
    # ddof=1 for sample std; fallback to 0 when length < 2
    if arr.size < 2:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def _safe_quantile(arr: np.ndarray, q: float) -> float:
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _linear_slope(arr: np.ndarray) -> float:
    """
    Slope of arr over index (0..n-1) via least squares.
    Returns 0 if not enough points.
    """
    n = arr.size
    if n < 2:
        return 0.0 if n == 1 else float("nan")
    t = np.arange(n, dtype=float)
    # Fit slope: cov(t, y) / var(t)
    t_mean = t.mean()
    y_mean = arr.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom == 0:
        return 0.0
    slope = np.sum((t - t_mean) * (arr - y_mean)) / denom
    return float(slope)


def _autocorr_lag1(arr: np.ndarray) -> float:
    """
    Lag-1 autocorrelation. Returns 0 if not enough points.
    """
    n = arr.size
    if n < 3:
        return 0.0 if n >= 1 else float("nan")
    x = arr[:-1]
    y = arr[1:]
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _burst_features(arr: np.ndarray, z_thresh: float = 2.0) -> Dict[str, float]:
    """
    Simple "burstiness" indicators based on z-score threshold:
    - fraction_above: fraction of points above mean + z_thresh * std
    - max_run_above: max consecutive run length above that threshold
    """
    out: Dict[str, float] = {}
    if arr.size == 0:
        return {
            "burst_fraction_above": float("nan"),
            "burst_max_run_above": float("nan"),
        }
    mu = arr.mean()
    sd = np.std(arr, ddof=1) if arr.size >= 2 else 0.0
    thr = mu + z_thresh * sd
    above = arr > thr
    frac = float(np.mean(above)) if above.size else float("nan")

    # Max consecutive run of True
    max_run = 0
    run = 0
    for v in above:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    out["burst_fraction_above"] = frac
    out["burst_max_run_above"] = float(max_run)
    return out


def compute_summary_pooling(
    seq: ArrayLike1D,
    prefix: str,
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    include_autocorr: bool = True,
    include_bursts: bool = True,
    burst_z: float = 2.0,
) -> Dict[str, float]:
    """
    Pool a variable-length numeric sequence into a fixed set of features.

    Parameters
    ----------
    seq : sequence of floats
        Any 1D numeric sequence (e.g., surprisal values).
    prefix : str
        Feature prefix, e.g. "S_HC" or "H_delta".
    quantiles : tuple
        Quantiles to compute on the sequence.
    include_autocorr : bool
        Add lag-1 autocorrelation.
    include_bursts : bool
        Add simple burstiness features.
    burst_z : float
        z-threshold for burst detection (mean + burst_z * std).

    Returns
    -------
    Dict[str, float]
        Mapping feature_name -> value.
    """
    arr = _to_clean_1d_array(seq)

    feats: Dict[str, float] = {}
    feats[f"{prefix}_n"] = float(arr.size)

    if arr.size == 0:
        # Fill with NaNs to keep consistent feature set
        feats[f"{prefix}_mean"] = float("nan")
        feats[f"{prefix}_std"] = float("nan")
        feats[f"{prefix}_min"] = float("nan")
        feats[f"{prefix}_max"] = float("nan")
        feats[f"{prefix}_slope"] = float("nan")
        if include_autocorr:
            feats[f"{prefix}_autocorr_lag1"] = float("nan")
        for q in quantiles:
            feats[f"{prefix}_q{int(q*100):02d}"] = float("nan")
        if include_bursts:
            feats[f"{prefix}_burst_fraction_above"] = float("nan")
            feats[f"{prefix}_burst_max_run_above"] = float("nan")
        return feats

    feats[f"{prefix}_mean"] = _safe_mean(arr)
    feats[f"{prefix}_std"] = _safe_std(arr)
    feats[f"{prefix}_min"] = float(np.min(arr))
    feats[f"{prefix}_max"] = float(np.max(arr))
    feats[f"{prefix}_slope"] = _linear_slope(arr)

    for q in quantiles:
        feats[f"{prefix}_q{int(q*100):02d}"] = _safe_quantile(arr, q)

    if include_autocorr:
        feats[f"{prefix}_autocorr_lag1"] = _autocorr_lag1(arr)

    if include_bursts:
        b = _burst_features(arr, z_thresh=burst_z)
        for k, v in b.items():
            feats[f"{prefix}_{k}"] = v

    return feats


def compute_window_pooling(
    seq: ArrayLike1D,
    prefix: str,
    n_windows: int = 5,
    window_quantiles: Tuple[float, ...] = (0.5, 0.9),
) -> Dict[str, float]:
    """
    Pool sequence into K windows to preserve coarse temporal behavior.
    Each window produces a small set of summary features.

    Strategy: split indices into K contiguous chunks (as evenly as possible).
    """
    arr = _to_clean_1d_array(seq)
    feats: Dict[str, float] = {}

    if n_windows <= 0:
        raise ValueError("n_windows must be >= 1")

    if arr.size == 0:
        # Fill windows with NaNs
        for w in range(n_windows):
            feats[f"{prefix}_w{w}_n"] = 0.0
            feats[f"{prefix}_w{w}_mean"] = float("nan")
            feats[f"{prefix}_w{w}_std"] = float("nan")
            for q in window_quantiles:
                feats[f"{prefix}_w{w}_q{int(q*100):02d}"] = float("nan")
        return feats

    # Window boundaries
    indices = np.array_split(np.arange(arr.size), n_windows)
    for w, idx in enumerate(indices):
        win = arr[idx]
        feats[f"{prefix}_w{w}_n"] = float(win.size)
        feats[f"{prefix}_w{w}_mean"] = _safe_mean(win)
        feats[f"{prefix}_w{w}_std"] = _safe_std(win)
        for q in window_quantiles:
            feats[f"{prefix}_w{w}_q{int(q*100):02d}"] = _safe_quantile(win, q)

    return feats


def fit_histogram_range_from_training(
    training_sequences: Iterable[ArrayLike1D],
    config: HistogramConfig,
) -> Tuple[float, float]:
    """
    Compute a stable histogram range from TRAINING data only.
    Use this range later for all subjects (train and test).

    Returns (lo, hi).
    """
    if config.fixed_range is not None:
        return config.fixed_range

    if config.percentile_range is None:
        raise ValueError("Either fixed_range or percentile_range must be set.")

    low_pct, high_pct = config.percentile_range

    all_vals: List[float] = []
    for seq in training_sequences:
        arr = _to_clean_1d_array(seq)
        if arr.size:
            all_vals.append(arr)
    if not all_vals:
        return (0.0, 1.0)

    flat = np.concatenate(all_vals)
    lo = float(np.percentile(flat, low_pct))
    hi = float(np.percentile(flat, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def compute_histogram_embedding(
    seq: ArrayLike1D,
    prefix: str,
    bins: int,
    value_range: Tuple[float, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Histogram embedding: counts in bins, normalized to sum to 1.
    Output features: prefix_hist_00 ... prefix_hist_{bins-1}
    """
    arr = _to_clean_1d_array(seq)
    feats: Dict[str, float] = {}

    lo, hi = value_range
    if bins <= 0:
        raise ValueError("bins must be >= 1")
    if hi <= lo:
        raise ValueError("value_range must have hi > lo")

    if arr.size == 0:
        for b in range(bins):
            feats[f"{prefix}_hist_{b:02d}"] = float("nan")
        return feats

    # Clip to range so outliers don't dominate
    clipped = np.clip(arr, lo, hi)
    counts, _ = np.histogram(clipped, bins=bins, range=(lo, hi))
    counts = counts.astype(float)
    denom = counts.sum()
    if denom <= 0:
        denom = eps
    probs = counts / denom

    for b in range(bins):
        feats[f"{prefix}_hist_{b:02d}"] = float(probs[b])

    return feats


def build_feature_embedding(
    seq: ArrayLike1D,
    prefix: str,
    *,
    do_summary: bool = True,
    do_windows: bool = True,
    n_windows: int = 5,
    do_hist: bool = True,
    hist_bins: int = 20,
    hist_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    One-stop function: build a feature embedding from ANY numeric sequence.

    You can enable/disable:
    - summary pooling (1)
    - window pooling (2)
    - histogram embedding (3)

    Note: for histogram embedding, pass a hist_range computed on training data.
    """
    feats: Dict[str, float] = {}

    if do_summary:
        feats.update(compute_summary_pooling(seq, prefix=prefix))

    if do_windows:
        feats.update(
            compute_window_pooling(
                seq, prefix=f"{prefix}_win", n_windows=n_windows
            )
        )

    if do_hist:
        if hist_range is None:
            raise ValueError(
                "hist_range is required for histogram embedding. "
                "Compute it from training data via fit_histogram_range_from_training()."
            )
        feats.update(
            compute_histogram_embedding(
                seq, prefix=f"{prefix}_hist", bins=hist_bins, value_range=hist_range
            )
        )

    return feats


def compute_delta_features(
    seq1: ArrayLike1D,
    seq2: ArrayLike1D,
    prefix: str = "D",
    hist_range: Optional[Tuple[float, float]] = None,
    *,
    do_summary: bool = True,
    do_windows: bool = True,
    n_windows: int = 5,
    do_hist: bool = True,
    hist_bins: int = 20,
) -> Dict[str, float]:
    """
    Calcola feature sulla differenza di due sequenze: delta = seq2 - seq1
    
    Parameters
    ----------
    seq1 : sequence of floats
        Baseline sequence (e.g., CN surprisal).
    seq2 : sequence of floats
        Comparison sequence (e.g., AD surprisal).
    prefix : str
        Feature prefix for delta features, default "D".
    hist_range : Optional[Tuple[float, float]]
        Range for histogram binning. If None, histogram is skipped.
    do_summary : bool
        Include summary pooling.
    do_windows : bool
        Include window pooling.
    n_windows : int
        Number of windows.
    do_hist : bool
        Include histogram embedding.
    hist_bins : int
        Number of histogram bins.
    
    Returns
    -------
    Dict[str, float]
        Delta feature mapping.
    """
    arr1 = _to_clean_1d_array(seq1)
    arr2 = _to_clean_1d_array(seq2)
    
    if arr1.size == 0 or arr2.size == 0:
        return {}
    
    if arr1.size != arr2.size:
        return {}
    
    # Compute delta
    delta = arr2 - arr1
    
    # Build embedding from delta sequence
    feats: Dict[str, float] = {}
    
    if do_summary:
        feats.update(compute_summary_pooling(delta, prefix=prefix))
    
    if do_windows:
        feats.update(
            compute_window_pooling(
                delta, prefix=f"{prefix}_win", n_windows=n_windows
            )
        )
    
    if do_hist and hist_range is not None:
        feats.update(
            compute_histogram_embedding(
                delta, prefix=f"{prefix}_hist", bins=hist_bins, value_range=hist_range
            )
        )
    
    return feats


def compute_histogram_ranges(
    sequences: Iterable[ArrayLike1D],
    config: HistogramConfig,
) -> Tuple[float, float]:
    """
    Alias per fit_histogram_range_from_training per compatibilità.
    Compute a stable histogram range from sequences.
    
    Returns (lo, hi).
    """
    return fit_histogram_range_from_training(sequences, config)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example sequences (e.g., surprisal list for a subject)
    S_hc = [1.2, 0.9, 1.0, 3.1, 0.8, 1.4, 2.2]*50
    S_ad = [1.5, 1.1, 0.7, 2.9, 1.3, 1.9, 2.6]*50
    S_delta = np.array(S_hc) - np.array(S_ad)

    # Suppose you computed histogram range on TRAINING set for this feature type
    # Here we just hardcode one for demo
    hist_range = (0.0, 5.0)

    emb_hc = build_feature_embedding(S_hc, prefix="S_HC", hist_range=hist_range)
    emb_ad = build_feature_embedding(S_ad, prefix="S_AD", hist_range=hist_range)
    emb_d  = build_feature_embedding(S_delta, prefix="S_DELTA", hist_range=(-5.0, 5.0))

    # Combine embeddings for a classifier input vector
    combined = {**emb_hc, **emb_ad, **emb_d}
    
    print("Feature embedding:")
    print("------------------")
    print(combined)
    print("------------------")

    # Turn into a stable vector (sorted keys)
    keys = sorted(combined.keys())
    x_vec = np.array([combined[k] for k in keys], dtype=float)

    print("n_features =", x_vec.size)
    print("first 10 features:", [(k, combined[k]) for k in keys[:10]])
    
    #stampa tutte le features una per riga
    print("\nAll features:")
    for k in keys:
        print(f"{k}: {combined[k]}")