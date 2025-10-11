from typing import Sequence, Union

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Energy distance utilities
# --------------------------------------------------------------------------------------


def multivariate_energy_distance(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    *,
    unbiased: bool = True,
    chunk_threshold: int = 200_000_000,
) -> float:
    """Compute (optionally unbiased) multivariate energy distance between two samples.

    The population energy distance between distributions X and Y is:
        ED(X, Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||

    This function returns a *sample* estimate. By default (``unbiased=True``) it
    uses the U-statistic versions for the intra-sample expectations that exclude
    the diagonal (i == j) terms. If ``unbiased=False`` it uses the simple mean of
    all pairwise distances including zeros on the diagonal (biased downward for
    small sample sizes).

    Parameters
    ----------
    Z_A : np.ndarray, shape (n, d)
        First sample.
    Z_B : np.ndarray, shape (m, d)
        Second sample.
    unbiased : bool, default=True
        Whether to use the unbiased U-statistic estimator for intra distances.
    chunk_threshold : int, default=200_000_000
        If n * m exceeds this threshold, the cross-distance mean is computed in
        chunks to reduce peak memory usage.

    Returns
    -------
    float
        Estimated energy distance.

    Raises
    ------
    ValueError
        If inputs are not 2D numeric arrays or have incompatible feature dims
        or contain NaNs / Infs.
    """
    # Validation
    if not isinstance(Z_A, np.ndarray) or not isinstance(Z_B, np.ndarray):
        raise ValueError("Z_A and Z_B must be NumPy arrays")
    if Z_A.ndim != 2 or Z_B.ndim != 2:
        raise ValueError("Z_A and Z_B must be 2D arrays")
    if Z_A.shape[1] != Z_B.shape[1]:
        raise ValueError("Feature dimension mismatch between Z_A and Z_B")
    if Z_A.shape[0] < 2 or Z_B.shape[0] < 2:
        raise ValueError("Each sample must contain at least 2 observations")
    if not np.isfinite(Z_A).all() or not np.isfinite(Z_B).all():
        raise ValueError("Inputs must not contain NaN or Inf values")

    n, m = Z_A.shape[0], Z_B.shape[0]

    # Cross mean distance (chunked if needed)
    prod = n * m
    if prod <= chunk_threshold:
        cross_dists = np.linalg.norm(Z_A[:, None, :] - Z_B[None, :, :], axis=-1)
        mean_cross_dists = cross_dists.mean()
    else:
        # Stream over B in blocks
        block_rows = max(1, chunk_threshold // n)
        accum = 0.0
        count = 0
        for start in range(0, m, block_rows):
            stop = min(m, start + block_rows)
            block = Z_B[start:stop]
            cross_block = np.linalg.norm(Z_A[:, None, :] - block[None, :, :], axis=-1)
            accum += cross_block.sum()
            count += cross_block.size
        mean_cross_dists = accum / count

    # Intra distances
    def compute_intra_mean(Z: np.ndarray, unbiased_flag: bool) -> float:
        """Compute intra-sample mean distance with memory optimization."""
        n_samples = Z.shape[0]

        # For small samples, use full matrix
        if n_samples * n_samples <= chunk_threshold:
            dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
            if unbiased_flag and n_samples > 1:
                # Exclude diagonal and use U-statistic
                sum_dists = dists[np.triu_indices(n_samples, k=1)].sum()
                return (2.0 / (n_samples * (n_samples - 1))) * sum_dists
            else:
                return dists.mean()

        # For large samples, compute incrementally with proper blocking
        total_sum = 0.0
        total_count = 0

        # Calculate block size to stay within memory limits
        block_size = max(1, int(np.sqrt(chunk_threshold // n_samples)))

        for i in range(0, n_samples, block_size):
            i_end = min(i + block_size, n_samples)
            Z_i = Z[i:i_end]

            # Process blocks in upper triangular pattern to avoid double counting
            for j in range(i, n_samples, block_size):
                j_end = min(j + block_size, n_samples)
                Z_j = Z[j:j_end]

                # Compute distances between blocks
                block_dists = np.linalg.norm(Z_i[:, None, :] - Z_j[None, :, :], axis=-1)

                if i == j:
                    # Diagonal block
                    if unbiased_flag:
                        # Only upper triangle (excluding diagonal)
                        mask = np.triu(np.ones_like(block_dists, dtype=bool), k=1)
                        block_sum = block_dists[mask].sum()
                        block_count = mask.sum()
                    else:
                        # All elements including diagonal
                        block_sum = block_dists.sum()
                        block_count = block_dists.size
                else:
                    # Off-diagonal block - all elements, counted once
                    block_sum = block_dists.sum()
                    block_count = block_dists.size

                total_sum += block_sum
                total_count += block_count

        if unbiased_flag and n_samples > 1:
            # For unbiased: we've summed upper triangle, but need mean of all i≠j pairs
            # Upper triangle has n(n-1)/2 pairs, but we want mean over n(n-1) pairs
            return (2.0 * total_sum) / (n_samples * (n_samples - 1))
        else:
            return total_sum / (n_samples * n_samples) if total_count > 0 else 0.0

    mean_intra_A = compute_intra_mean(Z_A, unbiased)
    mean_intra_B = compute_intra_mean(Z_B, unbiased)

    ED = 2.0 * mean_cross_dists - mean_intra_A - mean_intra_B
    return float(ED)


# --------------------------------------------------------------------------------------
# Permutation utilities
# --------------------------------------------------------------------------------------


def permutation_test(
    Z: Union[pd.DataFrame, np.ndarray],
    n_permutations: int,
    *,
    n_A: int | None = None,
    random_state: int | None = None,
    unbiased: bool = True,
    chunk_threshold: int = 200_000_000,
) -> list[float]:
    """Generate a null distribution of energy distances by label permutation.

    Parameters
    ----------
    Z : DataFrame | np.ndarray, shape (n_total, d)
        Concatenated sample (e.g., [A; B]). If a DataFrame is given, its numeric
        columns are used (non-numeric columns raise an error).
    n_permutations : int
        Number of permutations to perform.
    n_A : int | None, default=None
        Original size of group A. If None, splits floor(n/2) / ceil(n/2).
    random_state : int | None, default=None
        Seed for reproducibility.
    unbiased : bool, default=True
        Passed to energy distance estimator.
    chunk_threshold : int, default=200_000_000
        Threshold forwarded to energy distance estimator for chunking.

    Returns
    -------
    list[float]
        Energy distance values under the null (permuted labels).

    Raises
    ------
    ValueError
        If inputs invalid or n_permutations < 1.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if isinstance(Z, pd.DataFrame):
        Z_numeric = Z.select_dtypes(include=[np.number])
        if Z_numeric.shape[1] != Z.shape[1]:
            raise ValueError(
                "All columns in DataFrame must be numeric for permutation test"
            )
        Z_arr = Z_numeric.to_numpy()
    else:
        Z_arr = Z
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 2D")
    n_total = Z_arr.shape[0]
    if n_total < 4:
        raise ValueError("Need at least 4 rows for a meaningful permutation test")
    if n_A is None:
        n_A = n_total // 2
    if not (2 <= n_A <= n_total - 2):
        raise ValueError(
            f"n_A must be between 2 and {n_total - 2} inclusive (got {n_A})"
        )
    rng = np.random.default_rng(random_state)
    idx = np.arange(n_total)
    out: list[float] = []
    for _ in range(n_permutations):
        rng.shuffle(idx)
        A_idx = idx[:n_A]
        B_idx = idx[n_A:]
        dist = multivariate_energy_distance(
            Z_arr[A_idx],
            Z_arr[B_idx],
            unbiased=unbiased,
            chunk_threshold=chunk_threshold,
        )
        out.append(dist)
    return out


def calculate_p_value(
    null_stats: Sequence[float],
    observed_stat: float,
    *,
    smooth: bool = True,
) -> float:
    """Compute a (optionally smoothed) permutation p-value from a null distribution.

    Parameters
    ----------
    null_stats : sequence of float
        Sampled statistics under the null hypothesis.
    observed_stat : float
        Observed empirical statistic to compare against the null distribution.
    smooth : bool, default=True
        If True apply +1 / (N+1) smoothing (Phipson & Smyth 2010) to avoid zero p-values.
    """
    if len(null_stats) == 0:
        raise ValueError("null_stats must not be empty")
    arr = np.asarray(null_stats, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("null_stats contain non-finite values")
    extreme = np.sum(arr >= observed_stat)
    p_val = (extreme + 1) / (len(arr) + 1) if smooth else extreme / len(arr)
    return float(p_val)


# Public API
__all__ = [
    "calculate_p_value",
    "multivariate_energy_distance",
    "permutation_test",
]
