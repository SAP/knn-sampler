from typing import Callable, Sequence, Union

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
    """Compute multivariate energy distance between two samples.

    The multivariate energy distance is a metric for comparing probability
    distributions based on distances between sample points. Uses U-statistic
    estimators by default to reduce finite-sample bias.

    Parameters
    ----------
    Z_A : np.ndarray, shape (n, d)
        First sample with n observations and d features.
    Z_B : np.ndarray, shape (m, d)
        Second sample with m observations and d features.
    unbiased : bool, default=True
        Whether to use unbiased U-statistic estimators for intra-sample distances.
    chunk_threshold : int, default=200_000_000
        Approximate memory (in element * feature * 8 byte units) threshold used
        to decide when to fall back to chunked/block computations.

    Returns
    -------
    float
        Estimated energy distance. Non-negative for independent samples; equals
        0 in expectation when both samples come from the same distribution.

    Raises
    ------
    ValueError
        If inputs are invalid or contain non-finite values.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.random.normal(0, 1, (100, 2))
    >>> B = np.random.normal(1, 1, (100, 2))
    >>> ed = multivariate_energy_distance(A, B)
    >>> ed > 0
    True
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
    n_features = Z_A.shape[1]
    memory_needed = n * m * n_features * 8  # 8 bytes per float64
    if memory_needed <= chunk_threshold:
        cross_dists = np.linalg.norm(Z_A[:, None, :] - Z_B[None, :, :], axis=-1)
        mean_cross_dists = cross_dists.mean()
    else:
        # Stream over B in blocks to manage memory
        max_elements_per_chunk = chunk_threshold // (n_features * 8)
        chunk_size = max(1, max_elements_per_chunk // n)
        chunk_size = min(chunk_size, m)  # Don't exceed B sample size

        total_sum = 0.0
        total_count = 0
        for start in range(0, m, chunk_size):
            stop = min(m, start + chunk_size)
            block = Z_B[start:stop]
            cross_block = np.linalg.norm(Z_A[:, None, :] - block[None, :, :], axis=-1)
            total_sum += cross_block.sum()
            total_count += cross_block.size
        mean_cross_dists = total_sum / total_count

    # Intra distances
    def compute_intra_mean(Z: np.ndarray, unbiased_flag: bool) -> float:
        """Compute intra-sample mean distance with memory optimization.

        For unbiased=True, computes U-statistic estimator excluding diagonal.
        For unbiased=False, computes all pairwise distances including diagonal.
        """
        n_samples = Z.shape[0]
        n_features = Z.shape[1]

        # For small samples, use full matrix
        memory_needed = n_samples * n_samples * n_features * 8  # 8 bytes per float64
        if memory_needed <= chunk_threshold:
            dists = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
            if unbiased_flag and n_samples > 1:
                # Exclude diagonal and use U-statistic
                sum_dists = dists[np.triu_indices(n_samples, k=1)].sum()
                return (2.0 / (n_samples * (n_samples - 1))) * sum_dists
            else:
                return dists.mean()

        # For large samples, compute incrementally with proper blocking
        # Each block pair creates a (block_size x block_size x n_features) array
        max_block_elements = chunk_threshold // (n_features * 8)  # 8 bytes per float64
        block_size = max(1, int(np.sqrt(max_block_elements)))
        block_size = min(block_size, n_samples)

        total_sum = 0.0
        total_pairs = 0

        for i in range(0, n_samples, block_size):
            i_end = min(i + block_size, n_samples)
            Z_i = Z[i:i_end]
            actual_i_size = i_end - i

            for j in range(0, n_samples, block_size):
                j_end = min(j + block_size, n_samples)
                Z_j = Z[j:j_end]
                actual_j_size = j_end - j

                # Compute distances between blocks
                block_dists = np.linalg.norm(Z_i[:, None, :] - Z_j[None, :, :], axis=-1)

                if unbiased_flag:
                    if i == j:
                        # Diagonal block - only upper triangle excluding diagonal
                        mask = np.triu(
                            np.ones((actual_i_size, actual_j_size), dtype=bool), k=1
                        )
                        block_sum = block_dists[mask].sum()
                        block_pairs = mask.sum()
                    elif i < j:
                        # Upper triangular block - all elements
                        block_sum = block_dists.sum()
                        block_pairs = actual_i_size * actual_j_size
                    else:
                        # Lower triangular block - skip to avoid double counting
                        continue
                else:
                    # All pairwise distances including diagonal
                    block_sum = block_dists.sum()
                    block_pairs = actual_i_size * actual_j_size

                total_sum += block_sum
                total_pairs += block_pairs

        if unbiased_flag and n_samples > 1:
            # For unbiased: we've summed upper triangle pairs once (sum_{i<j} d_ij)
            # Unbiased estimator uses (2 / (n(n-1))) * sum_{i<j} d_ij
            return (
                (2.0 * total_sum) / (n_samples * (n_samples - 1))
                if total_pairs > 0
                else 0.0
            )
        else:
            return total_sum / total_pairs if total_pairs > 0 else 0.0

    mean_intra_A = compute_intra_mean(Z_A, unbiased)
    mean_intra_B = compute_intra_mean(Z_B, unbiased)

    return float(2.0 * mean_cross_dists - mean_intra_A - mean_intra_B)


# --------------------------------------------------------------------------------------
# Permutation utilities
# --------------------------------------------------------------------------------------


def permutation_test(
    Z: Union[pd.DataFrame, np.ndarray],
    n_permutations: int,
    *,
    n_A: int | None = None,
    metric_func: Callable[[np.ndarray, np.ndarray], float] | None = None,
    random_state: int | None = None,
    unbiased: bool = True,
    chunk_threshold: int = 200_000_000,
) -> list[float]:
    """Generate null distribution via label permutation for two-sample testing.

    Randomly reassigns group labels to observations and computes test statistics
    for each permutation, creating a null distribution under the hypothesis of
    no difference between groups.

    Parameters
    ----------
    Z : DataFrame | np.ndarray, shape (n_total, d)
        Concatenated observations from both groups.
    n_permutations : int
        Number of random permutations for null distribution generation.
    n_A : int | None, default=None
        Size of first group. If None, uses floor(n_total/2).
    metric_func : callable | None, default=None
        Function to compute test statistic between two groups.
        Should accept two arrays and return a float.
        If None, uses multivariate_energy_distance.
    random_state : int | None, default=None
        Seed for reproducible permutation generation.
    unbiased : bool, default=True
        Passed to energy distance computation when metric_func is None.
    chunk_threshold : int, default=200_000_000
        Memory management parameter for large datasets.

    Returns
    -------
    list[float]
        Null distribution of test statistics under permuted group labels.

    Raises
    ------
    ValueError
        If inputs are invalid or group sizes inappropriate.

    Examples
    --------
    >>> import pandas as pd
    >>> A = np.random.normal(0, 1, (50, 2))
    >>> B = np.random.normal(1, 1, (50, 2))
    >>> Z = pd.DataFrame(np.vstack([A, B]), columns=['X', 'Y'])
    >>> null_dist = permutation_test(Z, n_permutations=999, n_A=50, random_state=42)
    >>> len(null_dist)
    999
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
        if metric_func is None:
            dist = multivariate_energy_distance(
                Z_arr[A_idx],
                Z_arr[B_idx],
                unbiased=unbiased,
                chunk_threshold=chunk_threshold,
            )
        else:
            dist = metric_func(Z_arr[A_idx], Z_arr[B_idx])
        out.append(dist)
    return out


def calculate_p_value(
    null_stats: Sequence[float],
    observed_stat: float,
    *,
    smooth: bool = True,
) -> float:
    """Compute one-tailed permutation p-value from null distribution.

    Calculates the probability of observing a test statistic greater than or equal
    to the observed value under the null hypothesis (right-tailed test). Applies
    smoothing by default to avoid zero p-values.

    Parameters
    ----------
    null_stats : sequence of float
        Test statistics sampled under the null hypothesis.
    observed_stat : float
        Observed test statistic to compare against null distribution.
    smooth : bool, default=True
        Whether to apply +1 smoothing to prevent zero p-values.

    Returns
    -------
    float
        One-tailed p-value in [0, 1]. Tests H0: groups are identical vs
        H1: observed statistic indicates greater separation than expected by chance.

    Raises
    ------
    ValueError
        If null_stats is empty or contains non-finite values.

    Notes
    -----
    For two-tailed tests, multiply result by 2 or use absolute values of statistics.

    Examples
    --------
    >>> null_dist = [0.5, 0.8, 0.3, 0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.85]
    >>> observed = 0.95
    >>> p_value = calculate_p_value(null_dist, observed)
    >>> p_value  # (0 + 1) / (10 + 1)
    0.09090909090909091
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
