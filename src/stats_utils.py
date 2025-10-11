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

    The multivariate energy distance is a metric for comparing probability
    distributions based on distances between sample points. The population
    energy distance between distributions X and Y is:

        ED(X, Y) = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    where X, X' are independent copies from distribution X, and Y, Y' are
    independent copies from distribution Y.

    This function returns a sample estimate using Euclidean distances. By default,
    it uses U-statistic estimators that exclude diagonal terms (i=j) to reduce
    bias in finite samples.

    Parameters
    ----------
    Z_A : np.ndarray, shape (n, d)
        First sample with n observations and d features.
    Z_B : np.ndarray, shape (m, d)
        Second sample with m observations and d features.
    unbiased : bool, default=True
        Whether to use unbiased U-statistic estimators for intra-sample distances.
        If False, uses simple mean including diagonal zeros (faster but biased
        downward for small samples).
    chunk_threshold : int, default=200_000_000
        Memory management threshold. If n*m exceeds this value, cross-distances
        are computed in chunks to reduce peak memory usage from O(nm) to O(chunk_size).

    Returns
    -------
    float
        Estimated energy distance. Always non-negative; equals 0 if and only if
        both samples come from the same distribution (in population).

    Raises
    ------
    ValueError
        If inputs are not 2D numeric arrays, have incompatible feature dimensions,
        contain fewer than 2 observations each, or contain NaN/Inf values.

    Notes
    -----
    **Computational Complexity:**
    - Time: O(n²d + m²d + nmd) for distance computations
    - Memory: O(min(nm, chunk_threshold)) with chunking enabled

    **Statistical Properties:**
    - The energy distance satisfies the triangle inequality and is zero if and
      only if the distributions are identical
    - U-statistic version (unbiased=True) provides better finite-sample properties
    - For large samples, biased and unbiased versions converge to the same value

    Examples
    --------
    >>> import numpy as np
    >>> # Two samples from different distributions
    >>> A = np.random.normal(0, 1, (100, 2))
    >>> B = np.random.normal(1, 1, (100, 2))
    >>> ed = multivariate_energy_distance(A, B)
    >>> ed > 0  # Should be positive for different distributions
    True

    >>> # Same distribution should give distance near 0
    >>> C = np.random.normal(0, 1, (100, 2))
    >>> D = np.random.normal(0, 1, (100, 2))
    >>> ed_same = multivariate_energy_distance(C, D)
    >>> ed_same < ed  # Should be smaller
    True

    References
    ----------
    .. [1] Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of
           statistics based on distances. Journal of Statistical Planning and
           Inference, 143(8), 1249-1272.
    .. [2] Rizzo, M. L., & Székely, G. J. (2016). Energy distance. Wiley
           Interdisciplinary Reviews: Computational Statistics, 8(1), 27-38.
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

    return float(2.0 * mean_cross_dists - mean_intra_A - mean_intra_B)


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
    """Generate null distribution via label permutation for two-sample testing.

    Implements the permutation test procedure for testing whether two samples
    come from the same distribution. The method randomly reassigns group labels
    to observations and computes a test statistic for each permutation, creating
    a null distribution under the hypothesis of no difference between groups.

    The current implementation uses multivariate energy distance as the test
    statistic, but the permutation framework is general and applicable to other
    two-sample statistics.

    Parameters
    ----------
    Z : DataFrame | np.ndarray, shape (n_total, d)
        Concatenated observations from both groups, typically [Group_A; Group_B].
        If DataFrame, only numeric columns are used.
    n_permutations : int
        Number of random permutations for null distribution generation.
        Common values: 999, 1999, 4999 (providing p-value precision of ~0.001, 0.0005, 0.0002).
    n_A : int | None, default=None
        Size of first group in original partition. If None, uses floor(n_total/2).
        Must be between 2 and n_total-2 to ensure meaningful group comparisons.
    random_state : int | None, default=None
        Seed for reproducible permutation generation. Recommended for research
        and debugging purposes.
    unbiased : bool, default=True
        Passed to energy distance computation. Controls bias correction in
        finite-sample distance estimation.
    chunk_threshold : int, default=200_000_000
        Memory management parameter passed to distance computation for handling
        large datasets efficiently.

    Returns
    -------
    list[float]
        Null distribution of test statistics under permuted group labels.
        Length equals n_permutations. Higher values indicate greater separation
        between groups.

    Raises
    ------
    ValueError
        If n_permutations < 1, inputs invalid, or group sizes inappropriate
        for meaningful comparison.

    Notes
    -----
    **Statistical Foundation:**
    The permutation test relies on exchangeability: under the null hypothesis
    that both groups come from the same distribution, any reassignment of
    observations to groups is equally likely.

    **Recommended Usage:**
    - Use >= 999 permutations for p-values in scientific contexts
    - Set random_state for reproducible results
    - Ensure balanced group sizes when possible for optimal power

    **Computational Complexity:**
    - Time: O(n_permutations x [n²d + m²d]) where n,m are group sizes
    - Memory: O(n_total x d) for data storage plus distance computation overhead

    Examples
    --------
    >>> import pandas as pd
    >>> # Create test data
    >>> A = np.random.normal(0, 1, (50, 2))
    >>> B = np.random.normal(1, 1, (50, 2))
    >>> Z = pd.DataFrame(np.vstack([A, B]), columns=['X', 'Y'])
    >>>
    >>> # Generate null distribution
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
    """Compute permutation p-value from null distribution and observed statistic.

    Calculates the probability of observing a test statistic at least as extreme
    as the observed value, assuming the null hypothesis is true. Optionally applies
    smoothing to avoid zero p-values which can be problematic for multiple testing
    corrections and reporting.

    Parameters
    ----------
    null_stats : sequence of float
        Test statistics sampled under the null hypothesis (e.g., from permutation_test).
        Should contain at least 99 values for meaningful p-value estimation.
    observed_stat : float
        The empirically observed test statistic to compare against the null distribution.
    smooth : bool, default=True
        Whether to apply +1 smoothing to numerator and denominator. This prevents
        zero p-values and provides more conservative inference, as recommended by
        Phipson & Smyth (2010) for permutation tests.

    Returns
    -------
    float
        P-value in [0, 1]. With smoothing: minimum possible value is 1/(N+1) where
        N is len(null_stats). Without smoothing: minimum is 0.

    Raises
    ------
    ValueError
        If null_stats is empty or contains non-finite values (NaN, ±Inf).

    Notes
    -----
    **Statistical Interpretation:**
    - p < 0.05: Strong evidence against null hypothesis (conventional threshold)
    - p < 0.01: Very strong evidence against null hypothesis
    - p ≥ 0.05: Insufficient evidence to reject null hypothesis

    **Smoothing Rationale:**
    The +1 smoothing (smooth=True) is recommended practice because:
    - Prevents p-value of exactly 0, which is theoretically impossible
    - Provides more conservative inference
    - Better behaves under multiple testing corrections (FDR, Bonferroni)
    - Accounts for finite-sample uncertainty in permutation tests

    **Right-tailed Test:**
    Currently implements one-sided test (observed ≥ null). For two-sided tests,
    consider doubling the p-value or using absolute values of statistics.

    Examples
    --------
    >>> # Simulate null distribution and observed statistic
    >>> null_dist = [0.5, 0.8, 0.3, 0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.85]
    >>> observed = 0.95
    >>>
    >>> # Calculate p-value with smoothing
    >>> p_smooth = calculate_p_value(null_dist, observed, smooth=True)
    >>> # p_smooth = (0 + 1) / (10 + 1) = 0.091
    >>>
    >>> # Calculate without smoothing
    >>> p_raw = calculate_p_value(null_dist, observed, smooth=False)
    >>> # p_raw = 0 / 10 = 0.0

    References
    ----------
    .. [1] Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never
           be zero: calculating exact P-values when permutations are randomly drawn.
           Statistical Applications in Genetics and Molecular Biology, 9(1).
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
