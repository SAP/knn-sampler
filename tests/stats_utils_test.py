import unittest

import numpy as np

from src.stats_utils import (
    calculate_p_value,
    multivariate_energy_distance,
    permutation_test,
)


class TestStatsUtils(unittest.TestCase):
    """Unit tests for stats utilities following repository test conventions.

    Tests cover symmetry, chunked vs full computation, unbiased vs biased
    behavior, input validation, permutation reproducibility and p-value logic.
    """

    def test_energy_distance_zero_identical(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(20, 3))
        d = multivariate_energy_distance(A, A, unbiased=False)
        self.assertAlmostEqual(d, 0.0, places=12)

    def test_energy_distance_symmetry(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(15, 4))
        B = rng.normal(loc=1.0, size=(12, 4))
        d_ab = multivariate_energy_distance(A, B)
        d_ba = multivariate_energy_distance(B, A)
        self.assertAlmostEqual(d_ab, d_ba, places=12)

    def test_energy_distance_chunked_vs_full(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(10, 5))
        B = rng.normal(size=(11, 5))
        d_full = multivariate_energy_distance(A, B, chunk_threshold=10**12)
        d_chunk = multivariate_energy_distance(A, B, chunk_threshold=1)
        # use a relative tolerance similar to previous tests
        self.assertAlmostEqual(d_chunk, d_full, places=9)

    def test_energy_distance_unbiased_vs_biased_relation(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(30, 3))
        B = rng.normal(loc=0.5, size=(25, 3))
        d_unbiased = multivariate_energy_distance(A, B, unbiased=True)
        d_biased = multivariate_energy_distance(A, B, unbiased=False)
        # Biased intra mean includes diagonal zeros -> energy distance should be >= unbiased (allow tiny numeric slack)
        self.assertGreaterEqual(d_biased + 1e-12, d_unbiased)

    def test_energy_distance_input_validation(self):
        with self.subTest(case="wrong_ndarray_shape"):
            with self.assertRaises(ValueError):
                multivariate_energy_distance(np.array([1, 2, 3]), np.array([[1, 2, 3]]))
        with self.subTest(case="mismatched_feature_dim"):
            with self.assertRaises(ValueError):
                multivariate_energy_distance(np.ones((2, 2)), np.ones((3, 3)))
        with self.subTest(case="too_few_rows"):
            with self.assertRaises(ValueError):
                multivariate_energy_distance(np.ones((1, 2)), np.ones((2, 2)))
        with self.subTest(case="non_finite_values"):
            with self.assertRaises(ValueError):
                multivariate_energy_distance(
                    np.array([[np.nan, 0.0], [1.0, 2.0]]), np.ones((2, 2))
                )

    def test_permutation_test_length_and_reproducibility(self):
        rng = np.random.default_rng(10)
        A = rng.normal(size=(10, 2))
        B = rng.normal(loc=1.0, size=(10, 2))
        Z = np.vstack([A, B])
        out1 = permutation_test(Z, n_permutations=50, n_A=10, random_state=123)
        out2 = permutation_test(Z, n_permutations=50, n_A=10, random_state=123)
        self.assertEqual(len(out1), 50)
        self.assertEqual(out1, out2)

    def test_calculate_p_value_basic(self):
        null = [0.1, 0.2, 0.3, 0.4]
        p = calculate_p_value(null, observed_stat=0.35, smooth=True)
        self.assertAlmostEqual(p, (1 + 1) / (4 + 1))

    def test_calculate_p_value_no_smooth(self):
        null = [0.1, 0.2, 0.3, 0.4]
        p = calculate_p_value(null, observed_stat=0.4, smooth=False)
        self.assertAlmostEqual(p, 1 / 4)

    def test_calculate_p_value_validation(self):
        with self.assertRaises(ValueError):
            calculate_p_value([], 0.1)
        with self.assertRaises(ValueError):
            calculate_p_value([0.1, float("nan")], 0.1)

    def test_energy_distance_identical_unbiased_negative_bias(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(30, 4))
        d_unbiased = multivariate_energy_distance(A, A, unbiased=True)
        # slight negative bias when comparing the same sample with unbiased=True
        self.assertLess(d_unbiased, 0.0)
        pairwise = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
        mask = ~np.eye(A.shape[0], dtype=bool)
        mean_off = pairwise[mask].mean()
        theoretical_mag = 2 * mean_off / A.shape[0]
        # allow a small slack factor on the theoretical bound
        self.assertLessEqual(abs(d_unbiased), theoretical_mag * 1.25)
