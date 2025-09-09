from typing import Any

import numpy as np


def multivariate_energy_distance_imputed(Z_A, Z_B) -> np.floating:
    """Calculate multivariate energy distance"""
    cross_distances = np.linalg.norm(Z_A[:, np.newaxis] - Z_B[np.newaxis, :], axis=-1)
    intra_distances_A = np.linalg.norm(Z_A[:, np.newaxis] - Z_A[np.newaxis, :], axis=-1)
    intra_distances_B = np.linalg.norm(Z_B[:, np.newaxis] - Z_B[np.newaxis, :], axis=-1)
    ED = (
        2 * np.mean(cross_distances)
        - np.mean(intra_distances_A)
        - np.mean(intra_distances_B)
    )
    return ED


def permutation_test(Z, n_permutations) -> list[Any]:
    energy_distances = []
    for _ in range(n_permutations):
        Z_shuffled = Z.sample(frac=1).reset_index(drop=True)
        mid_point = len(Z_shuffled) // 2
        Z_A = Z_shuffled.iloc[:mid_point]
        Z_B = Z_shuffled.iloc[mid_point:]
        energy_distance_permuted = multivariate_energy_distance_imputed(
            Z_A.to_numpy(), Z_B.to_numpy()
        )
        energy_distances.append(energy_distance_permuted)
    return energy_distances


def calculate_p_value(energy_distances, original_energy_distance) -> float:
    count_extreme = sum(ed >= original_energy_distance for ed in energy_distances)
    p_value = count_extreme / len(energy_distances)
    return p_value
