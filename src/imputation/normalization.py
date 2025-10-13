"""
Normalization utilities for KNNxKDE imputation.
Implementation from the paper: https://arxiv.org/abs/2509.08366
"""

import numpy as np


def normalization(data, parameters=None):
    """Normalize data in the range [0, 1].

    Args:
        data: original data, shape (n, d)
        parameters: if None, default is min/max normalization

    Returns:
        norm_data: normalized data in [0, 1]
        norm_parameters: min_val and max_val used for each column, shape (n, d)
    """
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        # MinMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

        # Return norm_parameters for renormalization
        norm_parameters = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] back to the original range.

    Args:
        norm_data: normalized data, shape (n, d)
        norm_parameters: min_val and max_val used for each column

    Returns:
        renorm_data: renormalized data in the original range, shape (n, d)
    """
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data
