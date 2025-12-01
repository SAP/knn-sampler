from typing import Literal

import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances

type KNNxKdeMetric = Literal["nan_eucl"] | Literal["nan_std_eucl"]


def select_receivers(norm_miss_data, current_miss_pattern):
    """Select observations matching the specified missing pattern.

    Parameters
    ----------
    norm_miss_data : np.ndarray
        Normalized missing data of shape (n, d).
    current_miss_pattern : np.ndarray
        Current missing pattern, boolean array of shape (d,).

    Returns
    -------
    np.ndarray
        Array of indices corresponding to rows matching the missing pattern.
    """
    (n, d) = norm_miss_data.shape
    final_filter = np.ones(n).astype("bool")
    for i in range(d):
        cur_filter = np.isnan(norm_miss_data[:, i]) == current_miss_pattern[i]
        final_filter = np.logical_and(final_filter, cur_filter)
    id_receivers = np.where(final_filter)[0]
    return id_receivers


def select_givers(norm_miss_data, current_miss_pattern):
    """Select observations with complete entries for the missing pattern.

    Parameters
    ----------
    norm_miss_data : np.ndarray
        Normalized missing data of shape (n, d).
    current_miss_pattern : np.ndarray
        Current missing pattern, boolean array of shape (d,).

    Returns
    -------
    np.ndarray
        Array of indices corresponding to potential donor observations for kNNxKDE.
    """
    (n, d) = norm_miss_data.shape
    final_filter = np.ones(n).astype("bool")
    for i in range(d):
        if current_miss_pattern[i]:
            cur_filter = np.isnan(norm_miss_data[:, i]) != current_miss_pattern[i]
            final_filter = np.logical_and(final_filter, cur_filter)
    id_givers = np.where(final_filter)[0]
    return id_givers


def nan_std_euclidean_distances(data_receivers, data_givers, sigmas):
    """Compute NaN-aware standardized Euclidean distance between receiver and donor data.

    Uses feature standard deviations to adjust distances when observations contain
    missing values, ensuring consistent distance computation across incomplete patterns.

    Parameters
    ----------
    data_receivers : np.ndarray
        Normalized data for receiver observations of shape (n1, d).
    data_givers : np.ndarray
        Normalized data for donor observations of shape (n2, d).
    sigmas : np.ndarray
        Feature standard deviations of shape (d,).

    Returns
    -------
    np.ndarray
        NaN-aware standardized Euclidean distances of shape (n1, n2).
    """
    X = np.copy(data_receivers)
    Y = np.copy(data_givers)
    missing_X = np.isnan(X)
    missing_Y = np.isnan(Y)
    X[missing_X] = 0  # replace NaN with zeros
    Y[missing_Y] = 0
    dist = euclidean_distances(X, Y, squared=True)
    XX = X * X
    YY = Y * Y
    minus1 = np.dot(XX, missing_Y.T)
    minus2 = np.dot(missing_X, YY.T)
    dist = dist - minus1 - minus2  # adjust for missing values
    plus1 = np.dot(missing_X, np.tile(sigmas**2, (Y.shape[0], 1)).T)
    plus2 = np.dot(np.tile(sigmas**2, (X.shape[0], 1)), missing_Y.T)
    minus3 = np.dot(np.dot(missing_X, np.diag(sigmas**2)), missing_Y.T)
    dist = dist + plus1 + plus2 - minus3  # add the extra variances
    return np.sqrt(dist)


class KNNxKDE:
    """Hybrid k-nearest neighbors and kernel density estimation imputer.

    Combines distance-based neighbor selection with kernel density estimation
    to generate realistic imputed values for missing data patterns.

    Parameters
    ----------
    h : float, default=0.03
        Kernel bandwidth parameter for density estimation.
    tau : float, default=0.02
        Temperature parameter for distance-based neighbor weighting.
    metric : {'nan_eucl', 'nan_std_eucl'}, default='nan_std_eucl'
        Distance metric for neighbor selection:
        - 'nan_eucl': NaN-aware Euclidean distance
        - 'nan_std_eucl': NaN-aware standardized Euclidean distance

    Raises
    ------
    AttributeError
        If metric is not one of the supported options.
    """

    def __init__(
        self,
        h=0.03,
        tau=1.0 / 50.0,
        metric: KNNxKdeMetric = "nan_std_eucl",
    ):
        self.h = h
        self.tau = tau
        if metric in ["nan_eucl", "nan_std_eucl"]:
            self.metric: KNNxKdeMetric = metric
        else:
            raise AttributeError("Metric should be 'nan_eucl' or 'nan_std_eucl'")

    def impute_samples(self, miss_data, nb_draws=1000):
        """Generate multiple imputed samples using KNNxKDE methodology.

        For each missing value pattern, identifies donor observations, computes
        distance-weighted probabilities, and generates samples via kernel density
        estimation over selected neighbors.

        Parameters
        ----------
        miss_data : np.ndarray
            Data matrix with missing values of shape (n, d).
        nb_draws : int, default=1000
            Number of samples to generate for each missing cell.

        Returns
        -------
        dict or None
            Dictionary mapping (row_idx, col_idx) tuples to arrays of imputed samples.
            Returns None if imputation cannot be performed (no suitable donors).
        """
        (_n, d) = miss_data.shape
        sigmas = np.nanstd(miss_data, axis=0)
        all_miss_patterns = np.unique(np.isnan(miss_data), axis=0)
        imputed_samples = {}

        for _n, current_miss_pattern in enumerate(all_miss_patterns):
            if not np.logical_or.reduce(
                current_miss_pattern
            ):  # if there is no missing value
                continue  # do nothing
            if np.logical_and.reduce(
                current_miss_pattern
            ):  # if there are only missing values
                continue  # do nothing

            id_receivers = select_receivers(miss_data, current_miss_pattern)
            id_givers = select_givers(miss_data, current_miss_pattern)
            if len(id_givers) == 0:
                continue  # skip this pattern, continue with others

            data_receivers = miss_data[id_receivers]
            data_givers = miss_data[id_givers]

            if self.metric == "nan_std_eucl":
                d_ij = nan_std_euclidean_distances(data_receivers, data_givers, sigmas)
            elif self.metric == "nan_eucl":
                d_ij = nan_euclidean_distances(data_receivers, data_givers)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            d_ij[np.isnan(d_ij)] = np.inf
            p_ij = softmax(-d_ij / self.tau, axis=1)

            for i1 in range(len(id_receivers)):
                probs = p_ij[i1]
                neighbors = np.random.choice(
                    len(id_givers), p=probs, size=nb_draws
                )  # Corresponding shuffled id
                current_sample = data_givers[neighbors] + np.random.normal(
                    loc=0.0, scale=self.h, size=(nb_draws, d)
                )
                for i2 in range(d):
                    if current_miss_pattern[i2]:
                        imputed_samples[(id_receivers[i1], i2)] = current_sample[:, i2]

        return imputed_samples

    def impute_mean(self, miss_data, nb_draws=1000):
        """Generate mean-based imputations using KNNxKDE methodology.

        Similar to impute_samples but returns the mean of generated samples
        rather than individual samples, providing point estimates.

        Parameters
        ----------
        miss_data : np.ndarray
            Data matrix with missing values of shape (n, d).
        nb_draws : int, default=1000
            Number of samples to generate for computing means.

        Returns
        -------
        np.ndarray or None
            Copy of input data with missing values replaced by mean estimates.
            Returns None if imputation cannot be performed.
        """
        (_n, d) = miss_data.shape
        sigmas = np.nanstd(miss_data, axis=0)
        all_miss_patterns = np.unique(np.isnan(miss_data), axis=0)
        imputed_data = np.copy(miss_data)

        for _n, current_miss_pattern in enumerate(all_miss_patterns):
            if not np.logical_or.reduce(
                current_miss_pattern
            ):  # if there is no missing value
                continue  # do nothing
            if np.logical_and.reduce(
                current_miss_pattern
            ):  # if there are only missing values
                continue  # do nothing

            id_receivers = select_receivers(miss_data, current_miss_pattern)
            id_givers = select_givers(miss_data, current_miss_pattern)
            if len(id_givers) == 0:
                continue  # skip this pattern, continue with others

            data_receivers = miss_data[id_receivers]
            data_givers = miss_data[id_givers]

            if self.metric == "nan_std_eucl":
                d_ij = nan_std_euclidean_distances(data_receivers, data_givers, sigmas)
            elif self.metric == "nan_eucl":
                d_ij = nan_euclidean_distances(data_receivers, data_givers)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            d_ij[np.isnan(d_ij)] = np.inf
            p_ij = softmax(-d_ij / self.tau, axis=1)

            for i1 in range(len(id_receivers)):
                probs = p_ij[i1]
                neighbors = np.random.choice(
                    len(id_givers), p=probs, size=nb_draws
                )  # Corresponding shuffled id
                current_sample = data_givers[neighbors] + np.random.normal(
                    loc=0.0, scale=self.h, size=(nb_draws, d)
                )
                for i2 in range(d):
                    if current_miss_pattern[i2]:
                        imputed_data[(id_receivers[i1], i2)] = np.mean(
                            current_sample[:, i2]
                        )

        return imputed_data

    def local_distribution(self, miss_data):
        """Compute local probability distributions for missing value patterns.

        Analyzes neighborhood structure and distance-based weights to characterize
        the local distribution around each missing value pattern.

        Parameters
        ----------
        miss_data : np.ndarray
            Data matrix with missing values of shape (n, d).

        Returns
        -------
        dict
            Dictionary storing local distribution parameters (weights, values)
            for each missing pattern.
        """
        (_n, d) = miss_data.shape
        sigmas = np.nanstd(miss_data, axis=0)
        all_miss_patterns = np.unique(np.isnan(miss_data), axis=0)
        cells_distrib = {}  # store (weights, values)

        for _n, current_miss_pattern in enumerate(all_miss_patterns):
            if not np.logical_or.reduce(
                current_miss_pattern
            ):  # if there is no missing value
                continue  # do nothing
            if np.logical_and.reduce(
                current_miss_pattern
            ):  # if there are only missing values
                continue  # do nothing

            id_receivers = select_receivers(miss_data, current_miss_pattern)
            id_givers = select_givers(miss_data, current_miss_pattern)
            if len(id_givers) == 0:
                continue  # skip this pattern, continue with others

            data_receivers = miss_data[id_receivers]
            data_givers = miss_data[id_givers]

            if self.metric == "nan_std_eucl":
                d_ij = nan_std_euclidean_distances(data_receivers, data_givers, sigmas)
            elif self.metric == "nan_eucl":
                d_ij = nan_euclidean_distances(data_receivers, data_givers)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            d_ij[np.isnan(d_ij)] = np.inf
            p_ij = softmax(-d_ij / self.tau, axis=1)

            for i1 in range(len(id_receivers)):
                for i2 in range(d):
                    if current_miss_pattern[i2]:
                        cells_distrib[(id_receivers[i1], i2)] = (
                            p_ij[i1],
                            data_givers[:, i2],
                        )

        return cells_distrib
