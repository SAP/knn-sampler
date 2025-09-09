import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.stats_utils import multivariate_energy_distance_imputed


class EnergyDistanceCalculator:
    def __init__(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_indexes: np.ndarray,
    ):
        self.original_df: pd.DataFrame = original_df
        self.imputed_df: pd.DataFrame = imputed_df
        self.missing_indexes: np.ndarray = missing_indexes
        self.actual_values: pd.DataFrame = original_df.loc[missing_indexes, ["X", "Y"]]
        self.imputed_values: pd.DataFrame = imputed_df.loc[missing_indexes, ["X", "Y"]]
        self.z: pd.DataFrame = pd.concat(
            [self.imputed_values, self.actual_values], ignore_index=True
        )

    def plot_original_imputed(self, actual_values: pd.Series):
        if isinstance(actual_values, pd.Series):
            raise ValueError("actual_values should be a Series")
        plt.figure(figsize=(10, 6))

        non_missing_data = self.original_df.drop(index=self.missing_indexes)
        sns.scatterplot(
            x="X", y="Y", data=non_missing_data, color="blue", label="Original Data"
        )

        imputed_y_values = self.imputed_df.loc[self.missing_indexes, "Y"]
        sns.scatterplot(
            x=self.imputed_df.loc[self.missing_indexes, "X"],
            y=imputed_y_values,
            color="red",
            label="Imputed Data",
            marker="o",
        )

        sns.scatterplot(
            x=self.original_df.loc[self.missing_indexes, "X"],
            y=actual_values,
            color="green",
            label="True Data",
            marker="o",
        )

        plt.title("Comparison of Original, Imputed, and True Data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def permutation_test(self, n_permutations):
        energy_distances = []
        for _ in range(n_permutations):
            z_shuffled = self.z.sample(frac=1).reset_index(drop=True)
            mid_point = len(z_shuffled) // 2
            z_a = z_shuffled.iloc[:mid_point]
            z_b = z_shuffled.iloc[mid_point:]
            energy_distance_permuted = multivariate_energy_distance_imputed(
                z_a.to_numpy(), z_b.to_numpy()
            )
            energy_distances.append(energy_distance_permuted)
        return energy_distances

    def permutation_plot(
        self, energy_distances, method_name, p_value, sample_size, n_permutations
    ):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            energy_distances, bins=10, kde=True, color="blue", edgecolor="black"
        )
        plt.xlabel("Energy Distance")
        plt.ylabel("Frequency")
        plt.title(
            f"Energy Distance between permuted datasets ({method_name}) - sample size:{sample_size} - number of permutations: {n_permutations}"
        )
        plt.text(
            0.95,
            0.95,
            f"P-value: {p_value:.4f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )
        plt.show()
