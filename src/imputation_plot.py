import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.spatial.distance import euclidean

from src.data_preparation.data_preparation import Rate
from src.imputation_context import ImputationContext


class ImputationPlot:
    def __init__(
        self,
        imputer_name: str,
        plot_context: ImputationContext,
        sample_size: int | None,
        missing_values: int | Rate,
    ) -> None:
        plt.rcParams.update({"figure.max_open_warning": 0})
        self.plot_context = plot_context
        self.descriptor = self.plot_context.dataset_descriptor
        self.figures: list[Figure] = []  # Store all created figures
        self.sample_size = sample_size or "No specified size"
        self.missing_values = missing_values
        self.imputer_name = imputer_name

    def _create_figure(self, figsize: tuple[int, int] = (10, 10)):
        """Helper method to create and store a new figure"""
        fig, ax = plt.subplots(figsize=figsize)
        self.figures.append(fig)
        return fig, ax

    def plot_original_scatterplot(self, figzise: tuple[int, int] = (10, 10)):
        fig, ax = self._create_figure(figsize=figzise)
        sns.scatterplot(
            x=self.descriptor.input_column,
            y=self.descriptor.target_column,
            data=self.plot_context.context_data,
            ax=ax,
        )
        ax.set_title("Original Data")
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_original(self, figsize: tuple[int, int] = (10, 10)):
        """Plot the original data: X vs Y (e.g., Irradiation vs Produced Energy)"""
        fig, ax = self._create_figure(figsize=figsize)
        ax.scatter(
            self.plot_context.context_data[self.descriptor.input_column],
            self.plot_context.context_data[self.descriptor.target_column],
            color="blue",
            label="Original Data",
        )
        ax.set_title(
            f"Original: {self.descriptor.input_column} vs {self.descriptor.target_column}"
        )
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_imputation_results(
        self,
        figsize: tuple[int, int] = (10, 10),
    ):
        """Plot the imputed data: X vs Y (e.g., Irradiation vs Produced Energy)"""
        fig, ax = self._create_figure(figsize=figsize)
        ax.scatter(
            self.plot_context.predicted_data[self.descriptor.target_column],
            self.plot_context.predicted_data[self.descriptor.input_column],
            color="red",
            label="Imputed Data",
        )
        ax.set_title(
            f"{self.imputer_name}: {self.descriptor.input_column} vs {self.descriptor.target_column}"
        )
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_uncertainty_results(
        self,
        lower_percentile: float,
        upper_percentile: float,
        lower_bounds: list[float],
        upper_bounds: list[float],
        figsize: tuple[int, int] = (12, 12),
    ):
        fig, ax = self._create_figure(figsize=figsize)

        coverage_probability = np.mean(
            (
                self.plot_context.actual_data[self.descriptor.target_column]
                >= lower_bounds
            )
            & (
                self.plot_context.actual_data[self.descriptor.target_column]
                <= upper_bounds
            )
        )

        x = np.array(self.plot_context.actual_data[self.descriptor.input_column])
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        lower_sorted = lower_bounds[sort_idx]
        upper_sorted = upper_bounds[sort_idx]

        ax.scatter(
            self.plot_context.actual_data[self.descriptor.input_column],
            self.plot_context.actual_data[self.descriptor.target_column],
            color="green",
            label="True Data",
        )
        ax.scatter(
            self.plot_context.predicted_data[self.descriptor.input_column],
            self.plot_context.predicted_data[self.descriptor.target_column],
            color="red",
            label="Imputed Data",
        )
        ax.fill_between(
            x_sorted,
            lower_sorted,
            upper_sorted,
            color="gray",
            alpha=0.3,
            label="Confidence Interval",
        )

        ax.set_title(
            f"""
            {self.imputer_name}:
            Confidence interval for {lower_percentile}% and {upper_percentile}% percentiles with sample size: {self.sample_size} and missing_values: {self.missing_values}
            """
        )

        ax.text(
            0.55,
            0.95,
            f"Coverage probability: {coverage_probability:.2f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_original_imputed(self, figsize: tuple[int, int] = (10, 10)):
        fig, ax = self._create_figure(figsize=figsize)

        sns.scatterplot(
            x=self.descriptor.input_column,
            y=self.descriptor.target_column,
            data=self.plot_context.context_data,
            color="blue",
            label="Original Data",
            ax=ax,
        )
        sns.scatterplot(
            x=self.descriptor.input_column,
            y=self.descriptor.target_column,
            data=self.plot_context.actual_data,
            color="green",
            label="True Data",
            marker="o",
            alpha=0.5,
            ax=ax,
        )
        sns.scatterplot(
            x=self.descriptor.input_column,
            y=self.descriptor.target_column,
            data=self.plot_context.predicted_data,
            color="red",
            label="Imputed Data",
            marker="o",
            alpha=0.5,
            ax=ax,
        )

        ax.set_title(
            f"{self.imputer_name}: Comparison of Original, Imputed, and True Data"
        )
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_imputed_accuracy(self, figsize: tuple[int, int] = (10, 10)):
        distances = []
        for pred, actual in zip(
            self.plot_context.predicted_data[
                [self.descriptor.input_column, self.descriptor.target_column]
            ].values,
            self.plot_context.actual_data[
                [self.descriptor.input_column, self.descriptor.target_column]
            ].values,
            strict=False,
        ):
            distances.append(euclidean(pred, actual))
        cmap = LinearSegmentedColormap.from_list(
            "custom", ["green", "orange", "red"], N=256
        )
        fig, ax = self._create_figure(figsize=figsize)
        scatter = ax.scatter(
            self.plot_context.predicted_data[self.descriptor.input_column],
            self.plot_context.predicted_data[self.descriptor.target_column],
            c=distances,
            cmap=cmap,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax, label="Distance from true value")
        ax.set_title(f"{self.imputer_name}: Imputed Data Accuracy")
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(self.descriptor.target_column)
        ax.legend()

    def plot_local_distance(self, n=100, figsize: tuple[int, int] = (10, 10)):
        fig, ax = self._create_figure(figsize=figsize)

        x = self.plot_context.predicted_data[self.descriptor.input_column]
        y_pred = self.plot_context.predicted_data[self.descriptor.target_column]
        y_actual = self.plot_context.actual_data[self.descriptor.target_column]

        local_distances = []
        for i, _ in enumerate(x):
            start = max(0, i - n)
            end = min(len(x), i + n + 1)

            dists = []
            for j in range(start, end):
                if j < len(y_pred) and j < len(y_actual):
                    dists.append(
                        np.abs(y_pred[y_pred.index[j]] - y_actual[y_actual.index[j]])
                    )
            if dists:
                local_distances.append(np.mean(dists))
            else:
                local_distances.append(0)

        ax.plot(x, local_distances)
        ax.set_title(f"{self.imputer_name}: Local Distance from True Values")
        ax.set_xlabel(self.descriptor.input_column)
        ax.set_ylabel(f"Average of {n} closest neighbor's distances for true value")
        ax.legend()

    def show(self, block=True):
        for fig in self.figures:
            fig.show()
        if block:
            plt.show(block=True)

    def close_all(self):
        """Close all figures created by this instance"""
        for fig in self.figures:
            plt.close(fig)
        self.figures.clear()
