import pandas as pd
import pytest

from src.data_preparation.data_description import DataFrameMLData
from src.imputation import KnnImputer, LinearImputer
from src.utils import do_imputation, instantiate_imputers
from tests.testutils import prepare_generated_data


@pytest.fixture(scope="module")
def generated_data() -> DataFrameMLData:
    prepared, _actual_values, _generator = prepare_generated_data()
    return prepared


class TestDoImputation:
    def test_single_iteration_returns_one_timed_result(
        self, generated_data: DataFrameMLData
    ):
        imputer = KnnImputer(ml_data=generated_data)
        results = do_imputation(imputer)
        assert list(results.keys()) == [imputer]
        result, execution_time = results[imputer]
        assert isinstance(result, pd.DataFrame)
        assert execution_time >= 0.0
        assert not result["Y"].isna().any()

    def test_iterations_return_one_result_per_iteration(
        self, generated_data: DataFrameMLData
    ):
        imputer = KnnImputer(ml_data=generated_data)
        iterations = 3
        results = do_imputation(imputer, iterations=iterations)
        assert list(results.keys()) == [imputer]
        assert len(results[imputer]) == iterations
        for result, execution_time in results[imputer]:
            assert isinstance(result, pd.DataFrame)
            assert execution_time >= 0.0


class TestInstantiateImputers:
    def test_instantiates_with_kwargs(self, generated_data: DataFrameMLData):
        imputers = list(
            instantiate_imputers({KnnImputer: {"n_neighbors": 3}}, generated_data)
        )
        assert len(imputers) == 1
        assert isinstance(imputers[0], KnnImputer)
        assert imputers[0].knn_imputer.n_neighbors == 3

    def test_unknown_kwarg_raises_on_iteration(self, generated_data: DataFrameMLData):
        # instantiate_imputers is a generator: the error fires on iteration
        generator = instantiate_imputers(
            {LinearImputer: {"bogus_option": 1}}, generated_data
        )
        with pytest.raises(ValueError, match="bogus_option"):
            list(generator)
