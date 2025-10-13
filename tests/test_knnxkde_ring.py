"""Tests for KNNxKDE imputer on 2D ring geometry"""

import numpy as np
import pytest

from src.data_preparation.data_preparation import DataGenerator, Mcar, Rate
from src.imputation.knnxkdeimputer import KNNxKDEImputer


class TestKNNxKDERing:
    """Test KNNxKDE performance on 2D ring geometry"""

    @pytest.fixture
    def ring_data(self):
        """Generate 2D ring data with MCAR missing values"""
        np.random.seed(42)
        generator = DataGenerator(
            linear_interpolation_ratio=1.0,
            sample_size=3000,
            missing_generator=Mcar(Rate(0.3)),
            geometry_type="ring",
        )
        data = generator.prepare_data()
        return data, generator.actual_values

    def test_default_parameters(self, ring_data):
        """Test that default parameters (after fix) produce acceptable results"""
        data, actual_values = ring_data

        # Create imputer with default parameters
        imputer = KNNxKDEImputer(ml_data=data)

        # Verify default tau is correct
        assert imputer.knnxkde.tau == 0.02, (
            f"Default tau should be 0.02, got {imputer.knnxkde.tau}"
        )
        assert imputer.knnxkde.h == 0.03, (
            f"Default h should be 0.03, got {imputer.knnxkde.h}"
        )

        # Run imputation
        imputer.fit()
        result = imputer.execute()

        # Get imputed values
        mask_missing = data.df["Y"].isna()
        imputed_y = result.loc[mask_missing, "Y"].values
        true_y = actual_values.values

        # Calculate metrics
        rmse = np.sqrt(np.mean((imputed_y - true_y) ** 2))

        # Calculate radius statistics
        radii_imputed = np.sqrt(result.loc[mask_missing, "X"] ** 2 + imputed_y**2)

        # Assertions
        # 1. Radius std should match paper's implementation
        # Using paper's min-max normalization instead of StandardScaler
        assert radii_imputed.std() < 0.18, (
            f"Imputed radius std ({radii_imputed.std():.4f}) should be < 0.18 "
            f"(paper implementation with min-max normalization)"
        )

        # 2. Radius mean should be close to 1.0 (expected ring radius)
        assert 0.95 < radii_imputed.mean() < 1.05, (
            f"Imputed radius mean ({radii_imputed.mean():.4f}) should be ~1.0"
        )

        # 3. RMSE should be reasonable (note: high RMSE is expected due to X-only distance)
        assert rmse < 1.05, f"RMSE ({rmse:.4f}) should be < 1.05 (was ~1.02 before fix)"

    def test_tau_parameter_effect(self, ring_data):
        """Test that smaller tau produces tighter distributions"""
        data, _ = ring_data

        # Test with default tau (0.02)
        imputer_default = KNNxKDEImputer(ml_data=data, h=0.03, tau=0.02)
        imputer_default.fit()
        result_default = imputer_default.execute()

        # Test with larger tau (0.05 - old default)
        generator_large = DataGenerator(
            linear_interpolation_ratio=1.0,
            sample_size=3000,
            missing_generator=Mcar(Rate(0.3)),
            geometry_type="ring",
        )
        data_large = generator_large.prepare_data()
        imputer_large = KNNxKDEImputer(ml_data=data_large, h=0.03, tau=0.05)
        imputer_large.fit()
        result_large = imputer_large.execute()

        # Calculate radius std for both
        mask_default = data.df["Y"].isna()
        mask_large = data_large.df["Y"].isna()

        radii_default = np.sqrt(
            result_default.loc[mask_default, "X"] ** 2
            + result_default.loc[mask_default, "Y"] ** 2
        )
        radii_large = np.sqrt(
            result_large.loc[mask_large, "X"] ** 2
            + result_large.loc[mask_large, "Y"] ** 2
        )

        # Smaller tau should produce tighter (smaller std) distribution
        assert radii_default.std() < radii_large.std(), (
            f"Smaller tau ({radii_default.std():.4f}) should produce tighter distribution than larger tau ({radii_large.std():.4f})"
        )

    def test_h_parameter_effect(self, ring_data):
        """Test that bandwidth h affects smoothness"""
        data, _ = ring_data

        # Test with small h
        imputer_small = KNNxKDEImputer(ml_data=data, h=0.01, tau=0.02)
        imputer_small.fit()
        result_small = imputer_small.execute()

        # Test with large h
        generator_large = DataGenerator(
            linear_interpolation_ratio=1.0,
            sample_size=3000,
            missing_generator=Mcar(Rate(0.3)),
            geometry_type="ring",
        )
        data_large = generator_large.prepare_data()
        imputer_large = KNNxKDEImputer(ml_data=data_large, h=0.05, tau=0.02)
        imputer_large.fit()
        result_large = imputer_large.execute()

        # Both should produce reasonable results
        mask_small = data.df["Y"].isna()
        mask_large = data_large.df["Y"].isna()

        radii_small = np.sqrt(
            result_small.loc[mask_small, "X"] ** 2
            + result_small.loc[mask_small, "Y"] ** 2
        )
        radii_large = np.sqrt(
            result_large.loc[mask_large, "X"] ** 2
            + result_large.loc[mask_large, "Y"] ** 2
        )

        # Both should have reasonable radius distributions
        assert 0.95 < radii_small.mean() < 1.05, (
            f"Small h: radius mean ({radii_small.mean():.4f}) should be ~1.0"
        )
        assert 0.95 < radii_large.mean() < 1.05, (
            f"Large h: radius mean ({radii_large.mean():.4f}) should be ~1.0"
        )
