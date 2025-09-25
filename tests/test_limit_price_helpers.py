import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from limit_price_helpers import (
    prepare_and_split_data,
    run_classification_pipeline,
    run_regression_pipeline,
)


@pytest.fixture
def mock_data():
    """Creates a mock DataFrame for testing data preparation."""
    np.random.seed(9)
    n_samples = 100
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "sign_spread": np.random.choice([0, 1], n_samples),
        "spread_eur": np.random.uniform(-10, 10, n_samples),
        "spot_eur": np.random.uniform(50, 150, n_samples),
        "imb_eur": np.random.uniform(-5, 5, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_hyperopt_and_logging():
    """Mocks external dependencies to ensure fast, deterministic tests."""
    with patch("limit_price_helpers.fmin", autospec=True) as mock_fmin, patch(
        "limit_price_helpers.space_eval", autospec=True
    ) as mock_space_eval, patch(
        "limit_price_helpers.cross_val_score", autospec=True
    ) as mock_cross_val_score, patch(
        "limit_price_helpers.logger", autospec=True
    ):

        # Configure mocks to return predictable results from the optimization process.
        mock_fmin.return_value = {"regressor": 0, "lr": {"alpha": 1.0}}
        mock_space_eval.return_value = {"regressor": "lr", "lr": {"alpha": 1.0}}
        mock_cross_val_score.return_value = [0.9]
        yield


@patch("limit_price_helpers.limit_price_data_prep")
def test_prepare_and_split_data_full_integration(mock_data_prep, mock_data):
    """
    Tests the complete data preparation and splitting process, including
    data integrity, scaling, and parameter handling.
    """
    # Use a copy of the mock data to prevent in-place modification side effects
    mock_data_prep.return_value = mock_data.copy()
    data_dir = Path("/fake/path")

    # 1. Test basic functionality with default parameters
    results = prepare_and_split_data(data_dir)
    assert len(results) == 7
    (
        X_train_scaled,
        X_test_scaled,
        y1_train,
        y1_test,
        y2_train_scaled,
        y2_test,
        y2_scaler,
    ) = results

    # Check data characteristics
    assert X_train_scaled.shape[1] == X_test_scaled.shape[1] == 3
    assert len(y1_train) + len(y1_test) == len(mock_data)

    # Check scaling properties
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)
    assert isinstance(y2_scaler, MinMaxScaler)

    # Check that y2_test has the same indices as y1_test
    pd.testing.assert_index_equal(y1_test.index, y2_test.index)

    # 2. Test with `final_test=True`
    # Use a fresh copy of the mock data for this second test call
    mock_data_prep.return_value = mock_data.copy()
    results_final_test = prepare_and_split_data(data_dir, final_test=True)
    X_train_final, _, y1_train_final, _, _, y2_train_raw, _ = results_final_test
    assert len(y2_train_raw) == len(y1_train_final)

    # 3. Test reproducibility with `shuffle=False`
    mock_data_prep.return_value = mock_data.copy()
    results1 = prepare_and_split_data(data_dir, shuffle=False)
    mock_data_prep.return_value = mock_data.copy()
    results2 = prepare_and_split_data(data_dir, shuffle=False)
    np.testing.assert_array_equal(results1[0], results2[0])

    # 4. Verify mock function was called correctly
    assert mock_data_prep.call_count == 4
    mock_data_prep.assert_called_with(data_dir=data_dir)


def test_run_regression_pipeline():
    """
    Verifies the end-to-end functionality of the regression pipeline.
    """
    # 1. Create a minimal dataset with enough samples for KNeighborsRegressor
    X_train_scaled = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]
    )
    y2_train_scaled = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    X_test_scaled = np.array([[21.0, 22.0], [23.0, 24.0]])
    y2_test = pd.Series([10.0, 12.0])

    y_scaler = MinMaxScaler()
    y_scaler.fit(y2_train_scaled.reshape(-1, 1))

    # 2. Execute the pipeline.
    # The function returns the ensemble model and the list of members.
    ensemble_model, ensemble_members = run_regression_pipeline(
        X_train_scaled, X_test_scaled, y2_train_scaled, y2_test, y_scaler
    )

    # 3. Assert the expected outcomes.
    assert isinstance(ensemble_model, VotingRegressor)
    assert ensemble_members == "ols"
    assert len(ensemble_model.estimators) == 1

    # Check that the model can make predictions and that the predictions have the correct shape.
    predictions_scaled = ensemble_model.predict(X_test_scaled)
    assert predictions_scaled.shape[0] == X_test_scaled.shape[0]
