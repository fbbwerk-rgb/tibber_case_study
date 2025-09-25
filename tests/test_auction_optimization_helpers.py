import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

from auction_optimization_helpers import (
    load_models,
    load_demand_simulation_data,
    prepare_data_for_optimization,
    optimize_charging_schedule,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_dfs():
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    data = {
        "spot_fc_eur": np.array(
            [12.5, 15.0, 17.5, 12.0, 14.5, 16.0, 18.0, 13.0, 15.5, 11.0]
        ),
        "mean_load": np.array([2.5, 3.0, 1.5, 4.0, 3.5, 2.0, 4.5, 1.0, 2.0, 3.0]),
        "hour_of_day": dates.hour,
        "spot_eur": np.array([8.0, 10.0, 9.5, 7.5, 11.0, 12.0, 10.5, 8.5, 9.0, 7.0]),
        "imb_eur": np.array([0.5, -0.2, 1.0, -0.5, 0.8, 1.2, -0.8, 0.3, -0.1, 0.6]),
    }
    mock_data_df = pd.DataFrame(data, index=dates)

    X_test = mock_data_df[["spot_fc_eur", "mean_load"]]
    X_test_scaled = pd.DataFrame(
        np.array(
            [
                [0.881596, 0.198583, 0.677570],
                [0.345892, 0.659345, 0.123456],
                [0.912345, 0.234567, 0.876543],
                [0.556789, 0.456789, 0.543210],
                [0.234567, 0.789123, 0.345678],
                [0.876543, 0.123456, 0.987654],
                [0.111222, 0.222333, 0.333444],
                [0.444555, 0.555666, 0.666777],
                [0.777888, 0.888999, 0.000111],
                [0.999888, 0.777666, 0.555444],
            ]
        ),
        index=X_test.index,
        columns=["0", "1", "2"],
    )

    y2_test = pd.Series(
        np.array([5.0, 6.0, 7.0, 8.0, 9.0, 6.5, 7.5, 8.5, 9.5, 5.5]),
        index=dates,
        name="some_target_col",
    )

    return {
        "full_data_df": mock_data_df,
        "X_test_scaled": X_test_scaled,
        "y2_test": y2_test,
    }


@patch("auction_optimization_helpers.pd.read_csv", autospec=True)
def test_load_demand_simulation_data_success(mock_read_csv):
    mock_v_max_df = pd.DataFrame(
        {"mean_load": [10.0, 15.0]}, index=pd.Index([0, 1], name="Hour")
    )

    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = "100.5"

    with patch("builtins.open", return_value=mock_file):
        with patch(
            "auction_optimization_helpers.pd.read_csv", return_value=mock_v_max_df
        ):
            v_max_df, total_demand = load_demand_simulation_data(Path("/fake/dir"))

    assert isinstance(v_max_df, pd.DataFrame)
    assert not v_max_df.empty
    assert v_max_df.index.name == "Hour"
    assert total_demand == 100.5
    mock_file.__enter__.return_value.read.assert_called_once()


@pytest.mark.usefixtures("mock_dfs")
def test_prepare_data_for_optimization_success(mock_dfs):
    clf_mock = MagicMock()
    reg_mock = MagicMock()
    y2_scaler_mock = MagicMock()

    reg_predict_value = np.array(
        [12.5, 15.0, 17.5, 12.0, 14.5, 16.0, 18.0, 13.0, 15.5, 11.0]
    ).reshape(-1, 1)
    y2_scaler_inverse_value = np.array(
        [11.0, 13.5, 16.0, 11.5, 14.0, 15.0, 17.0, 12.0, 14.5, 10.0]
    ).reshape(-1, 1)
    prob_gain_value = np.array([[0.1, 0.9]] * 10)

    clf_mock.predict_proba.return_value = prob_gain_value
    reg_mock.predict.return_value = reg_predict_value
    y2_scaler_mock.inverse_transform.return_value = y2_scaler_inverse_value

    v_max_df = pd.DataFrame(
        {
            "mean_load": np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                ]
            )
        },
        index=pd.Index(range(24), name="Hour"),
    )

    with patch(
        "auction_optimization_helpers.yaml.safe_load", return_value={"TAU": 0.7}
    ):
        prepared_df = prepare_data_for_optimization(
            clf_mock,
            reg_mock,
            mock_dfs["X_test_scaled"],
            mock_dfs["y2_test"],
            y2_scaler_mock,
            mock_dfs["full_data_df"],
            v_max_df,
        )

    assert isinstance(prepared_df, pd.DataFrame)
    assert "expected_cost_eur" in prepared_df.columns
    assert "prob_gain" in prepared_df.columns
    assert "estimated_gain" in prepared_df.columns
    assert "v_max" in prepared_df.columns
    assert not prepared_df.empty

    expected_expected_cost = mock_dfs["full_data_df"]["spot_fc_eur"].iloc[
        :10
    ].values + prob_gain_value[:, 1] * y2_scaler_inverse_value.flatten() * (1 / 0.7)

    assert np.allclose(
        prepared_df["expected_cost_eur"].values, expected_expected_cost, rtol=1e-6
    )


@pytest.mark.usefixtures("mock_dfs")
@patch("auction_optimization_helpers.linprog", autospec=True)
def test_optimize_charging_schedule_valid_input(mock_linprog, mock_dfs):
    date_range = pd.date_range(
        start="2023-01-01 16:00:00", periods=18, freq="h", tz="UTC"
    )
    output_df = pd.DataFrame(index=date_range)
    output_df["expected_cost_eur"] = np.random.uniform(50, 100, 18)
    output_df["v_max"] = np.random.uniform(10, 50, 18)
    total_energy_demand = 100.0

    mock_linprog.return_value = MagicMock(success=True, x=np.random.uniform(0, 10, 18))

    result_df = optimize_charging_schedule(output_df, total_energy_demand)

    assert isinstance(result_df, pd.DataFrame)
    assert "optimal_plan" in result_df.columns
    assert not result_df["optimal_plan"].isnull().any()


@patch("auction_optimization_helpers.linprog", autospec=True)
def test_optimize_charging_schedule_no_solution(mock_linprog):
    date_range = pd.date_range(
        start="2023-01-01 16:00:00", periods=18, freq="h", tz="UTC"
    )
    output_df = pd.DataFrame(index=date_range)
    output_df["expected_cost_eur"] = np.random.uniform(50, 100, 18)
    output_df["v_max"] = np.random.uniform(10, 50, 18)
    total_energy_demand = 100.0

    mock_linprog.return_value = MagicMock(success=False, x=None)

    result_df = optimize_charging_schedule(output_df, total_energy_demand)

    assert isinstance(result_df, pd.DataFrame)
    assert "optimal_plan" in result_df.columns
