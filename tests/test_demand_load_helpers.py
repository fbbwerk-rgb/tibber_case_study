import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import logging

from src.demand_load_helpers import (
    calculate_hourly_load,
    run_monte_carlo_simulation,
    save_simulation_results,
    plot_and_summarize_results_with_path,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_ev_data():
    """
    A fixture to provide a sample DataFrame of EV scenarios with
    predictable times and loads.
    """
    data = {
        "to_charge_kwh": [10.0, 20.0, 5.0, 15.0],
        "arrival_time": ["17:30", "19:00", "23:00", "03:00"],
        "departure_time": ["06:00", "22:00", "08:00", "09:00"],
    }
    df = pd.DataFrame(data)

    # The logic inside calculate_hourly_load handles the datetime conversion
    return df


def test_calculate_hourly_load_correctly_sums_loads(sample_ev_data):
    """
    Tests that calculate_hourly_load correctly sums the charging load for each hour
    based on the cars' arrival and departure times.
    """
    # The expected loads are based on the sample data.
    expected_loads = np.zeros(24)
    expected_loads[18] = 10
    expected_loads[19] = 30
    expected_loads[20] = 30
    expected_loads[21] = 30
    expected_loads[22] = 10
    expected_loads[23] = 15
    expected_loads[0:6] = 15
    expected_loads[6:8] = 5

    calculated_loads = calculate_hourly_load(sample_ev_data.copy())

    np.testing.assert_array_equal(calculated_loads, expected_loads)


@patch("src.demand_load_helpers.calculate_hourly_load", autospec=True)
@patch("src.demand_load_helpers.generate_simulation_data", autospec=True)
def test_run_monte_carlo_simulation_returns_correct_dataframes(
    mock_generate_simulation_data, mock_calculate_hourly_load
):
    """
    Tests that run_monte_carlo_simulation correctly orchestrates the simulation
    and returns DataFrames with the correct structure and values.
    """
    num_simulations = 2
    num_combinations_per_sim = 3

    # Mock the return values for the helper functions
    mock_ev_data = pd.DataFrame({"to_charge_kwh": [10, 20, 30]})
    mock_hourly_loads = np.array(
        [50, 60, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    mock_generate_simulation_data.return_value = mock_ev_data
    mock_calculate_hourly_load.return_value = mock_hourly_loads

    with patch("src.demand_load_helpers.logger") as mock_logger:
        hourly_loads_df, total_demands_df = run_monte_carlo_simulation(
            num_simulations, num_combinations_per_sim, mock_logger
        )

    # Assertions on the returned DataFrames
    assert hourly_loads_df.shape == (num_simulations, 24)
    assert total_demands_df.shape == (num_simulations, 1)

    # Assert the correct values are returned
    expected_total_demand = mock_ev_data["to_charge_kwh"].sum()
    np.testing.assert_array_equal(
        total_demands_df.values.flatten(), [expected_total_demand] * num_simulations
    )
    np.testing.assert_array_equal(hourly_loads_df.iloc[0].values, mock_hourly_loads)


def test_save_simulation_results_creates_files():
    """
    Tests that `save_simulation_results` saves the two dataframes to the correct paths.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Prepare mock data
        hourly_loads_df = pd.DataFrame(np.random.rand(10, 24))
        hourly_loads_df.columns = [f"Hour_{h}" for h in range(24)]
        total_demands_df = pd.DataFrame(
            np.random.rand(10, 1), columns=["total_demand_kwh"]
        )

        # Call the function
        with patch("src.demand_load_helpers.logger") as mock_logger:
            save_simulation_results(
                hourly_loads_df, total_demands_df, temp_path, mock_logger
            )

        # Assert that the files exist
        v_max_path = temp_path / "v_max.csv"
        total_demand_path = temp_path / "total_demand.csv"

        assert v_max_path.exists()
        assert total_demand_path.exists()

        # Assert the content of v_max.csv
        read_v_max_df = pd.read_csv(v_max_path)
        assert "mean_load" in read_v_max_df.columns
        assert "Hour" in read_v_max_df.columns
        assert len(read_v_max_df) == 24

        # Assert the content of total_demand.csv
        with open(total_demand_path, "r") as f:
            content = f.read()

        expected_content = str(total_demands_df.mean().iloc[0])
        assert content == expected_content
