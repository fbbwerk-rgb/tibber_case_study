import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import pytz
from data_prep import (
    convert_entsoe_dates_to_utc,
    randomize_dataframe,
    get_cyclical_sun_times,
    get_dow_dummies,
)


@pytest.fixture
def sample_df():
    """Provides a sample DataFrame with a timezone-aware index for testing."""
    utc = pytz.UTC
    index = pd.to_datetime(
        ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"]
    ).tz_localize(utc)
    data = {"value1": [10.5, 20.2, 30.8], "value2": [100, 200, 300]}
    return pd.DataFrame(data, index=index)


def test_convert_entsoe_dates_to_utc_with_valid_data():
    """
    Tests that the date conversion function correctly handles a mix of CET
    and CEST formats and converts them to UTC.
    """
    sample_dates = pd.Series(
        [
            "01/01/2023 00:00:00 (CET) - 01/01/2023 01:00:00 (CET)",
            "26/03/2023 01:00:00 (CET) - 26/03/2023 03:00:00 (CEST)",
            "01/07/2023 00:00:00 (CEST) - 01/07/2023 01:00:00 (CEST)",
        ]
    )

    expected_utc = pd.Series(
        [
            pd.to_datetime("2022-12-31 23:00:00+00:00", utc=True),
            pd.to_datetime("2023-03-26 00:00:00+00:00", utc=True),
            pd.to_datetime("2023-06-30 22:00:00+00:00", utc=True),
        ]
    )

    converted_dates = convert_entsoe_dates_to_utc(sample_dates)

    # Check if the converted series matches the expected UTC values
    pd.testing.assert_series_equal(converted_dates, expected_utc, check_names=False)


def test_randomize_dataframe_adds_noise_and_renames(sample_df):
    """
    Tests that the randomization function adds noise to numerical columns and
    renames the columns as expected.
    """
    np.random.seed(0)  # Seed for reproducibility

    # Make a copy of the original data to ensure the noise is correctly added
    original_df = sample_df.copy()

    # Execute the function
    randomized_df = randomize_dataframe(original_df)

    # Assertions
    assert "value1_fc" in randomized_df.columns
    assert "value2_fc" in randomized_df.columns
    assert "value1" not in randomized_df.columns
    assert "value2" not in randomized_df.columns

    # Verify that the new columns are not identical to the originals
    assert not np.array_equal(
        randomized_df["value1_fc"].values, original_df["value1"].values
    )
    assert not np.array_equal(
        randomized_df["value2_fc"].values, original_df["value2"].values
    )


@patch("data_prep.sun", autospec=True)
def test_get_cyclical_sun_times(mock_sun, sample_df):
    """
    Tests that `get_cyclical_sun_times` correctly calculates and adds sine/cosine
    features for sunrise and sunset, by mocking the external 'astral' library.
    """
    # Mock the return values for sun() to be predictable
    mock_sun.side_effect = [
        {
            "sunrise": pd.to_datetime("2023-01-01 07:00:00+00:00", utc=True),
            "sunset": pd.to_datetime("2023-01-01 17:00:00+00:00", utc=True),
        },
        {
            "sunrise": pd.to_datetime("2023-01-02 07:00:00+00:00", utc=True),
            "sunset": pd.to_datetime("2023-01-02 17:00:00+00:00", utc=True),
        },
        {
            "sunrise": pd.to_datetime("2023-01-03 07:00:00+00:00", utc=True),
            "sunset": pd.to_datetime("2023-01-03 17:00:00+00:00", utc=True),
        },
    ]

    df_with_sun_times = get_cyclical_sun_times(sample_df)

    # Assert that the new columns exist
    expected_cols = [
        "time_to_sunrise_sin",
        "time_to_sunrise_cos",
        "time_to_sunset_sin",
        "time_to_sunset_cos",
    ]
    assert all(col in df_with_sun_times.columns for col in expected_cols)

    # Expected values for the first row (time_to_sunrise = -180 minutes, period = 1440)
    expected_sunrise_sin = np.sin(2 * np.pi * (-180) / (24 * 60))
    expected_sunrise_cos = np.cos(2 * np.pi * (-180) / (24 * 60))

    # Expected values for the first row (time_to_sunset = 420 minutes, period = 1440)
    expected_sunset_sin = np.sin(2 * np.pi * (420) / (24 * 60))
    expected_sunset_cos = np.cos(2 * np.pi * (420) / (24 * 60))

    # Test the values for the first row
    assert np.isclose(
        df_with_sun_times.iloc[0]["time_to_sunrise_sin"], expected_sunrise_sin
    )
    assert np.isclose(
        df_with_sun_times.iloc[0]["time_to_sunrise_cos"], expected_sunrise_cos
    )
    assert np.isclose(
        df_with_sun_times.iloc[0]["time_to_sunset_sin"], expected_sunset_sin
    )
    assert np.isclose(
        df_with_sun_times.iloc[0]["time_to_sunset_cos"], expected_sunset_cos
    )


def test_get_dow_dummies_correct_columns_and_values(sample_df):
    """
    Tests that `get_dow_dummies` correctly creates one-hot encoded columns
    for the day of the week.
    """
    # Sample index is Sunday, Monday, Tuesday
    df_with_dummies = get_dow_dummies(sample_df)

    # Assert that the new dummy columns are present
    assert "dow_0" in df_with_dummies.columns  # Monday
    assert "dow_1" in df_with_dummies.columns  # Tuesday
    assert "dow_6" in df_with_dummies.columns  # Sunday
    assert "dow_2" not in df_with_dummies.columns  # Wednesday

    # Assert the correct values (1 for the corresponding day, 0 otherwise)
    assert df_with_dummies.loc["2023-01-01 10:00:00+00:00", "dow_6"] == 1
    assert df_with_dummies.loc["2023-01-02 11:00:00+00:00", "dow_0"] == 1
    assert df_with_dummies.loc["2023-01-03 12:00:00+00:00", "dow_1"] == 1

    # Assert that rows have a single '1' and rest '0'
    assert all(
        df_with_dummies.loc["2023-01-01 10:00:00+00:00", ["dow_0", "dow_1"]] == 0
    )
    assert all(
        df_with_dummies.loc["2023-01-02 11:00:00+00:00", ["dow_1", "dow_6"]] == 0
    )
    assert all(
        df_with_dummies.loc["2023-01-03 12:00:00+00:00", ["dow_0", "dow_6"]] == 0
    )
