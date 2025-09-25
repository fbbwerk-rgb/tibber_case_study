"""
DATA PREPARATION AND FEATURE ENGINEERING

This module contains functions for retrieving and preparing energy market
data, including spot and imbalance prices, generation, consumption, and weather
data. It also includes functions for creating new features (e.g., cyclical sun
times and day-of-week dummies), randomizing data to simulate forecasts, and 
generating random data to simulate max demand load.

This is a core module that should be robust and self-contained, handling all
data-related tasks. It avoids model-specific logic and focuses solely on
transforming raw data into a clean, feature-rich dataset.
"""

# Necessary libraries
import pandas as pd
import pytz
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from astral import LocationInfo
from astral.sun import sun
from typing import Dict, Any, Union
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Initialize a logger for this module
logger = logging.getLogger(__name__)

# Turn off external logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("pkg_resources").setLevel(logging.ERROR)

# --- Helper Functions ---


def convert_entsoe_dates_to_utc(datetime_series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series of ENTSOE datetime strings to UTC timezone.

    The function handles the specific format from ENTSOE data, including
    both CET and CEST timezones, and converts them all to a uniform,
    UTC-aware datetime series.

    Args:
        datetime_series (pd.Series): A pandas Series containing datetime strings
                                     in the format 'DD/MM/YYYY HH:MM:SS (CET)' or
                                     'DD/MM/YYYY HH:MM:SS (CEST)'.

    Returns:
        pd.Series: A pandas Series with timezone-aware UTC datetime objects.

    Raises:
        ValueError: If the datetime format is not as expected.
    """
    try:
        # Split the string to isolate the time part and the timezone indicator
        start_times = datetime_series.str.split(" - ").str[0]
        # Clean the strings by removing timezone indicators
        clean_times = start_times.str.replace(" (CET)", "", regex=False).str.replace(
            " (CEST)", "", regex=False
        )

        # Convert to a naive datetime series
        dt_series = pd.to_datetime(
            clean_times, format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        if dt_series.isnull().any():
            raise ValueError(
                "Some datetime strings could not be parsed. Check data format."
            )

        # Determine DST status for each datetime based on the original string
        is_cest = start_times.str.contains("(CEST)", regex=False)

        # Apply timezone with DST consideration
        tz = pytz.timezone("Europe/Stockholm")
        utc_series = [
            tz.localize(dt, is_dst=is_cest.iloc[i]).astimezone(pytz.UTC)
            for i, dt in enumerate(dt_series)
        ]
        return pd.Series(utc_series)

    except Exception as e:
        logger.error(f"Failed to convert ENTSOE dates: {e}", exc_info=True)
        raise


def randomize_dataframe(df: pd.DataFrame, std_factor: float = 0.1) -> pd.DataFrame:
    """
    Adds a small amount of random Gaussian error to numerical columns to simulate
    forecast data.

    This function is useful for creating synthetic forecast data from historical
    actual data for model training and testing purposes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        std_factor (float): A factor to scale the standard deviation of the noise.
                            A higher value results in more noise.

    Returns:
        pd.DataFrame: A new DataFrame with randomized numerical columns, renamed
                      to indicate they are forecasts (e.g., 'column_fc').
    """
    forecast_df = df.copy()
    numerical_cols = forecast_df.select_dtypes(include=np.number).columns

    if numerical_cols.empty:
        logger.warning("No numerical columns found to randomize.")
        return forecast_df

    for col in numerical_cols:
        col_std = forecast_df[col].std()
        if np.isnan(col_std) or col_std == 0:
            logger.debug(
                f"Skipping randomization for column '{col}' due to zero or NaN standard deviation."
            )
            continue
        np.random.seed(9)
        errors = np.random.normal(
            loc=0, scale=col_std * std_factor, size=len(forecast_df)
        )
        forecast_df[col] = forecast_df[col] + errors

    # Rename columns to indicate they are forecasts
    rename_dict = {
        col: (
            f"{col.rsplit('_', 1)[0]}_fc_{col.rsplit('_', 1)[1]}"
            if "_" in col
            else f"{col}_fc"
        )
        for col in numerical_cols
    }
    return forecast_df.rename(columns=rename_dict)


def encode_cyclical_minutes(df: pd.DataFrame, time_column: str, period: int) -> None:
    """
    Encodes a numerical time column as cyclical sine and cosine features.

    This is a standard technique for handling cyclical features like time of day
    or time of year in machine learning models, as it captures the cyclical nature
    without creating a discontinuity (e.g., 23:59 and 00:00 being far apart).

    Args:
        df (pd.DataFrame): The DataFrame to modify in place.
        time_column (str): The name of the column containing the numerical time value.
        period (int): The period of the cycle (e.g., 24*60 for a day in minutes).
    """
    sin_col_name = f"{time_column}_sin"
    cos_col_name = f"{time_column}_cos"
    try:
        df[sin_col_name] = np.sin(2 * np.pi * df[time_column] / period)
        df[cos_col_name] = np.cos(2 * np.pi * df[time_column] / period)
    except Exception as e:
        logger.error(
            f"Failed to encode cyclical features for column '{time_column}': {e}",
            exc_info=True,
        )
        raise


# --- Data Loading and Processing Functions ---


def get_spot_prices(data_dir: Path) -> pd.DataFrame:
    """
    Retrieves and processes Day-ahead (spot) price data from CSV files.

    This function loads two years of ENTSOE spot price data, concatenates them,
    converts the datetime index to UTC, renames columns, and handles duplicate
    entries by averaging their values.

    Args:
        data_dir (Path): The directory path where the CSV files are located.

    Returns:
        pd.DataFrame: A processed DataFrame with a UTC datetime index and a
                      'spot_eur' column.
    """
    logger.info("Starting get_spot_prices function")
    try:
        # Define file paths
        file_path_2023 = data_dir / "GUI_ENERGY_PRICES_202212312300-202312312300.csv"
        file_path_2024 = data_dir / "GUI_ENERGY_PRICES_202312312300-202412312300.csv"

        # Check for file existence before loading
        if not file_path_2023.exists() or not file_path_2024.exists():
            raise FileNotFoundError(
                f"Missing one or both price data files in {data_dir}"
            )

        # Import data from CSVs
        spot_prices_2023_df = pd.read_csv(file_path_2023)
        spot_prices_2024_df = pd.read_csv(file_path_2024)
        spot_prices_df = pd.concat(
            [spot_prices_2023_df, spot_prices_2024_df], ignore_index=True
        )
        logger.info(f"Combined dataframes with {len(spot_prices_df)} total rows")

        # Convert start times to UTC and set as index
        spot_prices_df["Datetime UTC"] = convert_entsoe_dates_to_utc(
            spot_prices_df["MTU (CET/CEST)"]
        )
        spot_prices_df.set_index("Datetime UTC", inplace=True)

        # Drop all but the target column and rename
        spot_prices_df = spot_prices_df[["Day-ahead Price (EUR/MWh)"]].rename(
            columns={"Day-ahead Price (EUR/MWh)": "spot_eur"}
        )
        logger.info("Filtered and renamed columns to keep only 'spot_eur'")

        # If there are any duplicate indexes, average their values
        if spot_prices_df.index.duplicated().any():
            logger.warning(
                f"Found {spot_prices_df.index.duplicated().sum()} duplicate indexes, averaging their values"
            )
            spot_prices_df = spot_prices_df.groupby(spot_prices_df.index).mean()

        logger.info(f"Completed get_spot_prices. Final shape: {spot_prices_df.shape}")
        return spot_prices_df
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error in get_spot_prices: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_spot_prices: {e}", exc_info=True
        )
        raise


def get_imbalance_prices(data_dir: Path) -> pd.DataFrame:
    """
    Retrieves and processes imbalance price data from a CSV file.

    This function loads imbalance price data, converts columns with commas
    to numeric format, drops unnecessary columns, and sets a UTC-aware
    datetime index.

    Args:
        data_dir (Path): The directory path where the CSV file is located.

    Returns:
        pd.DataFrame: A processed DataFrame with a UTC datetime index and an
                      'imb_eur' column.
    """
    logger.info("Starting get_imbalance_prices function")
    file_path = data_dir / "esett_Export-2025-09-20T19_20_33_2023-2024_imb_price.csv"
    try:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Imbalance price data file not found at {file_path}"
            )

        imbalance_prices_df = pd.read_csv(file_path, sep=";")
        logger.debug(f"Loaded data with {len(imbalance_prices_df)} rows")

        # Convert string columns with commas to numeric values
        numeric_col = "Imbalance Purchase Price [EUR/MWh]"
        imbalance_prices_df[numeric_col] = pd.to_numeric(
            imbalance_prices_df[numeric_col].astype(str).str.replace(",", "."),
            errors="coerce",
        )

        # Drop unnecessary columns
        imbalance_prices_df = imbalance_prices_df.drop(
            columns=[
                "Date/Time CET/CEST",
                "MBA",
                "Imbalance Sales Price [EUR/MWh]",
                "Up Regulation Price [EUR/MWh]",
                "Down Regulation Price [EUR/MWh]",
                "Imbalance and Spot Price Difference [EUR/MWh]",
                "Value of Avoided Activation",
                "Incentivising Component",
                "Dominating Direction of Regulation Power per MBA",
            ]
        )
        logger.debug("Dropped unnecessary columns")

        # Assign UTC datetime as index
        imbalance_prices_df.rename(
            columns={"Date/Time UTC": "Datetime UTC"}, inplace=True
        )
        imbalance_prices_df["Datetime UTC"] = pd.to_datetime(
            imbalance_prices_df["Datetime UTC"], format="%d.%m.%Y/%H:%M"
        ).dt.tz_localize("UTC")
        imbalance_prices_df.set_index("Datetime UTC", inplace=True)

        # Relabel columns
        imbalance_prices_df.rename(
            columns={"Imbalance Purchase Price [EUR/MWh]": "imb_eur"}, inplace=True
        )
        logger.info(
            f"Completed get_imbalance_prices. Final shape: {imbalance_prices_df.shape}"
        )
        return imbalance_prices_df

    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error in get_imbalance_prices: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_imbalance_prices: {e}", exc_info=True
        )
        raise


def get_generation_data(data_dir: Path) -> pd.DataFrame:
    """
    Retrieves and processes electricity generation data from a CSV file.

    Args:
        data_dir (Path): The directory path where the CSV file is located.

    Returns:
        pd.DataFrame: A processed DataFrame with a UTC datetime index and
                      generation data columns.
    """
    logger.info("Starting get_generation_data function")
    file_path = data_dir / "esett_Export-2025-09-20T19_33_12_production.csv"
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Generation data file not found at {file_path}")

        generation_df = pd.read_csv(file_path, sep=";")

        # Convert string columns with commas to numeric values
        numeric_columns = [
            "Production Total [MWh]",
            "Hydro [MWh]",
            "Nuclear [MWh]",
            "Solar [MWh]",
            "Thermal [MWh]",
            "Wind Onshore [MWh]",
        ]
        for col in numeric_columns:
            generation_df[col] = pd.to_numeric(
                generation_df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

        # Create new column 'Wind [MWh]' by summing onshore and offshore
        generation_df["Wind [MWh]"] = (
            generation_df["Wind Onshore [MWh]"] + generation_df["Wind Offshore [MWh]"]
        )

        # Drop unnecessary columns
        generation_df = generation_df.drop(
            columns=[
                "Date/Time CET/CEST",
                "MBA",
                "Production Total [MWh]",
                "Energy Storage [MWh]",
                "Other [MWh]",
                "Wind Onshore [MWh]",
                "Wind Offshore [MWh]",
            ]
        )

        # Assign UTC datetime as index and relabel columns
        generation_df.rename(columns={"Date/Time UTC": "Datetime UTC"}, inplace=True)
        generation_df["Datetime UTC"] = pd.to_datetime(
            generation_df["Datetime UTC"], format="%d.%m.%Y/%H:%M"
        ).dt.tz_localize("UTC")
        generation_df.set_index("Datetime UTC", inplace=True)

        generation_df.rename(
            columns={
                "Hydro [MWh]": "hydro_mwh",
                "Nuclear [MWh]": "nuclear_mwh",
                "Solar [MWh]": "solar_mwh",
                "Thermal [MWh]": "thermal_mwh",
                "Wind [MWh]": "wind_mwh",
            },
            inplace=True,
        )

        logger.info(
            f"Completed get_generation_data. Final shape: {generation_df.shape}"
        )
        return generation_df
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error in get_generation_data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_generation_data: {e}", exc_info=True
        )
        raise


def get_consumption_data(data_dir: Path) -> pd.DataFrame:
    """
    Retrieves and processes electricity consumption data from a CSV file.

    Args:
        data_dir (Path): The directory path where the CSV file is located.

    Returns:
        pd.DataFrame: A processed DataFrame with a UTC datetime index and a
                      'demand_mwh' column.
    """
    logger.info("Starting get_consumption_data function")
    file_path = data_dir / "esett_Export-2025-09-20T19_34_52_consumption.csv"
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Consumption data file not found at {file_path}")

        consumption_df = pd.read_csv(file_path, sep=";")

        # Convert string columns with commas to numeric values
        numeric_col = "Consumption Total [MWh]"
        consumption_df[numeric_col] = pd.to_numeric(
            consumption_df[numeric_col].astype(str).str.replace(",", "."),
            errors="coerce",
        )

        # Drop unnecessary columns
        consumption_df = consumption_df.drop(
            columns=[
                "Date/Time CET/CEST",
                "MBA",
                "Metered [MWh]",
                "Profiled [MWh]",
                "Flex-settled [MWh]",
            ]
        )

        # Assign UTC datetime as index and relabel columns
        consumption_df.rename(columns={"Date/Time UTC": "Datetime UTC"}, inplace=True)
        consumption_df["Datetime UTC"] = pd.to_datetime(
            consumption_df["Datetime UTC"], format="%d.%m.%Y/%H:%M"
        ).dt.tz_localize("UTC")
        consumption_df.set_index("Datetime UTC", inplace=True)

        consumption_df.rename(
            columns={"Consumption Total [MWh]": "demand_mwh"}, inplace=True
        )

        logger.info(
            f"Completed get_consumption_data. Final shape: {consumption_df.shape}"
        )
        return consumption_df
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error in get_consumption_data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_consumption_data: {e}", exc_info=True
        )
        raise


def get_weather_data(data_dir: Path) -> pd.DataFrame:
    """
    Retrieves and processes weather data from CSV files.

    Args:
        data_dir (Path): The directory path where the CSV files are located.

    Returns:
        pd.DataFrame: A processed DataFrame with a UTC datetime index and weather
                      data columns.
    """
    logger.info("Starting get_weather_data function")
    file_path_2023 = data_dir / "ninja_weather_stockholm_2023.csv"
    file_path_2024 = data_dir / "ninja_weather_stockholm_2024.csv"
    try:
        if not file_path_2023.exists() or not file_path_2024.exists():
            raise FileNotFoundError(
                f"Missing one or both weather data files in {data_dir}"
            )

        weather_2023_df = pd.read_csv(file_path_2023, skiprows=3)
        weather_2024_df = pd.read_csv(file_path_2024, skiprows=3)
        weather_df = pd.concat([weather_2023_df, weather_2024_df], ignore_index=True)

        # Assign UTC datetime as index and drop unnecessary columns
        weather_df["Datetime UTC"] = pd.to_datetime(
            weather_df["time"], format="%Y-%m-%d %H:%M"
        ).dt.tz_localize("UTC")
        weather_df.set_index("Datetime UTC", inplace=True)
        weather_df = weather_df.drop(columns=["time", "local_time", "precsnoland"])

        # Relabel columns
        weather_df.rename(
            columns={"t2m": "temp_celsius", "prectotland": "precipitation_mmh"},
            inplace=True,
        )

        logger.info(f"Completed get_weather_data. Final shape: {weather_df.shape}")
        return weather_df
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error in get_weather_data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_weather_data: {e}", exc_info=True
        )
        raise


# --- Feature Engineering Functions ---


def get_cyclical_sun_times(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds cyclical sine and cosine features for sunrise and sunset
    times to the DataFrame.

    This function uses the 'astral' library to compute sunrise and sunset times
    for a given location (Stockholm, Sweden) and then encodes the time
    difference to these events as cyclical features. This is a crucial step
    for models predicting energy usage or generation, as solar events have a
    strong influence.

    Args:
        dataframe (pd.DataFrame): The input DataFrame. The index must be a
                                  timezone-aware datetime index.

    Returns:
        pd.DataFrame: A new DataFrame with the added cyclical sun time features.
    """
    logger.info("Starting get_cyclical_sun_times function")
    try:
        private_dataframe = dataframe.copy()
        private_dataframe.index = pd.to_datetime(private_dataframe.index, utc=True)

        location = LocationInfo(
            name="Stockholm",
            region="Sweden",
            timezone="Europe/Stockholm",
            latitude=59.3293,
            longitude=18.0686,
        )

        unique_dates = pd.to_datetime(private_dataframe.index.date).unique()
        sun_times = {
            date: sun(observer=location.observer, date=date, tzinfo=pytz.UTC)
            for date in unique_dates
        }
        sun_df = pd.DataFrame.from_dict(sun_times, orient="index")
        sun_df.index.name = "Datetime UTC"

        # Create mapping from date to sunrise/sunset
        date_map = pd.to_datetime(private_dataframe.index.date)
        private_dataframe["sunrise"] = date_map.map(sun_df["sunrise"])
        private_dataframe["sunset"] = date_map.map(sun_df["sunset"])

        # Calculate time difference in minutes
        private_dataframe["time_to_sunrise"] = (
            private_dataframe["sunrise"] - private_dataframe.index
        ).dt.total_seconds() / 60
        private_dataframe["time_to_sunset"] = (
            private_dataframe["sunset"] - private_dataframe.index
        ).dt.total_seconds() / 60

        # Encode cyclical features
        encode_cyclical_minutes(private_dataframe, "time_to_sunrise", 24 * 60)
        encode_cyclical_minutes(private_dataframe, "time_to_sunset", 24 * 60)

        # Join the new features back to the original dataframe
        dataframe = dataframe.join(
            private_dataframe[
                [
                    "time_to_sunrise_sin",
                    "time_to_sunrise_cos",
                    "time_to_sunset_sin",
                    "time_to_sunset_cos",
                ]
            ],
            how="left",
        )
        logger.info(f"Completed get_cyclical_sun_times. Final shape: {dataframe.shape}")
        return dataframe
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_cyclical_sun_times: {e}",
            exc_info=True,
        )
        raise


def get_dow_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new DataFrame with day of week (DoW) dummy variables.

    Args:
        df (pd.DataFrame): The input DataFrame. The index must be a datetime
                           index.

    Returns:
        pd.DataFrame: A new DataFrame with the added DoW dummy columns.
    """
    logger.info("Starting get_dow_dummies function.")
    try:
        # Create a copy and ensure index is datetime
        dataframe = df.copy()
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)

        # Extract the day of week (Monday=0, Sunday=6)
        weekday_series = dataframe.index.weekday

        # Create dummy variables and add to the DataFrame
        day_dummies = pd.get_dummies(weekday_series, prefix="dow", dtype=int)
        day_dummies.index = dataframe.index
        dataframe = pd.concat([dataframe, day_dummies], axis=1)

        logger.info(f"Finished get_dow_dummies. Final shape: {dataframe.shape}")
        return dataframe
    except Exception as e:
        logger.error(f"An error occurred in get_dow_dummies: {e}", exc_info=True)
        raise


def generate_random_time(min_time_str: str, max_time_str: str) -> str:
    """
    Generates a random time string within a specified range.

    Args:
        min_time_str: The minimum time in 'HH:MM' format.
        max_time_str: The maximum time in 'HH:MM' format.

    Returns:
        A randomly generated time string in 'HH:MM' format.
    """
    FMT = "%H:%M"
    min_time = datetime.strptime(min_time_str, FMT)
    max_time = datetime.strptime(max_time_str, FMT)
    time_diff = max_time - min_time
    np.random.seed(9)
    random_minutes = np.random.randint(0, int(time_diff.total_seconds() / 60))
    random_time = min_time + timedelta(minutes=random_minutes)
    return random_time.strftime(FMT)


# --- Main Orchestration Function ---


def limit_price_data_prep(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Orchestrates the data preparation process for limit price forecasting.

    This is the main function that coordinates the loading, cleaning, and
    feature engineering of all data sources to create a final, unified dataset
    ready for model training.

    Args:
        data_dir (Union[str, Path]): The Path object or string for the directory
                                     containing all raw data CSV files.

    Returns:
        pd.DataFrame: The final, prepared DataFrame with all features and
                      the target variable.
    """
    logger.info("Starting limit_price_data_prep function.")

    # Ensure data_dir is a Path object
    data_dir = Path(data_dir)

    try:
        # Step 1: Extract and join raw data
        logger.info("--> Extracting and joining raw data from sources...")
        generation_df = get_generation_data(data_dir=data_dir)
        consumption_df = get_consumption_data(data_dir=data_dir)
        weather_df = get_weather_data(data_dir=data_dir)

        # Inner join to ensure only matching timestamps are kept
        combined_df = generation_df.join([consumption_df, weather_df], how="inner")
        logger.info(f"Combined dataframes. Shape: {combined_df.shape}")

        # Step 2: Process price data
        logger.info("--> Processing price data and calculating spread.")
        spot_prices_df = get_spot_prices(data_dir=data_dir)
        imbalance_prices_df = get_imbalance_prices(data_dir=data_dir)
        combined_df = pd.concat(
            [combined_df, spot_prices_df[["spot_eur"]]], axis=1, join="inner"
        )

        prices_df = spot_prices_df.join(imbalance_prices_df, how="inner")
        prices_df["spread_eur"] = prices_df["imb_eur"] - prices_df["spot_eur"]
        prices_df["sign_spread"] = np.where(prices_df["spread_eur"] >= 0, 1, -1)
        logger.info(f"Price data processed. Shape: {prices_df.shape}")

        # Step 3: Generate simulated forecasts and cyclical features
        logger.info("--> Generating simulated forecasts and new features.")
        # Ensure the forecasts are created on the combined feature data
        forecasts_df = randomize_dataframe(combined_df, std_factor=0.15)
        full_features_df = get_cyclical_sun_times(forecasts_df)
        full_features_df = get_dow_dummies(full_features_df)
        logger.info(f"All features added. Shape: {full_features_df.shape}")

        # Step 4: Merge all data
        logger.info("--> Merging price data with features to create final dataset.")
        # Join the full_features_df with the prices_df
        full_data_df = prices_df.join(full_features_df, how="inner")
        logger.info(f"Final dataset created. Shape: {full_data_df.shape}")

        # Step 5: Restrict the data to specific hours
        logger.info("--> Restricting data to 17:00 - 09:00 the next day.")
        full_data_df = full_data_df.between_time("17:00", "09:00")
        logger.info(f"Data restriction complete. Final shape: {full_data_df.shape}")

        logger.info("limit_price_data_prep function finished successfully.")
        return full_data_df
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in limit_price_data_prep: {e}", exc_info=True
        )
        raise


def generate_simulation_data(
    num_combinations: int, car_models: dict, charging_speeds: list, limits: dict
) -> pd.DataFrame:
    """
    Generates a DataFrame of random EV charging scenarios for one simulation run.

    Args:
        num_combinations: The number of EV scenarios to generate.
        car_models: A dictionary of car models and their specifications.
        charging_speeds: A list of possible home charging speeds in kW.
        limits: A dictionary defining the min/max for arrival, departure, and initial SoC.

    Returns:
        A pandas DataFrame with all generated EV data for one simulation.
    """
    car_models_list = []
    battery_size_list = []
    home_max_charging_speed_list = []
    arrival_time_list = []
    departure_time_list = []
    initial_soc_perc_list = []

    for _ in range(num_combinations):
        np.random.seed(9)
        car_model_name = np.random.choice(list(car_models.keys()))
        car_info = car_models[car_model_name]
        home_charging_speed = np.random.choice(charging_speeds)

        car_models_list.append(car_model_name)
        battery_size_list.append(car_info["battery_size_kwh"])
        home_max_charging_speed_list.append(home_charging_speed)
        arrival_time_list.append(
            generate_random_time(
                limits["arrival_time"]["min"], limits["arrival_time"]["max"]
            )
        )
        departure_time_list.append(
            generate_random_time(
                limits["departure_time"]["min"], limits["departure_time"]["max"]
            )
        )
        initial_soc_perc_list.append(
            np.random.uniform(
                limits["initial_soc_perc"]["min"], limits["initial_soc_perc"]["max"]
            )
        )

    ev_data = pd.DataFrame(
        {
            "car_model": car_models_list,
            "battery_size_kwh": battery_size_list,
            "max_charging_speed_kw": home_max_charging_speed_list,
            "arrival_time": arrival_time_list,
            "departure_time": departure_time_list,
            "initial_soc_perc": initial_soc_perc_list,
        }
    )

    ev_data["to_charge_kwh"] = round(
        ev_data["battery_size_kwh"] * (1 - ev_data["initial_soc_perc"]) * 0.9, 3
    )

    return ev_data
