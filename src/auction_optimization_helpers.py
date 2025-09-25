""" auction_optimization_helpers.py

Helper functions for EV charging optimization.

This module provides functions to load data and models, prepare data for
optimization, and solve the cost optimization problem to determine an
optimal EV charging schedule. The goal is to minimize the total charging cost
over specified time windows.

"""

import logging
import pickle
import sys
import numpy as np
import pandas as pd
from datetime import timezone
from pathlib import Path
from scipy.optimize import linprog
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Imports from local modules
try:
    from data_prep import limit_price_data_prep
    from limit_price_helpers import prepare_and_split_data
except ImportError as e:
    print(f"Error: Required local module 'data_prep' not found. {e}", file=sys.stderr)
    sys.exit(1)

### Initialize logger
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_data_limit_price_prediction(data_dir: str, final_test: bool = False):
    """
    Loads and prepares data for the limit price prediction models.

    This function calls assumed helper functions `limit_price_data_prep` and
    `prepare_and_split_data` to load a full dataset and then split it into
    training and testing sets with appropriate scaling.

    Args:
        data_dir (str): The directory where the data files are located.
        final_test (bool): If True, returns y2_train instead of y2_test for testing purposes.

    Returns:
        tuple: A tuple containing the prepared and split data:
               (X_train_scaled, X_test_scaled, y1_train, y1_test,
               y2_train_scaled, y2_test, y2_scaler, full_data_df)

    Raises:
        Exception: Catches and logs any exceptions that occur during data
                   loading and preparation.
    """
    try:
        logger.info("--> Loading full data from directory: %s", data_dir)
        full_data_df = limit_price_data_prep(data_dir=data_dir)

        logger.info("----> Preparing and splitting data for modeling.")
        (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_test,
            y2_scaler,
        ) = prepare_and_split_data(
            data_dir=data_dir, shuffle=False, final_test=final_test
        )
        logger.info("----> Data loaded and prepared successfully.")

        return (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_test,
            y2_scaler,
            full_data_df,
        )
    except Exception as e:
        logger.error("----> An error occurred during data loading: %s", e)
        raise


def load_models():
    """
    Loads machine learning models from pickle files.

    It attempts to load two models, a classifier (model 1) and a regressor (model 2), from
    predefined file paths.

    Returns:
        tuple: A tuple containing the loaded classifier and regressor models.
               Returns (None, None) if the files are not found or an error
               occurs.
    """
    clf_blend_1 = None
    reg_blend_2 = None
    try:
        with open("./src/data/models/ensemble_clf.pkl", "rb") as f:
            clf_blend_1 = pickle.load(f)

        with open("./src/data/models/ensemble_reg.pkl", "rb") as f:
            reg_blend_2 = pickle.load(f)

        logger.info("--> Models loaded successfully.")
    except FileNotFoundError as e:
        logger.warning("--> Model files not found: %s. Skipping model loading.", e)
    except pickle.PickleError as e:
        logger.error("--> Error unpickling model files: %s", e)
    except Exception as e:
        logger.error("--> An unexpected error occurred while loading models: %s", e)

    return clf_blend_1, reg_blend_2


def load_demand_simulation_data(data_dir: Path) -> tuple[pd.DataFrame, float]:
    """
    Loads EV charging capacity (v_max) and total energy demand from
    simulation results.

    Args:
        data_dir (Path): The directory where the simulation results are saved.

    Returns:
        tuple[pd.DataFrame, float]: A tuple containing the v_max DataFrame
                                    and the total demand as a float.

    Raises:
        FileNotFoundError: If 'v_max.csv' or 'total_demand.csv' are not found.
        ValueError: If the total demand file does not contain a valid float.
    """
    v_max_path = data_dir / "v_max.csv"
    total_demand_path = data_dir / "total_demand.csv"

    try:
        logger.info("--> Loading v_max from %s", v_max_path)
        v_max_df = pd.read_csv(v_max_path, index_col="Hour")
        logger.info("--> Loading total demand from %s", total_demand_path)

        # Reads the single float value representing total demand from the file
        with open(total_demand_path, "r", encoding="utf-8") as f:
            total_demand = float(f.read().strip())
        logger.info("----> Demand simulation data loaded successfully.")
    except FileNotFoundError as e:
        logger.error("----> Required simulation file not found: %s", e)
        raise
    except ValueError as e:
        logger.error("----> Could not parse total demand as a float: %s", e)
        raise
    except Exception as e:
        logger.error(
            "----> An unexpected error occurred while loading demand data: %s", e
        )
        raise

    return v_max_df, total_demand


def prepare_data_for_optimization(
    clf_blend_1,
    reg_blend_2,
    X_test_scaled: pd.DataFrame,
    y2_test: pd.Series,
    y2_scaler,
    full_data_df: pd.DataFrame,
    v_max: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepares data for optimization by generating predictions from the models
    and merging them with relevant data from the full dataset.

    This function calculates the `expected_cost_eur` using model predictions,
    merges the data with `v_max` values, and formats the output DataFrame for
    the optimization function.

    Args:
        clf_blend_1 (object): The loaded classifier model.
        reg_blend_2 (object): The loaded regressor model.
        X_test_scaled (pd.DataFrame): The scaled feature data for testing.
        y2_test (pd.Series): The true target values for the regression model.
        y2_scaler (object): The scaler used for the regression target.
        full_data_df (pd.DataFrame): The complete dataset containing all features.
        v_max (pd.DataFrame): A DataFrame containing 'mean_load' values by hour.

    Returns:
        pd.DataFrame: The prepared DataFrame ready for optimization. Returns
                      None if required models are not provided.

    Raises:
        ValueError: If a required column is missing from the input DataFrames.
    """
    if clf_blend_1 is None or reg_blend_2 is None:
        logger.error("--> Models are not loaded. Cannot prepare data for optimization.")
        return None

    try:
        logger.info("----> Generating predictions from models.")

        # Predict the probability of a gain (positive class) from the classifier
        y1_pred_ensemble = clf_blend_1.predict_proba(X_test_scaled)

        # Predict the estimated gain from the regressor
        y2_pred_scaled = reg_blend_2.predict(X_test_scaled)

        # Inverse transform to get the actual gain value in original units
        y2_pred = y2_scaler.inverse_transform(y2_pred_scaled.reshape(-1, 1)).flatten()
        logger.info("------> Predictions generated successfully.")

        output_df = pd.DataFrame(y2_test.copy(), index=y2_test.index)
        output_df["prob_gain"] = y1_pred_ensemble[:, 1]
        output_df["estimated_gain"] = y2_pred

        logger.info("----> Merging with relevant columns from full dataset.")
        # Merge key price data points from the full dataset using the shared index
        output_df["spot_eur"] = full_data_df.loc[output_df.index, "spot_eur"]
        output_df["spot_fc_eur"] = full_data_df.loc[output_df.index, "spot_fc_eur"]
        output_df["imb_eur"] = full_data_df.loc[output_df.index, "imb_eur"]

        # Define risk tolerance parameter TAU from config.yaml
        with open(".\src\config.yaml", "r") as f:
            config = yaml.safe_load(f)
            TAU = config["TAU"]

        # Calculate the final expected cost: forecasted spot price minus the expected gain
        # The expected gain is the probability of a gain multiplied by the estimated gain amount.
        output_df["expected_cost_eur"] = (
            output_df["spot_fc_eur"]
            + output_df["prob_gain"] * output_df["estimated_gain"] * 1 / TAU
        )
        output_df["hour_of_day"] = output_df.index.hour
        output_df["v_max"] = output_df["hour_of_day"].map(
            dict(zip(range(24), v_max["mean_load"]))
        )
        output_df.drop(columns=["hour_of_day"], inplace=True)

        logger.info("------> Data preparation for optimization completed.")
        return output_df

    except KeyError as e:
        logger.error("------> Missing a required column in the dataframes: %s", e)
        raise ValueError(f"Missing a required column in the dataframes: {e}") from e
    except Exception as e:
        logger.error(
            "------> An unexpected error occurred during data preparation: %s", e
        )
        raise


def optimize_charging_schedule(
    output_df: pd.DataFrame,
    total_energy_demand: float,
    cost_col: str = "expected_cost_eur",
    vmax_col: str = "v_max",
    datetime_col: str = None,
) -> pd.DataFrame:
    """
    Extends a historical DataFrame with an 'optimal_plan' column that
    minimizes EV charging costs for each night session.

    This function iterates through night sessions (16:00 to 09:00 next day)
    and uses linear programming to find the optimal charging volume for each
    hour to meet a total energy demand while minimizing cost.

    Args:
        output_df (pd.DataFrame): The prepared DataFrame containing cost and
                                  v_max information.
        total_energy_demand (float): The total energy to be charged in kWh for
                                     each night session.
        cost_col (str): The name of the column containing the cost.
                        Defaults to 'expected_cost_eur'.
        vmax_col (str): The name of the column containing the maximum charging
                        volume. Defaults to 'v_max'.
        datetime_col (str, optional): The name of the datetime column if the
                                      index is not already a DatetimeIndex.
                                      Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with an added 'optimal_plan' column.

    Raises:
        ValueError: If `output_df` does not have a DatetimeIndex and
                    `datetime_col` is not specified.
        Exception: Catches and logs general exceptions from the optimization loop.
    """
    logger.info("--> Starting optimization of charging schedule.")

    df = output_df.copy()

    try:
        if datetime_col:
            df.index = pd.to_datetime(df[datetime_col])
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "----> Provide a datetime index or specify `datetime_col`."
            )

        df = df.sort_index()
        tz = df.index.tz or timezone.utc
        df["optimal_plan"] = 0.0
        start_dates = pd.date_range(
            start=df.index.min().floor("D"),
            end=df.index.max().ceil("D") - pd.Timedelta(days=1),
            freq="D",
            tz=tz,
        )

    except (ValueError, KeyError) as e:
        logger.error("----> Error preparing data for optimization: %s", e)
        raise
    except Exception as e:
        logger.error("----> An unexpected error occurred during initial setup: %s", e)
        raise

    for start_day in start_dates:
        start = start_day + pd.Timedelta(hours=16)
        end = start_day + pd.Timedelta(days=1, hours=10)
        target_df = df.loc[(df.index >= start) & (df.index < end)]

        if target_df.empty:
            logger.warning(
                "----> No data found for night session starting on %s. Skipping.",
                start.date(),
            )
            continue

        # Set up the linear programming problem
        # The objective function coefficients: `c` is the vector of costs to minimize
        c = target_df[cost_col].values

        # The equality constraint: a single constraint that the total energy charged equals total_energy_demand
        # A_eq is the matrix of coefficients, with one row of all ones to sum the variables
        A_eq = np.ones((1, len(c)))

        # b_eq is the right-hand side of the equality constraint
        b_eq = [total_energy_demand]

        # The bounds for each variable: `bounds` specifies that each hourly charging volume `x_i`
        # must be between 0 and the maximum capacity for that hour (v_max_i)
        bounds = [(0, v) for v in target_df[vmax_col].values]

        try:
            # Call the linear programming solver
            result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            if result.success:
                # If successful, assign the optimized hourly charging volumes (`result.x`) back to the DataFrame
                df.loc[target_df.index, "optimal_plan"] = result.x
                logger.info(
                    "----> Optimization successful for night session starting on %s.",
                    start.date(),
                )
            else:
                logger.warning(
                    "----> Optimization failed for night session starting on %s: %s",
                    start.date(),
                    result.message,
                )
                df.loc[target_df.index, "optimal_plan"] = 0.0

        except Exception as e:
            logger.error(
                "----> An error occurred during linprog for session on %s: %s",
                start.date(),
                e,
            )
            df.loc[target_df.index, "optimal_plan"] = 0.0

    return df


def save_optimized_schedule(df: pd.DataFrame, filepath: Path, logger):
    """
    Saves the optimized charging schedule to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing the optimized schedule.
        filepath (Path): The Path object for the output CSV file.
        logger (object): Logger object for logging information.

    Raises:
        IOError: If an error occurs while writing the file.
    """
    logger.info("--> Attempting to save optimized schedule to %s", filepath)
    try:
        df.to_csv(filepath)
        logger.info("----> Successfully saved optimized schedule to %s", filepath)
    except IOError as e:
        logger.error("----> Error saving the optimized schedule: %s", e)
        raise
    except Exception as e:
        logger.error("----> An unexpected error occurred while saving the file: %s", e)
        raise
