"""
DEMAND LOAD HELPERS

This module contains functions and constants for the Monte Carlo simulation of EV charging demand.

The module provides the core logic for the simulation, including:
- **EV Parameters**: Constants defining car models and charging limits.
- **Load Calculation**: A function to compute the aggregate hourly load from simulated EV data.
- **Simulation Orchestration**: A function to run the complete Monte Carlo simulation over multiple scenarios.
- **Analysis and Visualization**: A function to summarize the simulation results and generate a plot.

This module is a self-contained unit for modeling EV charging behavior and its impact on the electrical grid.
"""

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
from data_prep import generate_simulation_data
from typing import Union
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


### Initialize logger
logger = logging.getLogger(__name__)

# # Constants
# # Define reasonable ranges for EV and charging parameters
# CAR_MODELS = {
#     "Tesla Y": {"battery_size_kwh": 75},
#     "Volvo EX30": {"battery_size_kwh": 69},
#     "Volvo XC40": {"battery_size_kwh": 82},
#     "Polestar 2": {"battery_size_kwh": 70},
#     "VW ID.4": {"battery_size_kwh": 82},
# }
# TYPICAL_SWE_HOME_MAX_CHARGING_KW = [3.7, 7.4, 7.4, 11, 11, 11, 22, 22, 22]
# LIMITS = {
#     "arrival_time": {"min": "17:00", "max": "21:59"},
#     "departure_time": {"min": "05:00", "max": "10:01"},
#     "initial_soc_perc": {"min": 0.10, "max": 0.96},
# }


# --- Functions ---
def calculate_hourly_load(ev_data: pd.DataFrame) -> np.ndarray:
    """
    Calculates the aggregate potential charging load for each hour.

    The load is calculated as the sum of 'to_charge_kwh' for all EVs
    that are present for the entirety of a given hour.

    Args:
        ev_data: A DataFrame containing EV scenarios for a single simulation.

    Returns:
        A NumPy array of 24 elements, representing the aggregate load (kWh) for each hour.
    """
    ev_data["arrival_dt"] = pd.to_datetime(ev_data["arrival_time"], format="%H:%M")
    ev_data["departure_dt"] = pd.to_datetime(ev_data["departure_time"], format="%H:%M")
    ev_data.loc[
        ev_data["departure_dt"] < ev_data["arrival_dt"], "departure_dt"
    ] += timedelta(days=1)

    hourly_loads = np.zeros(24)
    for hour in range(24):
        hour_start = pd.to_datetime(f"{hour:02d}:00", format="%H:%M")
        hour_end = hour_start + timedelta(hours=1)

        adjusted_hour_start = hour_start
        adjusted_hour_end = hour_end
        if hour < 10:
            adjusted_hour_start += timedelta(days=1)
            adjusted_hour_end += timedelta(days=1)

        cars_present_in_hour = ev_data[
            (ev_data["arrival_dt"] <= adjusted_hour_start)
            & (ev_data["departure_dt"] >= adjusted_hour_end)
        ]

        hourly_loads[hour] = cars_present_in_hour["to_charge_kwh"].sum()

    return hourly_loads


def run_monte_carlo_simulation(
    num_simulations: int, num_combinations_per_sim: int, logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the full Monte Carlo simulation for EV charging loads and total demand.

    Args:
        num_simulations: The number of simulation runs.
        num_combinations_per_sim: The number of EVs to simulate per run.
        logger: Logger object for logging information.

    Returns:
        A tuple containing two DataFrames:
        - The first DataFrame has hourly loads for each simulation.
        - The second DataFrame has the total energy demand for each simulation.
    """
    logger.info(f"--> Starting {num_simulations} Monte Carlo simulations...")
    hourly_aggregate_loads = np.zeros((num_simulations, 24))
    total_demands = np.zeros(num_simulations)

    # Import CAR_MODELS, TYPICAL_SWE_HOME_MAX_CHARGING_KW, LIMITS from config.yaml
    with open(".\src\config.yaml", "r") as f:
        config = yaml.safe_load(f)
        CAR_MODELS = config["CAR_MODELS"]
        TYPICAL_SWE_HOME_MAX_CHARGING_KW = config["TYPICAL_SWE_HOME_MAX_CHARGING_KW"]
        LIMITS = config["LIMITS"]

    for sim in range(num_simulations):
        if sim % 50 == 0:
            logger.info(f"----> Running simulation {sim+1}/{num_simulations}...")

        ev_data = generate_simulation_data(
            num_combinations_per_sim,
            CAR_MODELS,
            TYPICAL_SWE_HOME_MAX_CHARGING_KW,
            LIMITS,
        )
        hourly_loads = calculate_hourly_load(ev_data)

        # Calculate the total energy demand for this simulation
        total_demand = ev_data["to_charge_kwh"].sum()

        hourly_aggregate_loads[sim, :] = hourly_loads
        total_demands[sim] = total_demand

    logger.info("----> Simulations complete.")
    hourly_loads_df = pd.DataFrame(
        hourly_aggregate_loads, columns=[f"Hour_{h}" for h in range(24)]
    )
    total_demands_df = pd.DataFrame(total_demands, columns=["total_demand_kwh"])

    return hourly_loads_df, total_demands_df


def save_simulation_results(
    hourly_loads_df: pd.DataFrame,
    total_demands_df: pd.DataFrame,
    save_path: Union[str, Path],
    logger,
):
    """
    Saves the simulation results DataFrames to CSV files.

    Args:
        hourly_loads_df: The DataFrame containing the hourly loads.
        total_demands_df: The DataFrame containing the total demands.
        save_path: The directory path to save the files.
        logger: Logger object for logging information.
    """
    save_path = Path(save_path)
    try:
        # Save hourly loads (v_max)
        v_max_path = save_path / "v_max.csv"
        mean_hourly_loads = hourly_loads_df.mean().rename("mean_load")

        # Add a name to the index before saving
        mean_hourly_loads = mean_hourly_loads.reset_index()
        mean_hourly_loads = mean_hourly_loads.rename(columns={"index": "Hour"})

        mean_hourly_loads.to_csv(v_max_path, index=False)
        logger.info(f"Hourly loads (v_max) saved to {v_max_path}")

        # Save total demands
        total_demand_path = save_path / "total_demand.csv"
        mean_total_demand = total_demands_df.mean().iloc[
            0
        ]  # Get the mean of the single column
        with open(total_demand_path, "w") as f:
            f.write(str(mean_total_demand))
        logger.info(f"Total demand saved to {total_demand_path}")

    except Exception as e:
        logger.error(f"Failed to save simulation results to CSV: {e}")
        raise


def plot_and_summarize_results_with_path(
    hourly_loads_df: pd.DataFrame, save_path: Path, logger
):
    """
    Calculates statistics and plots the results of the simulation, saving the plot to a specified path.

    Args:
        hourly_loads_df: A DataFrame containing the hourly loads from all simulations.
        save_path: The directory path to save the plot.
        logger: Logger object for logging information.
    """
    logger.info("--> Plotting Monte Carlo simulation.")

    hourly_stats = (
        hourly_loads_df.describe().loc[["mean", "std", "25%", "50%", "75%"]].transpose()
    )
    hourly_stats.index = hourly_stats.index.str.replace("Hour_", "")
    hourly_stats.index.name = "Hour"

    logger.debug("\n--- Hourly Aggregate Load Statistics ---")
    logger.debug(hourly_stats.to_string())

    mean_hourly_loads_df = hourly_stats["mean"]

    plt.figure(figsize=(12, 6))
    mean_hourly_loads_df.plot(label="Mean Hourly Aggregate Load")
    plt.fill_between(
        hourly_stats.index,
        hourly_loads_df.quantile(0.05),
        hourly_loads_df.quantile(0.95),
        color="gray",
        alpha=0.3,
        label="90% Confidence Interval",
    )
    plt.title("Monte Carlo Simulation of Maximum Hourly EV Charging Load")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Aggregate Load (kWh)")
    plt.xticks(np.arange(0, 24, 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    figures_dir = save_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    file_path = figures_dir / "hourly_charging_load_simulation.png"
    plt.savefig(file_path)
    logger.info(f"----> Plot saved to {file_path}")

    # # Save simulation results to the data directory
    # logger.info("Saving simulation results...")
    # save_simulation_results(mean_hourly_loads_df, file_name='v_max.csv', save_path=save_path, logger=logger)


# def save_simulation_results(df: pd.DataFrame, file_name: str, save_path: Union[str, Path], logger):
#     """
#     Saves the simulation results DataFrame to a CSV file.

#     Args:
#         df: The DataFrame containing the simulation results.
#         save_path: The directory path to save the file.
#         logger: Logger object for logging information.
#     """
#     save_path = Path(save_path)
#     file_path = save_path / file_name
#     try:
#         df.to_csv(file_path, index=False)
#         logger.info(f"Hourly loads saved to {file_path}")
#     except Exception as e:
#         logger.error(f"Failed to save hourly loads to CSV: {e}")
#         raise
