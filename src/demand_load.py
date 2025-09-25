"""demand_load.py

EV DEMAND LOAD SIMULATION

This is the main executable script for running the Monte Carlo simulation of electric vehicle (EV) charging demand.

The script orchestrates the entire simulation pipeline by:
- Setting up logging and ensuring reproducibility with fixed random seeds.
- Defining key simulation parameters, such as the number of vehicles and simulation runs.
- Calling the core simulation functions from `demand_load_helpers.py` to generate data and calculate loads.
- Analyzing the results and generating a visual plot of the aggregate demand.

To run the simulation, simply execute this script directly from the command line.
"""

import logging
import random
import numpy as np
from pathlib import Path
import sys
from demand_load_helpers import (
    run_monte_carlo_simulation,
    plot_and_summarize_results_with_path,
    save_simulation_results,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

### Initialize logger
logger = logging.getLogger(__name__)


def main():
    """Main function to run the entire simulation pipeline."""
    # Configure a logger for the script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # import custom modules here to avoid circular imports and logger mislabeling
    from demand_load_helpers import (
        run_monte_carlo_simulation,
        plot_and_summarize_results_with_path,
        save_simulation_results,
    )

    try:
        # Create directories for figures and models
        data_dir = Path("./src/data")
        fig_dir = data_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds for reproducibility
        np.random.seed(9)
        random.seed(42)

        # Define simulation parameters
        num_combinations_per_sim = 1000
        num_simulations = 500

        # Run the simulation
        logger.info("Starting Monte Carlo simulation...")
        hourly_loads_df, total_demands_df = run_monte_carlo_simulation(
            num_simulations, num_combinations_per_sim, logger=logger
        )

        # Analyze and visualize results, saving the plot to the figures directory
        logger.info("Analyzing and plotting results...")
        plot_and_summarize_results_with_path(
            hourly_loads_df, data_dir, logger=logger
        )  # Corrected this line

        # Save both hourly loads (v_max) and total demands
        logger.info("Saving simulation results...")
        # The logger argument is correctly passed as the final positional argument
        save_simulation_results(
            hourly_loads_df, total_demands_df, data_dir, logger=logger
        )

        logger.info("--> Simulation completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"--> File not found error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"--> An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
