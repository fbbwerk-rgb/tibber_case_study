"""auction_optimization.py 

Main script to run the EV charging optimization process.

This script orchestrates the entire workflow:
1. Loads required data for model predictions.
2. Loads pre-trained machine learning models.
3. Loads simulation data for demand and capacity.
4. Prepares the data for the optimization algorithm.
5. Executes the linear programming optimization to find the minimum cost
   charging schedule.
6. Saves the final optimized schedule to a CSV file.

"""

import logging
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main():
    """
    Main function to run the EV charging optimization pipeline.
    """

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)

    # Import custom modules here to avoid circular imports and logger mislabeling
    from auction_optimization_helpers import (
        load_data_limit_price_prediction,
        load_models,
        load_demand_simulation_data,
        prepare_data_for_optimization,
        optimize_charging_schedule,
        save_optimized_schedule,
    )

    data_dir = Path("./src/data")

    try:
        # Step 1: Load data for limit price predictions
        logger.info("Step 1: Loading data for limit price prediction models.")
        (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_test,
            y2_scaler,
            full_data_df,
        ) = load_data_limit_price_prediction(str(data_dir))

        # Step 2: Load limit price models
        logger.info("Step 2: Loading machine learning models.")
        clf_model, reg_model = load_models()
        if clf_model is None or reg_model is None:
            logger.error("Failed to load one or both models. Exiting.")
            return

        # Step 3: Load the simulated data
        logger.info("Step 3: Loading demand simulation data.")
        v_max_df, total_demand = load_demand_simulation_data(data_dir)
        logger.info("Loaded total demand: %.2f kWh", total_demand)

        # Step 4: Prepare data for optimization
        logger.info("Step 4: Preparing data for optimization.")
        output_df = prepare_data_for_optimization(
            clf_model,
            reg_model,
            X_test_scaled,
            y2_test,
            y2_scaler,
            full_data_df,
            v_max_df,
        )
        if output_df is None:
            logger.error("Data preparation failed. Exiting.")
            return

        # Step 5: Run the optimization with the loaded total_demand
        logger.info(
            "Step 5: Running optimization to find the optimal charging schedule."
        )
        optimized_df = optimize_charging_schedule(
            output_df,
            total_energy_demand=total_demand,
            cost_col="expected_cost_eur",
            vmax_col="v_max",
        )

        # Step 6: Save the result
        output_path = Path(data_dir) / "optimized_charging_schedule.csv"
        # output_path = Path('./optimized_charging_schedule.csv')
        save_optimized_schedule(optimized_df, output_path, logger)

    except FileNotFoundError as e:
        logger.error(
            "A required file was not found. "
            "Please ensure all necessary data and model files exist: %s",
            e,
        )
    except Exception as e:
        logger.critical(
            "An unexpected error occurred during the pipeline execution: %s", e
        )


if __name__ == "__main__":
    main()
