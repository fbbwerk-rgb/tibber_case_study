"""
limit_price.py

This is the main script that orchestrates the entire machine learning pipeline for
predicting limit price behavior. It coordinates key tasks by utilizing helper
functions from the `limit_price_helpers` and `data_prep` modules.

The script's primary responsibilities include:
- Setting up the logging environment and creating necessary output directories.
- Loading and preparing the raw data using `limit_price_data_prep`.
- Generating and saving a correlation matrix of the features to provide a
  visual overview of the data structure, for later evaluation.
- Calling `prepare_and_split_data` to handle data splitting and scaling for
  downstream models.
- Executing the classification pipeline (`run_classification_pipeline`) to train
  an ensemble model for predicting the sign of the price spread.
- Evaluating the classification model's performance, logging the results, and
  saving the model artifacts.
- Executing the regression pipeline (`run_regression_pipeline`) to train an
  ensemble model for predicting the magnitude of the price spread.
- Evaluating the regression model's performance, logging the results, and
  saving the model artifacts and performance plots.

This script serves as the top-level entry point for the limit price modeling
project, ensuring that all steps—from data ingestion to model evaluation—are
executed in a logical and reproducible sequence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Main execution block
def main():
    # Initialize logger for the main script
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Turn off external logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("hyperopt").setLevel(logging.ERROR)
    logging.getLogger("pkg_resources").setLevel(logging.ERROR)

    # Import custom modules here to avoid circular imports and logger mislabeling
    from data_prep import limit_price_data_prep
    from limit_price_helpers import (
        prepare_and_split_data,
        run_classification_pipeline,
        run_regression_pipeline,
        report_performance_regressor,
        save_classification_report_table,
    )

    try:
        # Create directories for figures and models
        data_dir = Path("./src/data")
        fig_dir = data_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        mod_dir = data_dir / "models"
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Start the main execution
        logger.info("Starting the full limit price modeling pipeline.")

        # Load and prepare the data
        full_data_df = limit_price_data_prep(data_dir=data_dir)

        # Generate correlation matrix
        logger.info("Generating and saving the correlation matrix.")
        correlation_matrix = full_data_df.corr(
            method="spearman", numeric_only=True
        )  # spearman to capture nonlinear correlations
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="coolwarm",
            vmax=1,
            vmin=-1,
            center=0,
            annot=False,
            fmt=".2f",
        )
        plt.title("Spearman correlation matrix of features for pricing models")
        plt.savefig(fig_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Splitting the data for modeling
        logger.info("Preparing and splitting data for modeling.")
        (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_test,
            y2_scaler,
        ) = prepare_and_split_data(data_dir=data_dir, shuffle=False)

        # Model 1 (ensemble classifier) specification
        logger.info("Running the classification pipeline (Model 1).")
        # Unpack the tuple returned by the function
        clf_blend_1, clf_ensemble_members = run_classification_pipeline(
            X_train_scaled, X_test_scaled, y1_train, y1_test
        )

        # Model 1 (ensemble classifier) evaluation
        logger.info("Evaluating and saving Model 1 (Classifier) performance.")
        y1_pred_ensemble = clf_blend_1.predict(X_test_scaled)
        report = classification_report(y1_test, y1_pred_ensemble, output_dict=True)
        logger.info(
            f"----> Ensemble Model 1 (Classifier) performance on test set: Accuracy of {report['accuracy']:.2f}"
        )

        # # Save performance of Model 1 and the model itself
        # save_classification_report_table(
        #     report=report,
        #     output_dir=fig_dir,
        #     filename="classification_report_ensemble.png",
        #     ensemble_members=clf_ensemble_members,
        # ) # Commented out to avoid saving model again, source of randomness due to hyperopt with low max_evals
        with open(mod_dir / "ensemble_clf.pkl", "wb") as file:
            pickle.dump(clf_blend_1, file)
        logger.info(f"----> Model 1 saved to: {mod_dir / 'ensemble_clf.pkl'}")

        # Model 2 (ensemble regression) specification
        logger.info("Running the regression pipeline (Model 2).")
        reg_blend_2, reg_ensemble_members = run_regression_pipeline(
            X_train_scaled, X_test_scaled, y2_train_scaled, y2_test, y_scaler=y2_scaler
        )  # Unpack the tuple returned by the function

        # Model 2 (ensemble regression) evaluation
        logger.info("Evaluating and saving Model 2 (Regressor) performance.")
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=y2_test.index)

        # Save performance of Model 2 and the model itself
        report_performance_regressor(
            reg_blend_2,
            X_test_scaled_df,
            y2_test,
            y2_scaler,
            fig_dir,
            ensemble_members=reg_ensemble_members,
        )

    except Exception as e:
        logger.error(
            f"An error occurred in the main execution block: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
