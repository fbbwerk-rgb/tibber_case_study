"""
limit_price_helpers.py

This module contains helper functions that support the main limit price modeling
pipeline defined in `limit_price.py`. The functions are responsible for key tasks
such as data handling, model training, hyperparameter optimization, and performance
visualization.

Key functions include:
- `prepare_and_split_data`: Manages the data loading, target isolation, and
  splitting into training and testing sets for both classification and regression.
- `save_classification_report_table`: Generates a visual, table-based representation
  of a classification report and saves it as a PNG file.
- `report_performance_regressor`: Evaluates a trained regression model using
  metrics like R-squared, MSE, and MAE, and saves the model as a pickle file.
- `plot_regression_results` and `plot_regression_line_plot`: Generate and save
  scatter plots and time-series plots to visualize regression model performance.
- `run_classification_pipeline`: Orchestrates the training and hyperparameter
  optimization (`hyperopt`) for the classification model ensemble.
- `run_regression_pipeline`: Orchestrates the training and hyperparameter
  optimization (`hyperopt`) for the regression model ensemble.

The module also handles logging and manages the suppression of external library
log messages to maintain a clean output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import pickle
from pathlib import Path
from typing import Union, Dict, Any
from data_prep import limit_price_data_prep
from scipy.stats import f
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

### Initialize logger
logger = logging.getLogger(__name__)

# Turn off external logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("pkg_resources").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def prepare_and_split_data(
    data_dir: Path, shuffle: bool = False, final_test: bool = False
):
    """
    Handles all data preparation, splitting, and scaling for both classification
    and regression tasks.

    This function loads the prepared data, isolates the target variables, splits
    the data into training and testing sets, and then scales both the features
    and the regression target variable.

    Args:
        data_dir (Path): The Path object for the directory containing data files.
        shuffle (bool): If True, shuffles the data before splitting. This should
                        typically be False for time-series data to maintain
                        chronological order.
        final_test (bool): If true, returns y2_train instead of y2_test for testing purposes.

    Returns:
        tuple: A tuple containing the scaled training and testing data:
               (X_train_scaled, X_test_scaled, y1_train, y1_test, y2_train_scaled, y2_test, y2_scaler)
    """
    logger.info("Collecting prepared data and scaling it.")
    full_data_df = limit_price_data_prep(data_dir=data_dir)

    logger.info("Isolating target variables and dropping redundant columns.")
    target_1 = full_data_df.pop("sign_spread")
    target_2 = full_data_df.pop("spread_eur")
    full_data_df.drop(columns=["spot_eur", "imb_eur"], inplace=True)
    logger.info(
        f"--> Targets isolated. Final features DataFrame shape: {full_data_df.shape}"
    )

    logger.info("Splitting data into training and testing sets.")
    X_train, X_test, y1_train, y1_test = train_test_split(
        full_data_df, target_1, test_size=0.3, random_state=42, shuffle=shuffle
    )

    y2_train = target_2.loc[y1_train.index]
    y2_test = target_2.loc[y1_test.index]
    logger.info(f"--> Successfully split data into training and testing sets")

    logger.info("Scaling features with StandardScaler.")
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    logger.info("----> Feature scaling complete.")

    logger.info("Scaling regression target (y2) with MinMaxScaler.")
    y2_scaler = MinMaxScaler()
    y2_train_scaled = y2_scaler.fit_transform(y2_train.values.reshape(-1, 1)).flatten()
    logger.info("----> Regression target scaling complete.")

    if final_test:
        return (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_train,
            y2_scaler,
        )
    else:
        return (
            X_train_scaled,
            X_test_scaled,
            y1_train,
            y1_test,
            y2_train_scaled,
            y2_test,
            y2_scaler,
        )


def save_classification_report_table(
    report: Dict[str, Any], output_dir: Path, filename: str, ensemble_members: str
):
    """
    Generates and saves a classification report as a table image.

    Args:
        report (dict): The classification report dictionary from sklearn.metrics.
        output_dir (Path): The directory path to save the image.
        filename (str): The filename for the saved image.
        ensemble_members (str): A string of the names of the models in the ensemble.
    """
    try:
        report_df = pd.DataFrame(report).transpose().round(2)
        report_df = report_df[["precision", "recall", "f1-score", "support"]]
        table_path = output_dir / filename

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.axis("tight")
        table = ax.table(
            cellText=report_df.values,
            colLabels=report_df.columns,
            rowLabels=report_df.index,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title(
            f"Classification Report for Ensemble Model\n({ensemble_members})",
            fontsize=12,
            fontweight="bold",
        )
        plt.savefig(table_path, bbox_inches="tight", dpi=150)
        logger.info(f"----> Classification report table saved to: {table_path}")
    except Exception as e:
        logger.error(f"Error saving classification report table: {e}", exc_info=True)
    finally:
        plt.close(fig)


def report_performance_regressor(
    model: object,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    y_scaler: MinMaxScaler,
    output_dir: Union[str, Path],
    ensemble_members: str,
) -> None:
    """
    Evaluates and reports the performance of a regression model.

    This function performs several key tasks:
    1. Makes predictions on scaled test data and inverse transforms them.
    2. Calculates and logs key regression metrics (R-squared, MSE, MAE).
    3. Generates and saves a scatter plot of true vs. predicted values.
    4. Generates and saves a time-series line plot of true vs. predicted values.
    5. Serializes and saves the trained model using pickle.

    Args:
        model (object): The trained regression model (e.g., a scikit-learn regressor).
        X_test_scaled (pd.DataFrame): The scaled feature test data.
        y_test (pd.Series): The true target values for the test set.
        y_scaler (MinMaxScaler): The fitted scaler object used for target variable.
        output_dir (Union[str, Path]): The directory to save the output artifacts.
        ensemble_members (str): A string of the names of the models in the ensemble.
    """
    logger.info("Starting performance report for regression model...")
    output_path = Path(output_dir)

    try:
        # Make predictions and inverse transform
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        logger.info("Predictions made and inverse transformed.")

        # Calculate and log performance metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"----> Regressor Performance on Test Set:")

        # Calculate F-statistic and p-value
        n = len(y_test)
        k = X_test_scaled.shape[1]  # number of features
        f_statistic = (r2 / k) / ((1 - r2) / (n - k - 1))
        p_value = 1 - f.cdf(f_statistic, k, n - k - 1)

        logger.info(f"----> F-statistic: {f_statistic:.4f}")
        logger.info(f"----> F-statistic p-value: {p_value:.4f}")

        logger.info(f"----> R-squared: {r2:.4f}")
        logger.info(f"----> Mean Squared Error: {mse:.4f}")
        logger.info(f"----> Mean Absolute Error: {mae:.4f}")

        # Generate and save plots
        plot_regression_results(
            y_test, y_pred, output_path, "regression_results.png", ensemble_members
        )
        plot_regression_line_plot(
            y_test, y_pred, output_path, "regression_line_plot.png", ensemble_members
        )

        # Save the trained model
        model_dir = output_path.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filename = "ensemble_reg.pkl"
        # with open(model_dir / model_filename, "wb") as file:
        #     pickle.dump(model, file) # Commented out to avoid saving model again, source of randomness due to hyperopt with low max_evals
        logger.info(f"----> Model saved to: {model_dir / model_filename}")

    except Exception as e:
        logger.error(
            f"An error occurred during regression performance reporting: {e}",
            exc_info=True,
        )
        raise


def plot_regression_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
    filename: str,
    ensemble_members: str,
):
    """
    Generates and saves a scatter plot of true vs. predicted values for regression.

    Args:
        y_true (pd.Series): The true target values.
        y_pred (np.ndarray): The predicted values.
        output_dir (Path): The directory path to save the plot.
        filename (str): The filename for the saved plot (e.g., 'regression_results.png').
        ensemble_members (str): A string of the names of the models in the ensemble.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
            label="Ideal Fit",
        )
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(
            f"True vs. Predicted Values for Ensemble Regression Model\n({ensemble_members})"
        )
        plt.legend()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        logger.info(f"----> Regression scatter plot saved to: {output_dir / filename}")
    except Exception as e:
        logger.error(f"Error plotting regression results: {e}", exc_info=True)
    finally:
        plt.close()


def plot_regression_line_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dir: Path,
    filename: str,
    ensemble_members: str,
):
    """
    Generates and saves a line plot of true vs. predicted values over time.

    Args:
        y_true (pd.Series): The true target values with a datetime index.
        y_pred (np.ndarray): The predicted values.
        output_dir (Path): The directory path to save the plot.
        filename (str): The filename for the saved plot (e.g., 'regression_line_plot.png').
        ensemble_members (str): A string of the names of the models in the ensemble.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true, label="True Values", color="blue")
        plt.plot(y_true.index, y_pred, label="Predicted Values", color="orange")
        plt.title(
            f"True vs Predicted Values for Ensemble Regression Model\n({ensemble_members})"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300)
        logger.info(f"----> Regression line plot saved to: {output_dir / filename}")
    except Exception as e:
        logger.error(f"Error plotting regression line plot: {e}", exc_info=True)
    finally:
        plt.close()


def run_classification_pipeline(X_train_scaled, X_test_scaled, y1_train, y1_test):
    """
    Executes the entire Model 1 classification pipeline, including base model
    training, hyperparameter optimization, and ensemble building.

    Args:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        y1_train (pd.Series): Training labels for the classification task.
        y1_test (pd.Series): Testing labels for the classification task.

    Returns:
        tuple: A tuple containing the trained VotingClassifier ensemble model and a string of the ensemble members.
    """
    logger.info("Starting Model 1: Classifying sign of price spread.")
    lst_clf = [
        ("rf", RandomForestClassifier()),
        ("nb", GaussianNB()),
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier()),
    ]

    logger.info("--> Defining and fitting classifiers for Model 1 (Classifier)")
    for tpl in lst_clf:
        clf_base = tpl[1]
        clf_name = tpl[0]
        clf_base.fit(X_train_scaled, y1_train)
        clf_sc = np.round(clf_base.score(X_test_scaled, y1_test), 2)
        logger.info(f"----> Classifier {clf_name} → Accuracy score = {clf_sc}")

    def objective_classification(params):
        classifier_type = params["classifier"]
        if classifier_type == "lr":
            clf = LogisticRegression(**params["lr"])
        elif classifier_type == "dt":
            clf = DecisionTreeClassifier(**params["dt"], class_weight="balanced")
        elif classifier_type == "rf":
            clf = RandomForestClassifier(**params["rf"], class_weight="balanced")
        elif classifier_type == "nb":
            clf = GaussianNB(**params["nb"])
        score = -np.mean(
            cross_val_score(
                clf, X_train_scaled, y1_train, cv=5, n_jobs=-1, scoring="f1_macro"
            )
        )
        return {"loss": score, "status": STATUS_OK}

    space_classification = {
        "classifier": hp.choice("classifier", ["lr", "dt", "rf", "nb"]),
        "lr": {"penalty": hp.choice("penalty", ["l2"]), "C": hp.loguniform("C", -5, 5)},
        "dt": {
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": hp.choice("max_depth_dt", range(1, 11)),
            "min_samples_split": hp.choice("min_samples_split_dt", range(2, 11)),
        },
        "rf": {
            "n_estimators": hp.choice("n_estimators", range(100, 1000, 50)),
            "max_depth": hp.choice("max_depth_rf", range(1, 11)),
            "min_samples_split": hp.choice("min_samples_split_rf", range(2, 11)),
        },
        "nb": {"var_smoothing": hp.loguniform("var_smoothing", -9, 0)},
    }

    logger.info("--> Starting hyperparameter optimization for Model 1 (Classifier)")
    trials = Trials()
    best = fmin(
        fn=objective_classification,
        space=space_classification,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials,
    )

    logger.info("----> Setting optimized hyperparameters for Model 1 (Classifier)")
    best_params = space_eval(space_classification, best)
    lst_clf_optimized = []

    # Check if 'classifier' key exists in best_params
    if "classifier" in best_params:
        for classifier_name, _ in lst_clf:
            if classifier_name == "rf" and "rf" in best_params:
                optimized_classifier = RandomForestClassifier(
                    **best_params["rf"], class_weight="balanced"
                )  # balanced to avoid bias from imbalanced data (more positive spreads makes it always guess positive as it would be right more often)
            elif classifier_name == "lr" and "lr" in best_params:
                optimized_classifier = LogisticRegression(**best_params["lr"])
            elif classifier_name == "dt" and "dt" in best_params:
                optimized_classifier = DecisionTreeClassifier(
                    **best_params["dt"], class_weight="balanced"
                )
            elif classifier_name == "nb" and "nb" in best_params:
                optimized_classifier = GaussianNB(**best_params["nb"])
            else:
                logger.warning(
                    f"Skipping {classifier_name} due to missing optimized parameters."
                )
                continue
            lst_clf_optimized.append((classifier_name, optimized_classifier))
    else:
        logger.warning(
            "No 'classifier' key found in best_params. Using default models."
        )
        # Ensure default models also have class_weight
        lst_clf_optimized = [
            ("rf", RandomForestClassifier(class_weight="balanced")),
            ("nb", GaussianNB()),
            ("lr", LogisticRegression()),
            ("dt", DecisionTreeClassifier(class_weight="balanced")),
        ]

    ensemble_members = ", ".join([name for name, _ in lst_clf_optimized])

    logger.info("----> Building an ensemble model for Model 1 (Classifier)")
    clf_blend_1 = VotingClassifier(estimators=lst_clf_optimized, voting="soft")
    clf_blend_1.fit(X_train_scaled, y1_train)
    return clf_blend_1, ensemble_members


def run_regression_pipeline(
    X_train_scaled, X_test_scaled, y2_train_scaled, y2_test, y_scaler
):
    """
    Executes the entire Model 2 regression pipeline. It uses the pre-scaled
    training target and the unscaled testing target.

    Args:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        y2_train_scaled (np.ndarray): Scaled training labels for the regression task.
        y2_test (pd.Series): Unscaled testing labels for the regression task.
        y_scaler (MinMaxScaler): The fitted scaler object for the regression target.

    Returns:
        tuple: A tuple containing the trained VotingRegressor ensemble model and a string of the ensemble members.
    """
    logger.info("Starting Model 2: Regressing on spread_eur.")

    lst_reg = [
        ("ols", LinearRegression()),
        ("ridge", Ridge()),
        ("lasso", Lasso()),
        ("rf_reg", RandomForestRegressor()),
        ("knn", KNeighborsRegressor()),
        ("bayes_ridge", BayesianRidge()),
    ]

    logger.info("--> Defining and fitting regression models for Model 2 (Regressor)")
    for tpl in lst_reg:
        reg_base = tpl[1]
        reg_name = tpl[0]
        reg_base.fit(X_train_scaled, y2_train_scaled)

        y2_pred_scaled = reg_base.predict(X_test_scaled)
        y2_pred_unscaled = y_scaler.inverse_transform(
            y2_pred_scaled.reshape(-1, 1)
        ).flatten()
        r2_sc = np.round(r2_score(y2_test, y2_pred_unscaled), 2)
        logger.info(f"----> Regressor {reg_name} → R-squared score = {r2_sc}")

    def objective_regression(params):
        regressor_type = params["regressor"]
        if regressor_type == "ols":
            reg = LinearRegression()
        elif regressor_type == "ridge":
            reg = Ridge(**params["ridge"])
        elif regressor_type == "lasso":
            reg = Lasso(**params["lasso"])
        elif regressor_type == "rf_reg":
            reg = RandomForestRegressor(**params["rf_reg"])
        elif regressor_type == "knn":
            reg = KNeighborsRegressor(**params["knn"])
        elif regressor_type == "bayes_ridge":
            reg = BayesianRidge(**params["bayes_ridge"])

        score = np.mean(
            cross_val_score(
                reg,
                X_train_scaled,
                y2_train_scaled,
                cv=5,
                n_jobs=-1,
                scoring="neg_mean_absolute_error",
            )
        )
        return {"loss": -score, "status": STATUS_OK}

    space_regression = {
        "regressor": hp.choice(
            "regressor", ["ols", "ridge", "lasso", "rf_reg", "knn", "bayes_ridge"]
        ),
        "ols": {},
        "ridge": {"alpha": hp.loguniform("alpha_ridge", -5, 5)},
        "lasso": {"alpha": hp.loguniform("alpha_lasso", -5, 5)},
        "rf_reg": {
            "n_estimators": hp.choice("n_estimators_rf_reg", range(100, 1000, 50)),
            "max_depth": hp.choice("max_depth_rf_reg", range(1, 11)),
            "min_samples_split": hp.choice("min_samples_split_rf_reg", range(2, 11)),
        },
        "knn": {
            "n_neighbors": hp.choice("n_neighbors_knn", range(2, 21)),
            "weights": hp.choice("weights_knn", ["uniform", "distance"]),
        },
        "bayes_ridge": {
            "alpha_1": hp.loguniform("alpha_1_bayes", -10, 0),
            "lambda_1": hp.loguniform("lambda_1_bayes", -10, 0),
        },
    }

    logger.info("--> Starting hyperparameter optimization for Model 2 (Regressor)")
    trials_reg = Trials()
    best_reg = fmin(
        fn=objective_regression,
        space=space_regression,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials_reg,
    )

    logger.info("----> Setting optimized hyperparameters for Model 2 (Regressor)")
    best_params_reg = space_eval(space_regression, best_reg)
    lst_reg_optimized = []

    if "regressor" in best_params_reg:
        for regressor_name, _ in lst_reg:
            if regressor_name == "ols":
                optimized_reg = LinearRegression()
            elif regressor_name == "ridge" and "ridge" in best_params_reg:
                optimized_reg = Ridge(**best_params_reg["ridge"])
            elif regressor_name == "lasso" and "lasso" in best_params_reg:
                optimized_reg = Lasso(**best_params_reg["lasso"])
            elif regressor_name == "rf_reg" and "rf_reg" in best_params_reg:
                optimized_reg = RandomForestRegressor(**best_params_reg["rf_reg"])
            elif regressor_name == "knn" and "knn" in best_params_reg:
                optimized_reg = KNeighborsRegressor(**best_params_reg["knn"])
            elif regressor_name == "bayes_ridge" and "bayes_ridge" in best_params_reg:
                optimized_reg = BayesianRidge(**best_params_reg["bayes_ridge"])
            else:
                logger.warning(
                    f"Skipping {regressor_name} due to missing optimized parameters."
                )
                continue
            lst_reg_optimized.append((regressor_name, optimized_reg))
    else:
        logger.warning(
            "No 'regressor' key found in best_params_reg. Using default models."
        )
        lst_reg_optimized = lst_reg

    ensemble_members = ", ".join([name for name, _ in lst_reg_optimized])

    logger.info("----> Building a voting regressor ensemble for Model 2 (Regressor)")
    reg_blend_2 = VotingRegressor(estimators=lst_reg_optimized, n_jobs=-1)
    reg_blend_2.fit(X_train_scaled, y2_train_scaled)
    return reg_blend_2, ensemble_members
