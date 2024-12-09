import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shap
import time


class PricePredictor:
    def __init__(
        self, cat_features, output_dir, iterations=1000, learning_rate=0.05, depth=6
    ):
        """
        Initializes the CatBoost Regressor model with default or user-specified parameters.
        """
        self.cat_features = cat_features
        self.output_dir = output_dir
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            cat_features=cat_features,
            loss_function="RMSE",
            verbose=100,
        )
        # self.is_trained = False

    def prepare_data(self, df, target_column, features_columns):
        """
        Apply transformations and split the data into train/test sets.
        """
        self.target_column = target_column
        self.features_columns = features_columns
        df["Log_Price"] = np.log1p(df[target_column])  # Log-transform the target
        X = df[features_columns]
        y = df["Log_Price"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Trains the CatBoost model on the training dataset.
        """
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        self.model.fit(train_pool)
        # self.is_trained = True

    def predict(self, X):
        """
        Makes predictions on the input dataset.
        """
        # if not self.is_trained:
        # raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)

    def calculate_all_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        """
        r2 = round(metrics.r2_score(y_true, y_pred), 4)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mape = round(metrics.mean_absolute_percentage_error(y_true, y_pred), 4)
        smape = round(
            1
            / len(y_true)
            * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))),
            4,
        )
        return {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
        }

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the model on train and test datasets, both transformed and original.
        """
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Predictions
        predicted_log_train = self.predict(X_train)
        predicted_log_test = self.predict(X_test)

        # Reverse logarithmic transformation
        predicted_train_original = np.expm1(predicted_log_train)
        y_train_original = np.expm1(y_train)
        predicted_test_original = np.expm1(predicted_log_test)
        y_test_original = np.expm1(y_test)

        # Metrics calculation
        dataset_names = [
            "Train Set (transformed)",
            "Test set (transformed)",
            "Train set (original values)",
            "Test set (original values)",
        ]

        dataset_values = [
            (y_train, predicted_log_train),
            (y_test, predicted_log_test),
            (y_train_original, predicted_train_original),
            (y_test_original, predicted_test_original),
        ]

        column_names = ["dataset", "r2", "mse", "rmse", "mae", "mape", "smape"]
        df_metrics = pd.DataFrame(columns=column_names)

        for name, values in zip(dataset_names, dataset_values):
            # Extract actual and predicted values
            y_true, y_pred = values
            # Calculate metrics
            metrics_row = self.calculate_all_metrics(y_true, y_pred)
            # Add the dataset name to the row
            metrics_row["dataset"] = name
            # Append the row to the DataFrame
            df_metrics = pd.concat(
                [df_metrics, pd.DataFrame([metrics_row])], ignore_index=True
            )

        metrics_path = os.path.join(output_dir, "model_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        print("Evaluation Metrics:")
        print(df_metrics)
        return df_metrics

    def visualize_results(self, X_test):
        """
        Generate and save SHAP summary plot and feature importance plot.
        """
        explainer = shap.TreeExplainer(
            self.model, feature_perturbation="tree_path_dependent"
        )
        shap_values = explainer(X_test)

        # SHAP summary plot
        shap_path = os.path.join(output_dir, "shap_summary_plot.png")
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()

        # Feature importance plot
        importances = self.model.feature_importances_
        features_importance_df = pd.DataFrame(
            {"Feature": X_test.columns, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)
        print(features_importance_df)
        features_path = os.path.join(code_dir, "../output_data/features_importance.csv")
        features_importance_df.to_csv(features_path, index=False)
        features_plot_path = os.path.join(output_dir, "features_plot.png")
        plt.figure(figsize=(10, 6))
        plt.barh(
            features_importance_df["Feature"],
            features_importance_df["Importance"],
            color="skyblue",
        )
        plt.gca().invert_yaxis()
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(features_plot_path, bbox_inches="tight")
        plt.close()

    def run_pipeline(self, df, target_column):
        """
        Complete pipeline from data preparation to evaluation and visualization.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, target_column, features_columns
        )
        self.train_model(X_train, y_train)
        metrics_df = self.evaluate_model(X_train, X_test, y_train, y_test)
        self.visualize_results(X_test)
        return metrics_df


if __name__ == "__main__":
    start_time = time.time()
    # Load data
    code_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
    data_path = os.path.join(code_dir, "../output_data/data_for_model.csv")
    df = pd.read_csv(data_path)

    # Specifying input characteristics for the model
    cat_features = ["Province", "State of the Building", "Subtype of Property_Grouped"]
    output_dir = os.path.join(code_dir, "../output_data/")
    target_column = "Price"
    features_columns = [
        "Livable Space (m2)",
        "Avg_Taxable_Income",
        "Province",
        "State of the Building",
        "Subtype of Property_Grouped",
    ]

    # Instantiate the predictor
    predictor = PricePredictor(cat_features, output_dir)

    # Run a pipeline
    predictor.run_pipeline(df, target_column="Price")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for modeling: {total_time}")
