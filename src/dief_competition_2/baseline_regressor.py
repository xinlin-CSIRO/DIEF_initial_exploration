# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
import pathlib
import pickle
from tpot import TPOTRegressor
from dask.distributed import LocalCluster
import loguru
import numpy as np
import plotly.express as px
import polars as pl
import sklearn
import sklearn.pipeline
import tpot
from dask.distributed import LocalCluster
from sklearn.pipeline import Pipeline
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter


class BaselineRegressor:
    def __init__(
        self,
        target_column: str = "Building_Power_kW",
        omit_columns: list[str] = None,
    ) -> None:
        self.target_column = target_column
        self.omit_columns = omit_columns or []

    def split_data(
        self,
        data: pl.DataFrame,
        filter_dr_events: bool = True,
    ) -> list[pl.DataFrame]:
        """
        Split the data into X and y
        """
        if filter_dr_events:
            data = data.filter(pl.col("Demand_Response_Flag") == 0)
        x = data.drop([self.target_column] + self.omit_columns)
        y = data[self.target_column]
        return x, y

    def get_or_create_baseline_model(
            self, data: pl.DataFrame, baseline_model_file: pathlib.Path
    ) -> sklearn.pipeline.Pipeline:
        # 🔥 Force delete cached model so it retrains
        if baseline_model_file.exists():
            print(f"🗑️ Removing cached model to force retraining: {baseline_model_file}")
            baseline_model_file.unlink()

        print("🚀 Training new TPOT model...")
        model = self.hyperparameter_train(data)

        # Save the new model
        with open(baseline_model_file, "wb") as f:
            pickle.dump(model.fitted_pipeline_, f)

        # ✅ Print all pipelines TPOT tried
        print("\n✅ TPOT evaluated the following pipelines:\n")
        for i, (pipeline_str, score_dict) in enumerate(model.evaluated_individuals_.items(), 1):
            score = score_dict['internal_cv_score']
            print(f"{i:03d}: Score={score:.4f}")
            print(f"Pipeline: {pipeline_str}\n")
        return model.fitted_pipeline_


    def hyperparameter_train(self, data: pl.DataFrame) -> tpot.TPOTRegressor:
        x, y = self.split_data(data)

        # Convert to pandas for sklearn compatibility
        x = x.drop(self.omit_columns).to_pandas()
        y = y.to_pandas()

        # 🔥 Drop datetime columns from X
        datetime_cols = x.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        x = x.drop(columns=datetime_cols)

        # Handle NaNs
        x = x.dropna()
        y = y.loc[x.index]  # align y with cleaned x

        model = TPOTRegressor(
            generations=5,  # fewer generations
            population_size=20,  # fewer candidates per generation
            max_time_mins=10,  # hard time cap in minutes
            verbosity=2,  # show progress
            n_jobs=-1,  # use all available CPUs
            random_state=42
        )
        cluster = LocalCluster(n_workers=12)
        with cluster.get_client() as client:
            model.fit(x, y.to_numpy())
            client.close()
        print("\n✅ TPOT evaluated the following pipelines:\n")
        for i, (pipeline_str, score_dict) in enumerate(model.evaluated_individuals_.items(), 1):
            score = score_dict.get('internal_cv_score', 'N/A')
            if isinstance(score, float):
                print(f"{i:03d}: CV Score={score:.4f}")
            else:
                print(f"{i:03d}: CV Score={score}")
            print(f"Pipeline: {pipeline_str}\n")

        return model

        # cluster = LocalCluster(n_workers=12)
        # with cluster.get_client() as client:
        #     model = tpot.TPOTRegressor(
        #         random_state=42,
        #     )
        #     model.fit(x, y.to_numpy())
        #     client.close()
        #
        # return model

    def train(
        self,
        data: pl.DataFrame,
        model: Pipeline,
    ) -> tpot.TPOTRegressor:
        """
        Classify the data
        """
        x, y = self.split_data(data)
        model.fit(x.drop(self.omit_columns), y.to_numpy())
        return model

    def test(
        self,
        model: tpot.TPOTRegressor,
        data: pl.DataFrame,
    ) -> np.ndarray:
        """
        Test the model
        """
        x, y = self.split_data(data, filter_dr_events=False)
        # predictions = model.predict(x.drop(self.omit_columns))
        columns_to_drop = [col for col in self.omit_columns if col in x.columns]
        if "dataset_id" in x.columns:
            columns_to_drop.append("dataset_id")

        x = x.drop(columns_to_drop)

        predictions = model.predict(x)


        return pl.DataFrame(x).with_columns(
            [
                pl.Series("Predictions", predictions),
                y,
            ]
        )

    @staticmethod
    def visualise_regressions(
        data: pl.DataFrame,
        output_dir: pathlib.Path,
        # dataset_id: str,
        logger: loguru._logger.Logger,
    ) -> None:
        """
        Visualise the predictions
        """
        x_test = data.drop(["Building_Power_kW"])
        y_test = data["Building_Power_kW"]
        predictions = data["Predictions"]

        # Visualise the predictions
        # logger.info(f"{dataset_id} Results:")
        logger.info(
            f"Mean squared error: {sklearn.metrics.mean_squared_error(y_test, predictions)}"
        )
        logger.info(
            f"Mean absolute error: {sklearn.metrics.mean_absolute_error(y_test, predictions)}"
        )
        logger.info(f"R2 score: {sklearn.metrics.r2_score(y_test, predictions)}")

        results_data = pl.DataFrame(x_test).with_columns(
            [pl.Series("Predictions", predictions), y_test]
        )
        results_data.write_csv(output_dir / "baseline_results.csv")

        min, max = results_data.select(
            min=pl.min_horizontal("Building_Power_kW", "Predictions").min(),
            max=pl.max_horizontal("Building_Power_kW", "Predictions").max(),
        ).row(0)
        dt = 0.1 * (max - min)
        min = min - dt
        max = max + dt

        # min = results_data
        fig = px.scatter(
            results_data,
            x="Predictions",
            y=y_test,
            # title=f"Predictions vs Actual ({dataset_id})",
            trendline="ols",
            opacity=0.5,
            range_x=[min, max],
            range_y=[min, max]
        )
        fig.update_layout(width=750, height=750, title_x=0.5)
        fig.write_html(output_dir / "baseline_avp.html")

        fig = px.line(
            data.sort("ts"),
            x="ts",
            y=["Building_Power_kW", "Predictions"],
            # title=f"Baseline Prediction ({dataset_id})",
            labels={"value": "Power kW", "ts": "Timestamp"},
        )
        fig.update_layout(hovermode="x")
        fig.write_html(output_dir / "baseline_predictions.html")
