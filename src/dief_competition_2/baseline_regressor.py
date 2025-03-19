# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
import pathlib
import pickle

import loguru
import numpy as np
import plotly.express as px
import polars as pl
import sklearn
import sklearn.pipeline
import tpot
from dask.distributed import LocalCluster
from sklearn.pipeline import Pipeline


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
        if baseline_model_file.exists():
            with open(baseline_model_file, "rb") as f:
                return pickle.load(f)
        else:
            model = self.hyperparameter_train(data)
            with open(baseline_model_file, "wb") as f:
                pickle.dump(model.fitted_pipeline_, f)
            return model.fitted_pipeline_

    def hyperparameter_train(self, data: pl.DataFrame) -> tpot.TPOTRegressor:
        x, y = self.split_data(data)
        cluster = LocalCluster(n_workers=12)
        with cluster.get_client() as client:
            model = tpot.TPOTRegressor(
                random_state=42,
                verbose=3,
                client=client,
            )
            model.fit(x.drop(self.omit_columns), y.to_numpy())
            client.close()
        return model

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
        predictions = model.predict(x.drop(self.omit_columns))
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
        dataset_id: str,
        logger: loguru._logger.Logger,
    ) -> None:
        """
        Visualise the predictions
        """
        x_test = data.drop(["Building_Power_kW"])
        y_test = data["Building_Power_kW"]
        predictions = data["Predictions"]

        # Visualise the predictions
        logger.info(f"{dataset_id} Results:")
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
            title=f"Predictions vs Actual ({dataset_id})",
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
            title=f"Baseline Prediction ({dataset_id})",
            labels={"value": "Power kW", "ts": "Timestamp"},
        )
        fig.update_layout(hovermode="x")
        fig.write_html(output_dir / "baseline_predictions.html")
