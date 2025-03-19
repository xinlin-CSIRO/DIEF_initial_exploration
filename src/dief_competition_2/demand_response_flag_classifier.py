# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
import pathlib

import loguru
import numpy as np
import plotly.express as px
import polars as pl
import sklearn
from sklearn.ensemble import RandomForestClassifier


class DemandResponseFlagClassifier:
    def __init__(
        self,
        target_column: str = "Demand_Response_Flag",
        omit_columns: list[str] = None,
    ) -> None:
        self.target_column = target_column
        self.omit_columns = omit_columns or []

    def split_data(
        self,
        data: pl.DataFrame,
    ) -> list[pl.DataFrame]:
        """
        Split the data into X and y
        """
        x = data.drop([self.target_column] + self.omit_columns)
        y = data[self.target_column]
        return x, y

    def train(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Classify the data
        """
        x, y = self.split_data(data)
        model = RandomForestClassifier(random_state=42)
        model.fit(x.drop(self.omit_columns), y)
        return model

    def test(
        self,
        model: RandomForestClassifier,
        data: pl.DataFrame,
    ) -> np.ndarray:
        """
        Test the model
        """
        x, y = self.split_data(data)
        predictions = model.predict(x.drop(self.omit_columns))
        return pl.DataFrame(x).with_columns(
            [
                pl.Series("Predictions", predictions),
                y,
            ]
        )

    @staticmethod
    def visualise_classifications(
        data: pl.DataFrame,
        output_dir: pathlib.Path,
        dataset_id: str,
        logger: loguru._logger.Logger,
    ) -> None:
        """
        Visualise the predictions
        """
        x_test = data.drop(["Demand_Response_Flag"])
        y_test = data["Demand_Response_Flag"]
        predictions = data["Predictions"]

        # Visualise the predictions
        logger.info(f"{dataset_id} Results:")
        logger.info(f"Accuracy: {sklearn.metrics.accuracy_score(y_test, predictions)}")
        logger.info(
            f"F1: {sklearn.metrics.f1_score(y_test, predictions, average='weighted')}"
        )
        logger.info(
            f"Confusion matrix:\n{sklearn.metrics.confusion_matrix(y_test, predictions)}"
        )
        logger.info(
            f"Classification report:\n{sklearn.metrics.classification_report(y_test, predictions)}"
        )

        results_data = pl.DataFrame(x_test).with_columns(
            [pl.Series("Predictions", predictions), y_test]
        )
        results_data.write_csv(output_dir / "drf_classification_results.csv")

        actual_totals = (
            results_data.filter(pl.col("Predictions") != 0)
            .group_by([pl.col("ts").dt.date()])
            .agg(pl.col("Building_Power_kW").abs().sum())
            .sort(["ts"])
        )
        predicted_totals = (
            results_data.filter(pl.col("Demand_Response_Flag") != 0)
            .group_by([pl.col("ts").dt.date()])
            .agg(pl.sum("Building_Power_kW"))
            .sort(["ts"])
        )

        vis_df = actual_totals.join(
            predicted_totals, on="ts", suffix="_predicted"
        ).rename(
            {
                "Building_Power_kW": "Flexibility kW",
                "Building_Power_kW_predicted": "Predicted Flexibility kW",
            }
        )
        vis_df.write_csv(output_dir / "drf_assignment_power.csv")

        fig = px.bar(
            vis_df,
            x="ts",
            y=["Flexibility kW", "Predicted Flexibility kW"],
            title=f"Actual vs Predicted ({dataset_id})",
            labels={"value": "Power kW", "ts": "Date"},
            barmode="group",
        )
        fig.update_layout(hovermode="x", title_x=0.5)
        fig.update_xaxes(type="category")
        fig.write_html(output_dir / "drf_predictions.html")

        fig = px.scatter(
            vis_df,
            x="Flexibility kW",
            y="Predicted Flexibility kW",
            title=f"Actual vs Predicted ({dataset_id})",
            trendline="ols",
        )
        fig.update_layout(width=750, height=750, title_x=0.5)
        fig.write_html(output_dir / "drf_avp.html")
