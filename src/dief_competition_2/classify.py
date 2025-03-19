# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
import argparse
import pathlib
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import polars as pl
import shap
import tpot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dief_competition_2.baseline_regressor import BaselineRegressor
from dief_competition_2.demand_response_flag_classifier import (
    DemandResponseFlagClassifier,
)
from dief_competition_2.init_utils.config import get_logger


class PredictionPipeline:
    def __init__(self, dataset_id: str, output_dir: str) -> None:
        self.dataset_id = dataset_id
        self.project_dir = (pathlib.Path(__file__).parent / ".." / "..").resolve()

        if output_dir is None:
            output_dir = f"output/{dataset_id}"
        self.output_dir = self.project_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            "example", log_dir=self.output_dir / "logs", file_mode="w"
        )

        self.omit_columns = []

        self.data_path = (
            self.project_dir
            / "data"
            / dataset_id
            / f"{dataset_id}_Dataset_subhourly.csv"
        )
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.daily_data_path = (
            self.project_dir / "data" / dataset_id / f"{dataset_id}_Dataset_daily.csv"
        )
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.daily_data_path}")


    def run(self) -> None:
        data = self.get_data()
        training_data, test_data = self.get_train_test_split(data)

        classifier = DemandResponseFlagClassifier()
        classifier_model = classifier.train(training_data)
        classification_predictions = classifier.test(classifier_model, test_data)
        classifier.visualise_classifications(
            classification_predictions,
            self.output_dir,
            self.dataset_id,
            self.logger,
        )

        regressor = BaselineRegressor()
        baseline_model_file = self.output_dir / "baseline_model.pkl"
        baseline_model = regressor.get_or_create_baseline_model(
            training_data,
            baseline_model_file,
        )
        baseline_model = regressor.train(training_data, baseline_model)
        baseline_predictions = regressor.test(baseline_model, test_data)
        regressor.visualise_regressions(
            baseline_predictions,
            self.output_dir,
            self.dataset_id,
            self.logger,
        )

    def get_baseline_model(
        self, regressor: "BaselineRegressor", data: pl.DataFrame
    ) -> tpot.TPOTRegressor:
        baseline_model_file = self.output_dir / "baseline_model.pkl"
        if baseline_model_file.exists():
            with open(baseline_model_file, "rb") as fp:
                return pickle.load(fp)
        else:
            regressor.train(data)

    @staticmethod
    def add_lagged_columns(
        data: pl.DataFrame,
        lag_period: str,
        lagged_columns: list[str],
    ) -> pl.DataFrame:
        lagged_data = (
            data.upsample("ts", every="15m", maintain_order=True)
            .with_columns((pl.col("ts") + pl.duration(minutes=15)).alias("ts"))
            .sort("ts")
            .rolling("ts", period=lag_period)
            .agg([pl.col(col) for col in lagged_columns])
        )
        for col in lagged_columns:
            lagged_data = lagged_data.with_columns(
                pl.col(col).list.to_struct(
                    n_field_strategy="max_width",
                    fields=lambda ii: f"{col}+{(ii + 1) * 15}m",
                )
            ).unnest(col)
        return data.join(lagged_data, on="ts", how="left").sort("ts")

    def get_data(self) -> pl.DataFrame:
        """
        Get the data from the dataset_id
        """
        # Load the data from the dataset_id
        data = pl.read_csv(self.data_path)
        self.logger.info(f"Data description:\n{data.describe()}")
        data = (
            data.with_row_index()
            .with_columns(
                [
                    pl.col("Timestamp_unix").diff().alias("diff"),
                    pl.col("Timestamp_Local").str.to_datetime(),
                    (pl.col("index") * 900),
                ]
            )
            .with_columns(
                pl.from_epoch("index", time_unit="s").alias("ts"),
            )
            .drop(["index", "Timestamp_unix", "Timestamp_Local", "diff"])
        )
        return self.add_lagged_columns(
            data,
            "1h",
            [
                "Building_Power_kW",
                "Dry_Bulb_Temperature_C",
                "Global_Horizontal_Radiation_W/m2",
            ],
        )

    def get_daily_data(
        self,
    ) -> pl.DataFrame:
        """
        Get the daily data
        """
        return pl.read_csv(self.daily_data_path).with_columns(
            pl.col("Timestamp_Local").str.to_date()
        )

    def get_train_test_split(
        self, data: pl.DataFrame, train_fraction=0.8
    ) -> list[np.ndarray]:
        """
        Split the data into X and y
        """
        training_data, test_data = train_test_split(
            data,
            train_size=train_fraction,
            random_state=42,
            stratify=data["Demand_Response_Flag"],
        )
        return training_data, test_data

    def explain_drf_classification(
        self,
        model: RandomForestClassifier,
        x_train: pl.DataFrame,
    ) -> None:
        explainer = shap.Explainer(model, x_train.to_numpy())
        predictions = model.predict(x_train.to_numpy())

        if "ts" in x_train.columns:
            x_train = x_train.sort("ts").with_columns(pl.col("ts").dt.epoch())
        shap_values = explainer(x_train.to_pandas(), check_additivity=False)
        expected_value = predictions.mean()
        sv_data = pl.DataFrame(shap_values.values, schema=x_train.columns).with_columns(
            [
                pl.from_epoch(x_train["ts"], time_unit="us").alias("timestamp"),
                pl.sum_horizontal(pl.all()).alias("predicted_value"),
            ]
        )

        updatemenus = [
            dict(
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=False,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=500, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                direction="left",
                pad={"l": -10, "r": 30, "t": 87},
                showactive=False,
                type="buttons",
                x=0,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Datetime: ",
                "visible": True,
                "xanchor": "center",
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }

        frames = []

        for date_index, date_group in sv_data.group_by([pl.col("ts").dt.date()]):
            for row_index, row in enumerate(date_group.iter_rows(named=True)):
                timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                row_data = sv_data.drop("timestamp").to_numpy()[row_index]
                predicted_value = row["predicted_value"]
                fig = go.Figure(
                    data=[
                        go.Waterfall(
                            name=timestamp,
                            orientation="h",
                            base=expected_value,
                            x=row_data,
                            y=sv_data.drop("timestamp").columns,
                            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in row_data],
                        )
                    ]
                )
                fig.add_vline(
                    x=expected_value,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text=f"E[f(X)] = {expected_value:.3f}",
                    annotation_position="bottom",
                    annotation_yshift=-15,
                    layer="below",
                )
                fig.add_vline(
                    x=predicted_value,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text=f"f(x) = {predicted_value:.3f}",
                    annotation_position="top",
                    annotation_yshift=20,
                )
                frames.append(
                    go.Frame(data=fig.data, name=timestamp, layout=fig.layout)
                )
                slider_step = {
                    "args": [
                        [timestamp],
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": timestamp,
                    "method": "animate",
                }
                sliders_dict["steps"].append(slider_step)

            layout = go.Layout(
                xaxis=dict(title="SHAP value per interval"),
                yaxis=dict(title="Feature"),
                hovermode="y",
                updatemenus=updatemenus,
                sliders=[sliders_dict],
            )

            fig = go.Figure(data=frames[0]["data"], frames=frames, layout=layout)
            fig.write_html(self.output_dir / f"{date_index}_shap_waterfall.html")


def parse_arguments(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a model")
    parser.add_argument(
        "--dataset_id", type=str, default="Mascot_15", help="The dataset id"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="The output directory"
    )
    return parser.parse_args(args)


def main(args: list[str] = sys.argv) -> None:
    cfg = parse_arguments(args)
    pipeline = PredictionPipeline(cfg.dataset_id, cfg.output_dir)
    pipeline.run()


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / ".." / ".." / "data"
    for file in tqdm(
        [x for x in data_dir.glob("*") if x.is_dir()], desc="Processing datasets"
    ):
        main(["--dataset_id", file.name, "--output_dir", f"output/lagged/{file.name}"])
