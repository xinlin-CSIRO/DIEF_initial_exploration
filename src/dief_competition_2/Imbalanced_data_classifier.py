# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2025.
import argparse
import pathlib
import pickle
import sys
import loguru
import numpy as np
import plotly.graph_objects as go
import polars as pl
import shap
import tpot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loguru import logger
from dief_competition_2.baseline_regressor import BaselineRegressor
from dief_competition_2.demand_response_flag_classifier import (
    DemandResponseFlagClassifier,
)
from dief_competition_2.init_utils.config import get_logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
)
import numpy as np
import sys
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from imblearn.metrics import geometric_mean_score
def undersample(X, y, timesteps=None, features=None):
    # print("🟡 Original class distribution:")
    orig_counts = Counter(y)
    for label, count in orig_counts.items():
        print(f"  Class {label}: {count}")

    X_np = X.to_numpy() if hasattr(X, "to_numpy") else X  # Handle DataFrame or ndarray

    if X_np.ndim == 3:
        n_samples, n_timesteps, n_features = X_np.shape
        timesteps = timesteps or n_timesteps
        features = features or n_features
        X_flat = X_np.reshape((n_samples, -1))
    elif X_np.ndim == 2:
        n_samples, n_features = X_np.shape
        timesteps = 1
        features = n_features
        X_flat = X_np
    else:
        raise ValueError("Unsupported shape for X")

    # Resample
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_flat, y)

    # Reshape back
    try:
        X_res = X_res.reshape((-1, timesteps, features))
    except ValueError:
        print(f"❌ Cannot reshape X_res of shape {X_res.shape} to ({-1}, {timesteps}, {features})")
        raise

    print("\n🟢 Resampled class distribution:")
    resampled_counts = Counter(y_res)
    for label, count in resampled_counts.items():
        print(f"  Class {label}: {count}")

    print("\n📊 Imbalance Rate (Original):")
    total = sum(orig_counts.values())
    for label in sorted(orig_counts):
        imbalance = orig_counts[label] / total
        print(f"  Class {label}: {imbalance:.2%}")

    print("\n📈 Resampling Strategy:")
    if len(set(orig_counts.values())) > 1:
        print("  ➤ Under-sampling applied.")
    else:
        print("  ➤ No sampling needed (already balanced).")

    return X_res, y_res
class DNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First dense layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second dense layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply first layer + ReLU
        x = self.relu(self.fc2(x))  # Apply second layer + ReLU
        x = self.fc3(x)  # Output layer
        return x
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        # x: [batch_size, seq_len, input_size]
        lstm_out, (hn, cn) = self.lstm(x)        # hn: [1, batch_size, hidden_size]
        hn = hn.squeeze(0)                       # hn: [batch_size, hidden_size]
        output = self.fc(hn)                     # output: [batch_size, num_classes]
        return output
class Data_Prep_Pipeline:
    def __init__(self, dataset_id: str, output_dir: str) -> None:
        self.dataset_id = dataset_id
        self.project_dir = (pathlib.Path(__file__).parent / ".." / "..").resolve()

        if output_dir is None:
            output_dir = f"output/{dataset_id}"
        self.output_dir = self.project_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size =32
        self.epoches=10
        self.lr=0.0001
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
        self.logger.info(f"Data description:\n{data.describe()}\n")
        data = (
            data.with_row_index()
            .with_columns(
                [
                    # pl.col("Timestamp_unix").diff().alias("diff"),
                    pl.col("Timestamp_Local").str.to_datetime(),
                    (pl.col("index") * 900),
                ]
            )
            .with_columns(
                pl.from_epoch("index", time_unit="s").alias("ts"),
            )
            # .drop(["index", "Timestamp_unix", "Timestamp_Local", "diff"])
            .drop(["index", "Timestamp_Local"])
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

        training_data = training_data.to_pandas()
        test_data = test_data.to_pandas()
        # Define features
        exclude_cols = ["Demand_Response_Flag", "ts"]
        feature_cols = [col for col in training_data.columns if col not in exclude_cols]

        # Extract features and labels
        train_X = training_data[feature_cols]
        train_Y = training_data["Demand_Response_Flag"]
        test_X = test_data[feature_cols]
        test_Y = test_data["Demand_Response_Flag"]

        # Drop rows with NaNs (keep label alignment)
        train_X = train_X.dropna()
        train_Y = train_Y.loc[train_X.index]
        test_X = test_X.dropna()
        test_Y = test_Y.loc[test_X.index]

        # Remap labels to 0, 1, 2
        label_map = {-1: 0, 0: 1, 1: 2}
        train_Y = train_Y.replace(label_map)
        test_Y = test_Y.replace(label_map)
        scaler = StandardScaler()
        train_X[feature_cols] = scaler.fit_transform(train_X[feature_cols])
        test_X[feature_cols] = scaler.transform(test_X[feature_cols])

        return train_X, train_Y, test_X,test_Y

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
def metrics (all_labels: list, all_preds: list):
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    gmean = geometric_mean_score(all_labels, all_preds, average='macro')

    return [acc, precision_macro, recall_macro, f1_macro, mcc, gmean]
class Model_training_class:
    def __init__(self) -> None:
        self.hidden_size =32
        self.epoches=10
        self.lr=0.0001

    def lstm_training (self, train_X, train_Y, trained_on_id):
        # Scale input
        total_features = train_X.shape[1]
        # Final NaN check (just in case)
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        train_X_np = train_X.to_numpy().reshape(-1, 1, total_features)
        # Wrap in PyTorch Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        input_size = train_X.shape[1]
        num_classes = 3
        model = LSTMClassifier(input_size, self.hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_LSTM.pth")
    def lstm_testing (self, test_X,test_Y,trained_on_id, test_on_id):
        # Scale input
        total_features = test_X.shape[1]
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"
        # Reshape to [batch, seq_len=1, features]

        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)

        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))

        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = test_X.shape[1]

        num_classes = 3
        model = LSTMClassifier(input_size, self.hidden_size, num_classes)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_LSTM.pth"))
        model.eval()  # Important! Switch model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        # Individual metrics
        acc, precision_macro, recall_macro, f1_macro, mcc, gmean =metrics(all_labels, all_preds)
        result = (f"Model: LSTM | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")

        return result

    def lstm_resampled_training(self, train_X, train_Y, trained_on_id):
        total_features = train_X.shape[1]
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"

        train_X, train_Y = undersample(train_X, train_Y, train_X.shape[0], total_features)
        train_X_np = train_X.reshape(-1, 1, total_features)

        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = LSTMClassifier(input_size=total_features, hidden_size=self.hidden_size, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_resampled_LSTM.pth")

    def lstm_resampled_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        total_features = test_X.shape[1]
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"

        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)
        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))
        test_loader = DataLoader(test_ds, batch_size=32)

        model = LSTMClassifier(input_size=total_features, hidden_size=self.hidden_size, num_classes=3)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_resampled_LSTM.pth"))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(all_labels, all_preds)

        result = (f"Model: LSTM-Resampled | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")

        return result

    def lstm_classweighted_training(self, train_X, train_Y, trained_on_id):
        total_features = train_X.shape[1]
        train_X_np = train_X.to_numpy().reshape(-1, 1, total_features)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_Y)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = LSTMClassifier(input_size=total_features, hidden_size=self.hidden_size, num_classes=3)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_classweighted_LSTM.pth")

    def lstm_classweighted_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        total_features = test_X.shape[1]
        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)

        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))
        test_loader = DataLoader(test_ds, batch_size=32)

        model = LSTMClassifier(input_size=total_features, hidden_size=self.hidden_size, num_classes=3)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_classweighted_LSTM.pth"))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(all_labels, all_preds)
        result = (f"Model: LSTM-Classweighted | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def dnn_training(self, train_X, train_Y, trained_on_id):
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        train_ds = TensorDataset(torch.Tensor(train_X.to_numpy()), torch.LongTensor(train_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        input_size = train_X.shape[1]
        num_classes = 3
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_DNN.pth")

    def dnn_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = test_X.shape[1]
        num_classes = 3
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_DNN.pth"))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(all_labels, all_preds)
        result = (f"Model: DNN | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def xgbpost_training(self, train_X, train_Y, trained_on_id):
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(train_X, train_Y)
        model.save_model(f"trained_on_{trained_on_id}_XGB.model")

    def xgbpost_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        model = XGBClassifier()
        model.load_model(f"trained_on_{trained_on_id}_XGB.model")
        preds = model.predict(test_X)
        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(test_Y.tolist(), preds.tolist())
        result = (f"Model: XGB | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def dnn_resampled_training(self, train_X, train_Y, trained_on_id):
        total_features = train_X.shape[1]
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"

        train_X, train_Y = undersample(train_X, train_Y, train_X.shape[0], total_features)
        train_X = train_X.reshape(train_X.shape[0], -1)

        train_ds = TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_Y))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        input_size = total_features
        num_classes = 3
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_resampled_DNN.pth")

    def dnn_resampled_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = test_X.shape[1]
        num_classes = 3
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_resampled_DNN.pth"))
        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(all_labels, all_preds)
        result = (f"Model: DNN-Resampled | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def dnn_classweighted_training(self, train_X, train_Y, trained_on_id):
        input_size = train_X.shape[1]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_Y)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        train_ds = TensorDataset(torch.Tensor(train_X.to_numpy()), torch.LongTensor(train_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = DNNClassifier(input_size, self.hidden_size, num_classes=3)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        model.eval()
        torch.save(model.state_dict(), f"trained_on_{trained_on_id}_classweighted_DNN.pth")

    def dnn_classweighted_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        input_size = test_X.shape[1]
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))
        test_loader = DataLoader(test_ds, batch_size=32)

        model = DNNClassifier(input_size, self.hidden_size, num_classes=3)
        model.load_state_dict(torch.load(f"trained_on_{trained_on_id}_classweighted_DNN.pth"))
        model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(all_labels, all_preds)
        result = (f"Model: DNN-Classweighted | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def xgbpost_reweighted_training(self, train_X, train_Y, trained_on_id):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_Y), y=train_Y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        sample_weights = train_Y.map(class_weight_dict)

        model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            learning_rate=0.05,
            max_depth=6,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(train_X, train_Y, sample_weight=sample_weights)
        model.save_model(f"trained_on_{trained_on_id}_reweighted_XGB.model")

    def xgbpost_reweighted_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        model = XGBClassifier()
        model.load_model(f"trained_on_{trained_on_id}_reweighted_XGB.model")
        preds = model.predict(test_X)

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(test_Y.tolist(), preds.tolist())
        result = (f"Model: XGB-Reweighted | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

    def xgbpost_resampled_training(self, train_X, train_Y, trained_on_id):
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(train_X, train_Y)
        model.save_model(f"trained_on_{trained_on_id}_resampled_XGB.model")

    def xgbpost_resampled_testing(self, test_X, test_Y, trained_on_id, test_on_id):
        model = XGBClassifier()
        model.load_model(f"trained_on_{trained_on_id}_resampled_XGB.model")
        preds = model.predict(test_X)

        acc, precision_macro, recall_macro, f1_macro, mcc, gmean = metrics(test_Y.tolist(), preds.tolist())
        result = (f"Model: XGB-Resampled | Trained on: {trained_on_id} | Tested on: {test_on_id} | "
                  f"Acc: {acc:.4f}, Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, "
                  f"F1: {f1_macro:.4f}, MCC: {mcc:.4f}, G-Mean: {gmean:.4f}")
        return result

def parse_arguments(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a model")
    parser.add_argument(
        "--dataset_id", type=str, default="Mascot_15", help="The dataset id"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="The output directory"
    )
    return parser.parse_args(args)

def plot_radar(results: dict, save_path="radar_plot.png"):
    labels = ["Accuracy", "Precision", "Recall", "F1", "MCC", "G-Mean"]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # repeat first angle to close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, metrics in results.items():
        values = metrics + [metrics[0]]  # loop back to the start
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_title("Model Performance Comparison (Snowflake Chart)", size=14)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def data_prep(args: list[str] = sys.argv):
    cfg = parse_arguments(args)
    pipeline = Data_Prep_Pipeline(cfg.dataset_id, cfg.output_dir)
    data = pipeline.get_data()
    train_X, train_Y, test_X, test_Y = pipeline.get_train_test_split(data)
    return (train_X, train_Y, test_X, test_Y, cfg.dataset_id)

def trainer (train_X, train_Y, dataset_id) -> None:
    pipeline = Model_training_class ()

    pipeline.lstm_training(train_X, train_Y, dataset_id)

    pipeline.lstm_resampled_training(train_X, train_Y, dataset_id)

    pipeline.lstm_classweighted_training(train_X, train_Y, dataset_id)

    pipeline.dnn_training(train_X, train_Y, dataset_id)

    pipeline.dnn_resampled_training(train_X, train_Y, dataset_id)

    pipeline.dnn_classweighted_training(train_X, train_Y, dataset_id)

    pipeline.xgbpost_training(train_X, train_Y, dataset_id)

    pipeline.xgbpost_reweighted_training(train_X, train_Y, dataset_id)

    pipeline.xgbpost_resampled_training(train_X, train_Y, dataset_id)
def tester(test_X, test_Y, trained_on_id, test_on_id) -> list:
    pipeline = Model_training_class()
    results = []
    results.append(pipeline.lstm_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.lstm_resampled_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.lstm_classweighted_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.dnn_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.dnn_resampled_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.dnn_classweighted_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.xgbpost_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.xgbpost_reweighted_testing(test_X, test_Y, trained_on_id, test_on_id))
    results.append(pipeline.xgbpost_resampled_testing(test_X, test_Y, trained_on_id, test_on_id))

    return results

# def main(args: list[str] = sys.argv) -> None:
#     cfg = parse_arguments(args)
#     pipeline = PredictionPipeline(cfg.dataset_id, cfg.output_dir)
#     results = {}
#     data = pipeline.get_data()
#     train_X, train_Y, test_X, test_Y = pipeline.get_train_test_split(data)
#     pipeline.lstm_training( train_X, train_Y,  cfg.dataset_id)
#     all_labels, all_preds = pipeline.lstm_testing(test_X, test_Y, cfg.dataset_id)
#     results["LSTM"]=visualized_aligned (all_labels =all_labels, all_preds =all_preds, logger =pipeline.logger, model_name ="LSTM", dataset_id =cfg.dataset_id)
#
#     pipeline.lstm_resampled_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.lstm_resampled_testing(test_X, test_Y, cfg.dataset_id)
#     results["LSTM-US"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-US", dataset_id=cfg.dataset_id)
#
#     pipeline.lstm_classweighted_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.lstm_classweighted_testing(test_X, test_Y, cfg.dataset_id)
#     results["LSTM-CW"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-CW", dataset_id=cfg.dataset_id)
#     # Normal DNN
#     pipeline.dnn_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.dnn_testing(test_X, test_Y, cfg.dataset_id)
#     results["DNN"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN", dataset_id=cfg.dataset_id)
#     # Resampled DNN
#     pipeline.dnn_resampled_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.dnn_resampled_testing(test_X, test_Y, cfg.dataset_id)
#     results["DNN-US"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-US", dataset_id=cfg.dataset_id)
#     # Classweighted DNN
#     pipeline.dnn_classweighted_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.dnn_classweighted_testing(test_X, test_Y, cfg.dataset_id)
#     results["DNN-CW"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-CW", dataset_id=cfg.dataset_id)
#
#     # Normal XGB
#     pipeline.xgbpost_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.xgbpost_testing(test_X, test_Y, cfg.dataset_id)
#     results["Xgboost"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger,  model_name="Xgboost", dataset_id=cfg.dataset_id)
#     # Re-weighted XGB
#     pipeline.xgbpost_reweighted_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.xgbpost_reweighted_testing(test_X, test_Y, cfg.dataset_id)
#     results["Xgboost-US"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost-US", dataset_id=cfg.dataset_id)
#
#     # Resampled XGB
#     pipeline.xgbpost_resampled_training(train_X, train_Y, cfg.dataset_id)
#     all_labels, all_preds = pipeline.xgbpost_resampled_testing(test_X, test_Y, cfg.dataset_id)
#     results["Xgboost-CW"] = visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger,
#                                                model_name="Xgboost-CW", dataset_id=cfg.dataset_id)
#
#     # all_labels, all_preds= pipeline.lstm_run_resampled( train_X, train_Y, test_X,test_Y) #lstm_run_classweighted
#     # results["LSTM-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-US",  dataset_id=cfg.dataset_id)
#     # #
#     # all_labels, all_preds = pipeline.lstm_run_classweighted( train_X, train_Y, test_X,test_Y)  # lstm_run_classweighted
#     # results["LSTM-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-CW", dataset_id=cfg.dataset_id)
#     #
#     # all_labels, all_preds = pipeline.dnn_run( train_X, train_Y, test_X,test_Y)
#     # results["DNN"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN",dataset_id=cfg.dataset_id)
#     # all_labels, all_preds = pipeline.dnn_run_resampled( train_X, train_Y, test_X,test_Y) #dnn_run_classweighted
#     # results["DNN-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-US",
#     #                    dataset_id=cfg.dataset_id)
#     # all_labels, all_preds = pipeline.dnn_run_classweighted(  train_X, train_Y, test_X,test_Y)
#     # results["DNN-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-CW",
#     #                    dataset_id=cfg.dataset_id)
#     #
#     # all_labels, all_preds = pipeline.xgbpost_run( train_X, train_Y, test_X,test_Y)
#     # results["Xgboost"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost",
#     #                    dataset_id=cfg.dataset_id)
#     # all_labels, all_preds = pipeline.xgbpost_run_resampled( train_X, train_Y, test_X,test_Y)
#     # results["Xgboost-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost-US",
#     #                    dataset_id=cfg.dataset_id)
#     # all_labels, all_preds = pipeline.xgbpost_run_re_weighted( train_X, train_Y, test_X,test_Y)
#     # results["Xgboost-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost-CW",
#     #                    dataset_id=cfg.dataset_id)
#     #
#     plot_radar(results, save_path=pipeline.output_dir / f"{cfg.dataset_id}_model_comparison_radar.png")

if __name__ == "__main__":
    datasets = {}

    data_dir = pathlib.Path(__file__).parent / ".." / ".." / "data"

    for file in tqdm([x for x in data_dir.glob("*") if x.is_dir()], desc="Loading datasets"):
        train_X, train_Y, test_X, test_Y, dataset_id = data_prep(
            ["--dataset_id", file.name, "--output_dir", f"output/lagged/{file.name}"])
        datasets[dataset_id] = {
            "train_X": train_X,
            "train_Y": train_Y,
            "test_X": test_X,
            "test_Y": test_Y,
        }
    print(f"Number of datasets loaded: {len(datasets)}")
    for dataset_id, data in datasets.items():
        print(f"\nTraining models for dataset: {dataset_id}")
        trainer (data["train_X"], data["train_Y"], dataset_id)
    # ====== Testing Phase ======
    print("\n==== Testing Models ====")
    all_results = []

    dataset_ids = list(datasets.keys())

    for trained_on_id in dataset_ids:
        for test_on_id in dataset_ids:
            print(f"\nTesting models: Trained on {trained_on_id} | Tested on {test_on_id}")
            results = tester(
                datasets[test_on_id]["test_X"],
                datasets[test_on_id]["test_Y"],
                trained_on_id,
                test_on_id
            )
            all_results.extend(results)

    # ====== Print Final Results ======
    print("\n==== Final Testing Results ====")
    for res in all_results:
        print(res)

    df_results = pd.DataFrame([{
        "Model": r.split("|")[0].split(":")[1].strip(),
        "Trained_on": r.split("|")[1].split(":")[1].strip(),
        "Tested_on": r.split("|")[2].split(":")[1].strip(),
        "Acc": float(r.split("|")[3].split(",")[0].split(":")[1]),
        "Precision": float(r.split("Precision:")[1].split(",")[0]),
        "Recall": float(r.split("Recall:")[1].split(",")[0]),
        "F1": float(r.split("F1:")[1].split(",")[0]),
        "MCC": float(r.split("MCC:")[1].split(",")[0]),
        "G-Mean": float(r.split("G-Mean:")[1]),
    } for r in all_results])

    output_path = pathlib.Path(__file__).parent / "final_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for trained_on in dataset_ids:
            for tested_on in dataset_ids:
                table = df_results[
                    (df_results["Trained_on"] == trained_on) & (df_results["Tested_on"] == tested_on)
                    ]
                sheet_name = f"{trained_on[:6]}_to_{tested_on[:6]}"  # Limit sheet name to 31 chars if needed
                table.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nSaved all results to: {output_path}")




