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
class PredictionPipeline:
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

    def lstm_run(self, train_X, train_Y, test_X,test_Y):
        # Scale input

        total_features = train_X.shape[1]
        # Final NaN check (just in case)
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"

        # Reshape to [batch, seq_len=1, features]

        train_X_np = train_X.to_numpy().reshape(-1, 1, total_features)
        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)


        # Wrap in PyTorch Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y.to_numpy()))
        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

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
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        acc = correct / total
        print(f"Test Accuracy: {acc:.2%}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        # Individual metrics
        return(all_labels, all_preds)

    def lstm_run_resampled (self,train_X, train_Y, test_X,test_Y) -> None:

        total_features = train_X.shape[1]

        # Final NaN check (just in case)
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"

        train_X, train_Y = undersample(train_X, train_Y, train_X.shape[0], total_features)
        train_X_np = train_X.reshape(-1, 1, total_features)
        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)

        # Wrap in PyTorch Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y))
        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = total_features
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
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        acc = correct / total
        print(f"Test Accuracy: {acc:.2%}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        return (all_labels, all_preds)

    def lstm_run_classweighted(self,train_X, train_Y, test_X,test_Y):
        total_features = train_X.shape[1]
        train_X_np = train_X.to_numpy().reshape(-1, 1, total_features)
        test_X_np = test_X.to_numpy().reshape(-1, 1, total_features)

        # Compute class weights
        # class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1, 2], y=train_Y)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_Y)

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X_np), torch.LongTensor(train_Y.to_numpy()))
        test_ds = TensorDataset(torch.Tensor(test_X_np), torch.LongTensor(test_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        # Model
        model = LSTMClassifier(input_size=total_features, hidden_size=self.hidden_size, num_classes=3)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 🔥 class-weighted loss
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        return all_labels, all_preds

    def dnn_run(self,train_X, train_Y, test_X,test_Y):

        # Final NaN check (just in case)
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"

        # Wrap in PyTorch Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X.to_numpy()), torch.LongTensor(train_Y.to_numpy()))
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = train_X.shape[1]  # Number of features
        num_classes = 3  # Number of classes

        # Instantiate the DNN model
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()  # For classification
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        acc = correct / total
        print(f"Test Accuracy: {acc:.2%}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        return all_labels, all_preds

    def dnn_run_resampled(self,train_X, train_Y, test_X,test_Y) -> None:

        total_features = train_X.shape[1]
        # Final NaN check (just in case)
        assert not train_X.isnull().any().any(), "NaNs in train_X after scaling"
        assert not test_X.isnull().any().any(), "NaNs in test_X after scaling"

        train_X, train_Y = undersample(train_X, train_Y, train_X.shape[0], total_features)
        train_X = train_X.reshape(train_X.shape[0], -1)  # Make sure it's 2D: [samples, features]

        # Wrap in PyTorch Dataset & DataLoader
        train_ds = TensorDataset(torch.Tensor(train_X), torch.LongTensor(train_Y))
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = total_features
        num_classes = 3  # Number of classes

        # Instantiate the DNN model
        model = DNNClassifier(input_size, self.hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        acc = correct / total
        print(f"Test Accuracy: {acc:.2%}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        return all_labels, all_preds

    def dnn_run_classweighted(self,train_X, train_Y, test_X,test_Y):
        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_Y)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # Convert to PyTorch tensors
        train_ds = TensorDataset(torch.Tensor(train_X.to_numpy()), torch.LongTensor(train_Y.to_numpy()))
        test_ds = TensorDataset(torch.Tensor(test_X.to_numpy()), torch.LongTensor(test_Y.to_numpy()))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        input_size = train_X.shape[1]
        num_classes = 3

        # Define DNN model
        model = DNNClassifier(input_size=input_size, hidden_size=self.hidden_size, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 🔥 class-weighted
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epoches):
            model.train()
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                _, predicted = torch.max(pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        return all_labels, all_preds

    def xgbpost_run(self,train_X, train_Y, test_X,test_Y):
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

        # Predictions
        preds = model.predict(test_X)

        return test_Y.tolist(), preds.tolist()
    def xgbpost_run_re_weighted(self,train_X, train_Y, test_X,test_Y):
        # Class weights for imbalance handling
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_Y), y=train_Y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Map weights to each training sample
        sample_weights = train_Y.map(class_weight_dict)

        # Train XGBoost
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
        # Predictions
        preds = model.predict(test_X)
        return test_Y.tolist(), preds.tolist()

    def xgbpost_run_resampled(self,train_X, train_Y, test_X,test_Y):
        # Model (no class weights)
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

        # Predictions
        preds = model.predict(test_X)

        return test_Y.tolist(), preds.tolist()


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


def parse_arguments(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a model")
    parser.add_argument(
        "--dataset_id", type=str, default="Mascot_15", help="The dataset id"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="The output directory"
    )
    return parser.parse_args(args)


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

def visualized_aligned(
    all_labels: list,
    all_preds: list,
    model_name: str,
    logger,
    dataset_id: str
):
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    gmean = geometric_mean_score(all_labels, all_preds, average='macro')

    # Print results
    logger.info(f"{dataset_id}'s {model_name} Results:")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"Precision (macro): {precision_macro:.4f}")
    logger.info(f"Recall (macro): {recall_macro:.4f}")
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"G-Mean: {gmean:.4f}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(all_labels, all_preds)}\n")
    logger.info(f"Classification report:\n{classification_report(all_labels, all_preds)}")
    return [acc, precision_macro, recall_macro, f1_macro, mcc, gmean]



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
    plt.close()

def main(args: list[str] = sys.argv) -> None:
    cfg = parse_arguments(args)
    pipeline = PredictionPipeline(cfg.dataset_id, cfg.output_dir)
    results = {}
    data = pipeline.get_data()
    train_X, train_Y, test_X, test_Y = pipeline.get_train_test_split(data)

    all_labels, all_preds =pipeline.lstm_run( train_X, train_Y, test_X, test_Y)
    results["LSTM"]=visualized_aligned (all_labels =all_labels, all_preds =all_preds, logger =pipeline.logger, model_name ="LSTM", dataset_id =cfg.dataset_id)

    all_labels, all_preds= pipeline.lstm_run_resampled( train_X, train_Y, test_X,test_Y) #lstm_run_classweighted
    results["LSTM-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-US",  dataset_id=cfg.dataset_id)

    all_labels, all_preds = pipeline.lstm_run_classweighted( train_X, train_Y, test_X,test_Y)  # lstm_run_classweighted
    results["LSTM-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="LSTM-CW", dataset_id=cfg.dataset_id)

    all_labels, all_preds = pipeline.dnn_run( train_X, train_Y, test_X,test_Y)
    results["DNN"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN",dataset_id=cfg.dataset_id)
    all_labels, all_preds = pipeline.dnn_run_resampled( train_X, train_Y, test_X,test_Y) #dnn_run_classweighted
    results["DNN-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-US",
                       dataset_id=cfg.dataset_id)
    all_labels, all_preds = pipeline.dnn_run_classweighted(  train_X, train_Y, test_X,test_Y)
    results["DNN-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="DNN-CW",
                       dataset_id=cfg.dataset_id)

    all_labels, all_preds = pipeline.xgbpost_run( train_X, train_Y, test_X,test_Y)
    results["Xgboost"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost",
                       dataset_id=cfg.dataset_id)
    all_labels, all_preds = pipeline.xgbpost_run_resampled( train_X, train_Y, test_X,test_Y)
    results["Xgboost-US"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost-US",
                       dataset_id=cfg.dataset_id)
    all_labels, all_preds = pipeline.xgbpost_run_re_weighted( train_X, train_Y, test_X,test_Y)
    results["Xgboost-CW"]=visualized_aligned(all_labels=all_labels, all_preds=all_preds, logger=pipeline.logger, model_name="Xgboost-CW",
                       dataset_id=cfg.dataset_id)

    plot_radar(results, save_path=pipeline.output_dir / f"{cfg.dataset_id}_model_comparison_radar.png")

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / ".." / ".." / "data"
    for file in tqdm(
        [x for x in data_dir.glob("*") if x.is_dir()], desc="Processing datasets"
    ):
        main(["--dataset_id", file.name, "--output_dir", f"output/lagged/{file.name}"])
