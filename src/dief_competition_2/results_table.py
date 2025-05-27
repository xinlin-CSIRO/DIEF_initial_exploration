import pandas as pd
import re

# Paste your big results string here:
results_text = """==== Testing Models ====

Testing models: Trained on Melbourne_05_Y5 | Tested on Melbourne_05_Y5

Testing models: Trained on Melbourne_05_Y5 | Tested on Sydney_05_Y5

Testing models: Trained on Sydney_05_Y5 | Tested on Melbourne_05_Y5

Testing models: Trained on Sydney_05_Y5 | Tested on Sydney_05_Y5

==== Final Testing Results ====
Model: LSTM | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9749, Precision: 0.5916, Recall: 0.3375, F1: 0.3372, MCC: 0.0595, G-Mean: 0.4748
Model: LSTM-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.6480, Precision: 0.3625, Recall: 0.6738, F1: 0.3188, MCC: 0.1577, G-Mean: 0.7554
Model: LSTM-Classweighted | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.8322, Precision: 0.3952, Recall: 0.6926, F1: 0.4132, MCC: 0.2323, G-Mean: 0.7758
Model: DNN | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9748, Precision: 0.4917, Recall: 0.3466, F1: 0.3539, MCC: 0.0837, G-Mean: 0.4824
Model: DNN-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.6614, Precision: 0.3638, Recall: 0.6748, F1: 0.3242, MCC: 0.1637, G-Mean: 0.7588
Model: DNN-Classweighted | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.8289, Precision: 0.4004, Recall: 0.7077, F1: 0.4192, MCC: 0.2425, G-Mean: 0.7886
Model: XGB | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9764, Precision: 0.9030, Recall: 0.3867, F1: 0.4261, MCC: 0.2497, G-Mean: 0.5169
Model: XGB-Reweighted | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.7851, Precision: 0.3957, Recall: 0.8452, F1: 0.4058, MCC: 0.2709, G-Mean: 0.8802
Model: XGB-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9764, Precision: 0.9030, Recall: 0.3867, F1: 0.4261, MCC: 0.2497, G-Mean: 0.5169
Model: LSTM | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9750, Precision: 0.3250, Recall: 0.3333, F1: 0.3291, MCC: 0.0000, G-Mean: 0.4714
Model: LSTM-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.6611, Precision: 0.3623, Recall: 0.6935, F1: 0.3213, MCC: 0.1606, G-Mean: 0.7713
Model: LSTM-Classweighted | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.8190, Precision: 0.3921, Recall: 0.7079, F1: 0.4068, MCC: 0.2234, G-Mean: 0.7839
Model: DNN | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9752, Precision: 0.6584, Recall: 0.3399, F1: 0.3420, MCC: 0.0751, G-Mean: 0.4767
Model: DNN-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.6696, Precision: 0.3637, Recall: 0.7018, F1: 0.3257, MCC: 0.1666, G-Mean: 0.7788
Model: DNN-Classweighted | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.8110, Precision: 0.3947, Recall: 0.7212, F1: 0.4086, MCC: 0.2323, G-Mean: 0.7951
Model: XGB | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9759, Precision: 0.9920, Recall: 0.3649, F1: 0.3883, MCC: 0.1837, G-Mean: 0.4974
Model: XGB-Reweighted | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.7702, Precision: 0.3775, Recall: 0.7277, F1: 0.3729, MCC: 0.2101, G-Mean: 0.7975
Model: XGB-Resampled | Trained on: Melbourne_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9759, Precision: 0.9920, Recall: 0.3649, F1: 0.3883, MCC: 0.1837, G-Mean: 0.4974
Model: LSTM | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9748, Precision: 0.3249, Recall: 0.3333, F1: 0.3291, MCC: 0.0000, G-Mean: 0.4714
Model: LSTM-Resampled | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.7429, Precision: 0.3618, Recall: 0.5769, F1: 0.3346, MCC: 0.1367, G-Mean: 0.6910
Model: LSTM-Classweighted | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.7866, Precision: 0.3711, Recall: 0.5983, F1: 0.3665, MCC: 0.1636, G-Mean: 0.7037
Model: DNN | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9748, Precision: 0.3249, Recall: 0.3333, F1: 0.3291, MCC: 0.0000, G-Mean: 0.4714
Model: DNN-Resampled | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.7041, Precision: 0.3718, Recall: 0.6029, F1: 0.3459, MCC: 0.1571, G-Mean: 0.7119
Model: DNN-Classweighted | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.7795, Precision: 0.3717, Recall: 0.6097, F1: 0.3657, MCC: 0.1689, G-Mean: 0.7130
Model: XGB | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9682, Precision: 0.4276, Recall: 0.3786, F1: 0.3930, MCC: 0.1006, G-Mean: 0.5116
Model: XGB-Reweighted | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.8264, Precision: 0.3788, Recall: 0.6275, F1: 0.3869, MCC: 0.1975, G-Mean: 0.7294
Model: XGB-Resampled | Trained on: Sydney_05_Y5 | Tested on: Melbourne_05_Y5 | Acc: 0.9682, Precision: 0.4276, Recall: 0.3786, F1: 0.3930, MCC: 0.1006, G-Mean: 0.5116
Model: LSTM | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9750, Precision: 0.3250, Recall: 0.3333, F1: 0.3291, MCC: 0.0000, G-Mean: 0.4714
Model: LSTM-Resampled | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.7522, Precision: 0.3745, Recall: 0.5848, F1: 0.3435, MCC: 0.1479, G-Mean: 0.7066
Model: LSTM-Classweighted | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.8152, Precision: 0.3912, Recall: 0.6907, F1: 0.4043, MCC: 0.2122, G-Mean: 0.7721
Model: DNN | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9750, Precision: 0.3250, Recall: 0.3333, F1: 0.3291, MCC: 0.0000, G-Mean: 0.4714
Model: DNN-Resampled | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.7111, Precision: 0.3841, Recall: 0.6483, F1: 0.3660, MCC: 0.1771, G-Mean: 0.7448
Model: DNN-Classweighted | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.7871, Precision: 0.3829, Recall: 0.6813, F1: 0.3857, MCC: 0.1956, G-Mean: 0.7626
Model: XGB | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9799, Precision: 0.8525, Recall: 0.4983, F1: 0.5778, MCC: 0.4444, G-Mean: 0.6113
Model: XGB-Reweighted | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.8581, Precision: 0.4367, Recall: 0.8665, F1: 0.4830, MCC: 0.3238, G-Mean: 0.8930
Model: XGB-Resampled | Trained on: Sydney_05_Y5 | Tested on: Sydney_05_Y5 | Acc: 0.9799, Precision: 0.8525, Recall: 0.4983, F1: 0.5778, MCC: 0.4444, G-Mean: 0.6113

"""

# Step 1. Parse the results into structured list
pattern = r"Model: (.*?) \| Trained on: (.*?) \| Tested on: (.*?) \| Acc: (.*?), Precision: (.*?), Recall: (.*?), F1: (.*?), MCC: (.*?), G-Mean: (.*?)$"
matches = re.findall(pattern, results_text, re.MULTILINE)

# Build a DataFrame
columns = ["Model", "Trained_on", "Tested_on", "Acc", "Precision", "Recall", "F1", "MCC", "GMean"]
df = pd.DataFrame(matches, columns=columns)

# Convert numerical columns to float
for col in ["Acc", "Precision", "Recall", "F1", "MCC", "GMean"]:
    df[col] = df[col].astype(float)

# Step 2. Create 4 tables
melbourne = "Melbourne_05_Y5"
sydney = "Sydney_05_Y5"

table1 = df[(df["Trained_on"] == melbourne) & (df["Tested_on"] == melbourne)]
table2 = df[(df["Trained_on"] == melbourne) & (df["Tested_on"] == sydney)]
table3 = df[(df["Trained_on"] == sydney) & (df["Tested_on"] == melbourne)]
table4 = df[(df["Trained_on"] == sydney) & (df["Tested_on"] == sydney)]

# Step 3. Save into an Excel file
with pd.ExcelWriter("final_results.xlsx") as writer:
    table1.to_excel(writer, sheet_name="Train Melbourne - Test Melb", index=False)
    table2.to_excel(writer, sheet_name="Train Melbourne - Test Syd", index=False)
    table3.to_excel(writer, sheet_name="Train Sydney - Test Melb", index=False)
    table4.to_excel(writer, sheet_name="Train Sydney - Test Syd", index=False)

print("✅ Saved final_results.xlsx successfully!")
