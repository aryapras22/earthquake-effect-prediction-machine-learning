import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Read data
data = pd.read_csv("../train_earthquake.csv")
print(data)

# Mengambil data old building
data2 = data[["old_building"]]
print(data2)

# Menghapus data NA dari oldbuilding
print("\n==== Detect Missing Values ====\n")
x = data2.isna().sum()
print("Jumlah Missing value :", x)

# Drop Missing Values
data3 = data2.dropna()
print(data3)

plt.boxplot(
    data3["old_building"]
)  # Plotting the boxplot with the preserved column name
plt.show()

outliers = []


def detect_outlier(data):

    threshold = 3
    mean_1 = np.mean(data)
    std_1 = np.std(data)

    for y in data:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


print("===== Outliers ====")

outlier_datapoints = detect_outlier(data3["old_building"])
print("Jumlah Outliers : ", len(outlier_datapoints))

dataNonOutliers = []


# Dropping outliers
def remove_outlier(data, threshold):

    mean_1 = np.mean(data)
    std_1 = np.std(data)

    for y in data:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) < threshold:
            dataNonOutliers.append(y)

    return dataNonOutliers


print("\n data tanpa outlier")
x = remove_outlier(data3["old_building"], 3)
data4 = pd.DataFrame(x)
print(data4)

plt.boxplot(data4)  # Plotting the boxplot with the preserved column name
plt.show()


print("\nNormalisasi\n")
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data4)
df_normalized = pd.DataFrame(np_scaled)
print(df_normalized)
plt.boxplot(df_normalized)
plt.show()
