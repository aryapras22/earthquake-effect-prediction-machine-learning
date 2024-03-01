import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
import seaborn as sns

data = pd.read_csv("../train_earthquake.csv")
df = pd.DataFrame(data)
print(df)

data2 = df["old_building"]
print(data2)

print("==== Detect Missing Values ====")
x = data2.isna().sum()
print("Jumlah Missing value :", x)
# Drop Missing Values
data3 = data2.dropna()
print(data3)

# Remove Outliers
plt.boxplot(data3)
plt.show()
