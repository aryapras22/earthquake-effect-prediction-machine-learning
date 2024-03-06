# %%
# Feature Selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Chi Square
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# PCA
from sklearn.decomposition import PCA
# Feature Importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# %%
# Load the dataset
data = pd.read_csv('../train_earthquake.csv')
df = pd.DataFrame(data)
print(df)
df = df.dropna()

# Select features and target variable
X = df[['old_building', 'height_before_eq (ft)', 'wall_binding', 'wall_material']]
y = df['damage_grade']

print("X shape:", X.shape)
print("y shape:", y.shape)


# %%
# Chi Square
best_features = SelectKBest(score_func=chi2, k=2)
X_new = best_features.fit_transform(X, y)
print("Shape :", X_new.shape)
print("Data : \n", X_new)


# %%
# Seleksi fitur dengan PCA
pca = PCA(n_components=2)  # memilih 2 komponen
X_pca = pca.fit_transform(X)
print("Hasil PCA :\n", X_pca)

# Plot Hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of IRIS Dataset")
plt.colorbar(label="Species")
plt.show()

# %%
# Model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Menghitung Feature Importance
importances = model.feature_importances_

# Plot Feature Importance
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), data.feature_names, rotation=15)
plt.xlabel("Fitur")
plt.ylabel("Importance")
plt.show()


#%% 
model = RandomForestClassifier()
model.fit(X, y)

# Hitung Feature Importance
importances = model.feature_importances_

# Plot Feature Importance
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), data.feature_names, rotation=15)
plt.xlabel("Fitur")
plt.ylabel("Importance")
plt.show()


#%%'
model = GradientBoostingClassifier()
model.fit(X, y)

importances = model.feature_importances_

plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), iris.feature_names, rotation=15)
plt.xlabel("Fitur")
plt.ylabel("Importance")
plt.show()
