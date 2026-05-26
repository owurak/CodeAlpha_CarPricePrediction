import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("car data.csv")

# Preview data
print("First 5 rows:")
print(df.head())

# -----------------------
# DATA CLEANING
# -----------------------

# Drop Car Name (not useful for ML)
df = df.drop("Car_Name", axis=1)

# Convert Year → Car Age
df["Current_Year"] = 2026
df["Car_Age"] = df["Current_Year"] - df["Year"]
df = df.drop(["Year", "Current_Year"], axis=1)

# One-hot encoding for categorical features
df = pd.get_dummies(df, drop_first=True)

# -----------------------
# SPLIT DATA
# -----------------------
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# MODEL
# -----------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------
# EVALUATION
# -----------------------
print("\nR2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# -----------------------
# SAMPLE PREDICTION
# -----------------------
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Prediction:", prediction[0])