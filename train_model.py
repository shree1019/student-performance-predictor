import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("student-mat.csv", sep=";")

df["trend"] = (df["G2"] - df["G1"]) / df["G1"]
df["trend"] = df["trend"].replace([np.inf, -np.inf], 0).fillna(0)

features = ["G1", "G2", "studytime", "failures", "absences", "trend"]
target = "G3"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

joblib.dump(model, "student_performance_model.pkl")
joblib.dump(scaler, "scaler.pkl")
