import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("student-mat.csv", sep=";")


print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include=np.number) 
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(df["G3"], kde=True, bins=15, color="blue")
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.show()


sns.pairplot(df[["G1", "G2", "G3", "studytime", "failures"]], diag_kind="kde", corner=True)
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


categorical_features = ["school", "sex", "address", "famsize", "Pstatus"]
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature], y=df["G3"], palette="Set2")
    plt.title(f"G3 Distribution by {feature}")
    plt.xlabel(feature)
    plt.ylabel("G3")
    plt.show()


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
