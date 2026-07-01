import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load Dataset
df = pd.read_csv(r"C:\Users\konat\OneDrive\Documents\data.csv")
print(df.head())


print("Info:")
print(df.info())

print("\nShape Before Cleaning:")
print(df.shape)

# Remove unnecessary columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

print("\nShape After Cleaning:")
print(df.shape)

# Convert target column
df["diagnosis"] = df["diagnosis"].map({
    "M": 0,
    "B": 1
})

# Features and Target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Hyperparameter Grid (18 Configurations)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# Model
rf = RandomForestClassifier(random_state=42)

# Grid Search CV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    return_train_score=True,
    n_jobs=-1
)

# Start Timer
start = time.time()

grid_search.fit(X, y)

# End Timer
end = time.time()

# Results
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "mean_train_score",
        "mean_test_score",
        "mean_fit_time"
    ]
]

# Overfitting Gap
results["overfitting_gap"] = (
    results["mean_train_score"]
    - results["mean_test_score"]
)

# Save Results
results.to_csv(
    "hyperparameter_results.csv",
    index=False
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Accuracy:")
print(round(grid_search.best_score_, 4))

print("\nTotal Training Time:")
print(round(end - start, 2), "seconds")

print("\nResults saved as hyperparameter_results.csv")