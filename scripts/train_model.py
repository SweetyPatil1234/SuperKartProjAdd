import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor,
                               AdaBoostRegressor, GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# Load data
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH  = "data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

TARGET = "Product_Store_Sales_Total"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

# Encode categorical columns
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train  = pd.get_dummies(X_train, columns=cat_cols)
X_test   = pd.get_dummies(X_test,  columns=cat_cols)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Define models
models_params = {
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }
}

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

results    = []
all_logs   = {}
best_model = None
best_score = -np.inf
best_name  = ""

for model_name, mp in models_params.items():
    print(f"\nTraining: {model_name}")

    grid = GridSearchCV(
        estimator  = mp["model"],
        param_grid = mp["params"],
        cv         = 3,
        scoring    = "r2",
        n_jobs     = -1
    )
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    y_pred = best_estimator.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"Best Parameters : {grid.best_params_}")
    print(f"R2 Score        : {r2:.4f}")
    print(f"RMSE            : {rmse:.4f}")
    print(f"MAE             : {mae:.4f}")

    log = {
        "model"         : model_name,
        "best_params"   : grid.best_params_,
        "best_cv_score" : round(grid.best_score_, 4),
        "r2_score"      : round(r2,   4),
        "rmse"          : round(rmse, 4),
        "mae"           : round(mae,  4)
    }
    all_logs[model_name] = log
    results.append({
        "Model"   : model_name,
        "R2 Score": round(r2,   4),
        "RMSE"    : round(rmse, 4),
        "MAE"     : round(mae,  4)
    })

    if r2 > best_score:
        best_score = r2
        best_model = best_estimator
        best_name  = model_name

# Save model and logs
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(best_model.feature_names_in_.tolist(),
            "models/train_columns.pkl")

with open("logs/experiment_logs.json", "w") as f:
    json.dump(all_logs, f, indent=4)

with open("logs/best_model_info.json", "w") as f:
    json.dump({
        "best_model"    : best_name,
        "best_r2_score" : round(best_score, 4)
    }, f, indent=4)

print(f"\nBest Model : {best_name}")
print(f"Best R2    : {best_score:.4f}")
print("Model and logs saved successfully!")
