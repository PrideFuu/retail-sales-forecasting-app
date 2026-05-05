import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os
import mlflow

print("WORKING DIR:", os.getcwd())

mlflow.set_tracking_uri("file:./mlruns")
print("MLFLOW URI:", mlflow.get_tracking_uri())
print("Current folder:", os.getcwd())
mlflow.set_tracking_uri("file:./mlruns")
print("MLflow tracking URI:", mlflow.get_tracking_uri())
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------
# Daten laden
# ---------------------------
df = pd.read_csv("data/timeseries_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])

df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

if "description" in df.columns:
    df["is_holiday"] = df["description"].notna().astype(int)
else:
    df["is_holiday"] = 0


# ---------------------------
# Feature Engineering
# ---------------------------
df_fe = df.copy()

df_fe["year"] = df_fe["date"].dt.year
df_fe["month"] = df_fe["date"].dt.month
df_fe["day_of_week"] = df_fe["date"].dt.dayofweek
df_fe["day_of_month"] = df_fe["date"].dt.day
df_fe["is_weekend"] = df_fe["day_of_week"].isin([5, 6]).astype(int)

df_fe["lag_1"] = df_fe["unit_sales"].shift(1)
df_fe["lag_7"] = df_fe["unit_sales"].shift(7)
df_fe["rolling_mean_7"] = df_fe["unit_sales"].rolling(window=7).mean()
df_fe["rolling_std_7"] = df_fe["unit_sales"].rolling(window=7).std()

df_fe = df_fe.dropna()


# ---------------------------
# Train-Test-Split
# ---------------------------
train_fe = df_fe[df_fe["date"] <= "2013-12-31"]
test_fe = df_fe[(df_fe["date"] >= "2014-01-01") & (df_fe["date"] <= "2014-03-31")]

feature_cols = [
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "dcoilwtico"
]

X_train = train_fe[feature_cols]
y_train = train_fe["unit_sales"]

X_test = test_fe[feature_cols]
y_test = test_fe["unit_sales"]


# ---------------------------
# Metriken
# ---------------------------
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_true, y_pred)
    }


# ---------------------------
# MLflow Setup
# ---------------------------
mlflow.set_experiment("Retail Sales Forecasting - Week 3")

best_models = []


# ---------------------------
# Modell 1: Linear Regression Baseline
# ---------------------------
with mlflow.start_run(run_name="Linear Regression Baseline"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    metrics = get_metrics(y_test, pred_lr)

    mlflow.log_param("model_type", "LinearRegression")

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    mlflow.sklearn.log_model(lr, "model")

    best_models.append({
        "name": "Linear Regression",
        "model": lr,
        "metrics": metrics
    })

    print("Linear Regression:")
    print(metrics)


# ---------------------------
# Modell 2: Random Forest mit HyperOpt
# ---------------------------
def objective_rf(params):
    params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = get_metrics(y_test, preds)

    with mlflow.start_run(run_name="Random Forest HyperOpt"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_params(params)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, "model")

    return {
        "loss": metrics["rmse"],
        "status": STATUS_OK,
        "model": model,
        "metrics": metrics,
        "params": params
    }


space_rf = {
    "n_estimators": hp.quniform("rf_n_estimators", 50, 300, 50),
    "max_depth": hp.quniform("rf_max_depth", 3, 20, 1),
    "min_samples_split": hp.quniform("rf_min_samples_split", 2, 10, 1),
    "min_samples_leaf": hp.quniform("rf_min_samples_leaf", 1, 5, 1)
}

trials_rf = Trials()

fmin(
    fn=objective_rf,
    space=space_rf,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials_rf
)

best_rf_result = min(trials_rf.results, key=lambda x: x["loss"])

best_models.append({
    "name": "Random Forest",
    "model": best_rf_result["model"],
    "metrics": best_rf_result["metrics"],
    "params": best_rf_result["params"]
})

print("\nBest Random Forest:")
print(best_rf_result["metrics"])
print(best_rf_result["params"])


# ---------------------------
# Modell 3: Gradient Boosting mit HyperOpt
# ---------------------------
def objective_gb(params):
    params = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "random_state": 42
    }

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = get_metrics(y_test, preds)

    with mlflow.start_run(run_name="Gradient Boosting HyperOpt"):
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_params(params)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, "model")

    return {
        "loss": metrics["rmse"],
        "status": STATUS_OK,
        "model": model,
        "metrics": metrics,
        "params": params
    }


space_gb = {
    "n_estimators": hp.quniform("gb_n_estimators", 50, 300, 50),
    "learning_rate": hp.uniform("gb_learning_rate", 0.01, 0.2),
    "max_depth": hp.quniform("gb_max_depth", 2, 8, 1),
    "min_samples_split": hp.quniform("gb_min_samples_split", 2, 10, 1),
    "min_samples_leaf": hp.quniform("gb_min_samples_leaf", 1, 5, 1)
}

trials_gb = Trials()

fmin(
    fn=objective_gb,
    space=space_gb,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials_gb
)

best_gb_result = min(trials_gb.results, key=lambda x: x["loss"])

best_models.append({
    "name": "Gradient Boosting",
    "model": best_gb_result["model"],
    "metrics": best_gb_result["metrics"],
    "params": best_gb_result["params"]
})

print("\nBest Gradient Boosting:")
print(best_gb_result["metrics"])
print(best_gb_result["params"])


# ---------------------------
# Champion Model auswählen und speichern
# ---------------------------
champion = min(best_models, key=lambda x: x["metrics"]["rmse"])

print("\nChampion Model:")
print(champion["name"])
print(champion["metrics"])

joblib.dump(champion["model"], "champion_model.pkl")
joblib.dump(feature_cols, "model_features.pkl")

print("\nChampion model saved as champion_model.pkl")
print("Feature list saved as model_features.pkl")