import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Sales Forecast App", layout="wide")

# ---------------------------
# Load Model & Data
# ---------------------------
model = joblib.load("champion_model.pkl")
feature_cols = joblib.load("model_features.pkl")

df = pd.read_csv("data/timeseries_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ---------------------------
# Title
# ---------------------------
st.title("📊 Retail Sales Forecasting App")

st.write(
    "Diese App hilft dabei, zukünftige Verkäufe vorherzusagen, "
    "um bessere Entscheidungen für Lager, Personal und Aktionen zu treffen."
)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Einstellungen")

forecast_days = st.sidebar.slider(
    "Forecast-Tage",
    min_value=7,
    max_value=60,
    value=30
)

# ---------------------------
# Data Overview
# ---------------------------
st.subheader("Historische Verkäufe")

daily_sales = df.groupby("date")["unit_sales"].sum().reset_index()

fig, ax = plt.subplots()
ax.plot(daily_sales["date"], daily_sales["unit_sales"])
ax.set_title("Sales Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
st.pyplot(fig)

# ---------------------------
# Forecast Logic
# ---------------------------
history = daily_sales.copy()
last_oil = df["dcoilwtico"].ffill().bfill().iloc[-1]

predictions = []

for i in range(forecast_days):
    next_date = history["date"].max() + pd.Timedelta(days=1)

    lag_1 = history["unit_sales"].iloc[-1]
    lag_7 = history["unit_sales"].iloc[-7] if len(history) >= 7 else lag_1

    rolling_mean_7 = history["unit_sales"].tail(7).mean()
    rolling_std_7 = history["unit_sales"].tail(7).std()

    row = pd.DataFrame([{
        "year": next_date.year,
        "month": next_date.month,
        "day_of_week": next_date.dayofweek,
        "day_of_month": next_date.day,
        "is_weekend": int(next_date.dayofweek in [5, 6]),
        "is_holiday": 0,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "dcoilwtico": last_oil
    }])

    row = row[feature_cols]

    pred = model.predict(row)[0]

    predictions.append({
        "date": next_date,
        "forecast": pred
    })

    history = pd.concat([
        history,
        pd.DataFrame([{"date": next_date, "unit_sales": pred}])
    ], ignore_index=True)

forecast_df = pd.DataFrame(predictions)

# ---------------------------
# Show Forecast
# ---------------------------
st.subheader("Forecast")

st.dataframe(forecast_df)

fig2, ax2 = plt.subplots()
ax2.plot(daily_sales["date"], daily_sales["unit_sales"], label="History")
ax2.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast")
ax2.legend()
ax2.set_title("Sales Forecast")
st.pyplot(fig2)

# ---------------------------
# Business Explanation
# ---------------------------
st.subheader("Business Nutzen")

st.write(
    "Diese Prognosen helfen dabei, zukünftige Nachfrage besser einzuschätzen. "
    "So können Lagerbestände optimiert, Personal effizient geplant und Umsätze gesteigert werden."
)