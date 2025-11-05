
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thailand Inflation LSTM (Annual)", layout="wide")

st.title("Thailand Headline Inflation YoY — Annual LSTM Forecaster")
st.caption("Upload your thai_headline_inflation_yoy_annual.csv or use the sample format. The app trains an LSTM, evaluates on a holdout split, and forecasts future years.")

DEFAULT_WINDOW = 5
DEFAULT_EPOCHS = 300
DEFAULT_BATCH = 4
DEFAULT_HORIZON = 3

with st.sidebar:
    st.header("Model settings")
    window = st.number_input("Look-back window in years", min_value=2, max_value=20, value=DEFAULT_WINDOW, step=1)
    epochs = st.number_input("Training epochs", min_value=50, max_value=2000, value=DEFAULT_EPOCHS, step=50)
    batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=DEFAULT_BATCH, step=1)
    horizon = st.number_input("Forecast years", min_value=1, max_value=20, value=DEFAULT_HORIZON, step=1)
    test_ratio = st.slider("Test split ratio", 0.05, 0.5, 0.2, 0.05)
    st.markdown("---")
    shuffle_train = st.checkbox("Shuffle training batches", value=False)
    dropout_rate = st.slider("Dropout rate", 0.0, 0.8, 0.2, 0.05)

st.subheader("Upload data")
uploaded = st.file_uploader("CSV with columns: Date, Inflation_YoY_pct", type=["csv"])

sample = pd.DataFrame({
    "Date": pd.to_datetime(["2015-12-01","2016-12-01","2017-12-01","2018-12-01","2019-12-01","2020-12-01","2021-12-01","2022-12-01","2023-12-01"]),
    "Inflation_YoY_pct": [ -0.9, 0.2, 0.7, 1.1, 0.7, -0.8, 1.2, 6.1, 1.2 ]
})
st.markdown("If you do not have a file, you can download a sample template below.")

csv_buf = io.StringIO()
sample.to_csv(csv_buf, index=False)
st.download_button("Download sample template CSV", data=csv_buf.getvalue(), file_name="thai_headline_inflation_yoy_annual_sample.csv", mime="text/csv")

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def load_dataframe(file_like):
    df = pd.read_csv(file_like)
    if "Date" not in df.columns or "Inflation_YoY_pct" not in df.columns:
        raise ValueError("CSV must have Date and Inflation_YoY_pct columns")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Inflation_YoY_pct"]).copy()
    df["Inflation_YoY_pct"] = pd.to_numeric(df["Inflation_YoY_pct"], errors="coerce")
    df = df.dropna(subset=["Inflation_YoY_pct"])
    return df

# Data section
if uploaded is not None:
    df = load_dataframe(uploaded)
    st.success(f"Loaded {len(df)} rows from upload. Range {df['Date'].min().date()} to {df['Date'].max().date()}")
else:
    st.info("No file uploaded. Using the sample template in-memory so you can test the app.")
    df = sample.copy()

series = df.set_index("Date")["Inflation_YoY_pct"].astype(float)

if len(series) < max(10, window + 3):
    st.warning("Not enough data for the selected window. Add more annual points or reduce the window length.")
    st.stop()

scaler = MinMaxScaler((0, 1))
scaled = scaler.fit_transform(series.values.reshape(-1, 1))

X, y = create_sequences(scaled, window)

# chronological split
split = int(len(X) * (1 - test_ratio))
if split <= 0 or split >= len(X):
    st.error("Invalid split. Adjust the test split ratio.")
    st.stop()

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

with st.expander("See training set sizes"):
    st.write(f"Train sequences {X_train.shape[0]}")
    st.write(f"Test sequences {X_test.shape[0]}")

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window, 1)),
    Dropout(dropout_rate),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

train_btn = st.button("Train model")

if train_btn:
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=(X_test, y_test),
        shuffle=shuffle_train,
        verbose=0
    )
    dur = time.time() - start
    st.success(f"Training complete in {dur:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    st.metric("Test RMSE", f"{rmse:.3f}")
    st.metric("Test MAE", f"{mae:.3f}")

    # Plot test set predictions
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.title("Test window — Actual vs Predicted")
    plt.xlabel("Test time steps")
    plt.ylabel("Inflation YoY percent")
    plt.legend()
    st.pyplot(fig1, clear_figure=True)

    # Forecast future
    last_seq = scaled[-window:].astype(np.float32).copy().reshape(window, 1)
    future_scaled = []
    for _ in range(int(horizon)):
        p = model.predict(last_seq.reshape(1, window, 1), verbose=0)[0, 0]
        future_scaled.append(p)
        # update the rolling window
        last_seq = np.vstack([last_seq[1:], [[p]]]).reshape(window, 1)

    future_vals = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    last_year = series.index.max().year
    future_years = [pd.Timestamp(f"{y}-12-01") for y in range(last_year + 1, last_year + 1 + len(future_vals))]
    forecast_df = pd.DataFrame({"Date": future_years, "Forecast_YoY_pct": future_vals})

    # Combined table
    hist_df = df.rename(columns={"Inflation_YoY_pct": "Actual_YoY_pct"})[["Date", "Actual_YoY_pct"]]
    combined = hist_df.merge(forecast_df, how="outer", on="Date")
    combined_display = combined.sort_values("Date").reset_index(drop=True)

    st.subheader("History and Forecast table")
    st.dataframe(combined_display, use_container_width=True)

    # Plot history + forecast line
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(hist_df["Date"], hist_df["Actual_YoY_pct"], label="Actual")
    plt.plot(forecast_df["Date"], forecast_df["Forecast_YoY_pct"], label="Forecast")
    plt.title("Thailand Headline Inflation YoY — History and forecast")
    plt.xlabel("Year")
    plt.ylabel("Inflation YoY percent")
    plt.legend()
    st.pyplot(fig2, clear_figure=True)

    # Allow download of forecast as CSV
    st.download_button(
        "Download forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name="annual_lstm_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Set your parameters then click Train model to run training and see results.")

