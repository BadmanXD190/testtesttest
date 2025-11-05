
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Thailand Inflation — Results Viewer", layout="wide")
st.title("Thailand Headline Inflation YoY — Results Viewer")
st.caption("Upload the historical data and the forecast CSVs to visualize the results. No model training happens here.")

st.markdown("### 1) Upload data")
col1, col2 = st.columns(2)
with col1:
    hist_file = st.file_uploader("Historical CSV (columns: Date, Inflation_YoY_pct)", type=["csv"], key="hist")
with col2:
    fcst_file = st.file_uploader("Forecast CSV (columns: Date, Forecast_YoY_pct)", type=["csv"], key="fcst")

def load_hist(f):
    df = pd.read_csv(f)
    if "Date" not in df.columns or "Inflation_YoY_pct" not in df.columns:
        raise ValueError("Historical CSV must have columns: Date, Inflation_YoY_pct")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df.rename(columns={"Inflation_YoY_pct": "Actual_YoY_pct"})[["Date","Actual_YoY_pct"]]

def load_fcst(f):
    df = pd.read_csv(f)
    if "Date" not in df.columns or "Forecast_YoY_pct" not in df.columns:
        raise ValueError("Forecast CSV must have columns: Date, Forecast_YoY_pct")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df[["Date","Forecast_YoY_pct"]]

hist_df = None
fcst_df = None
if hist_file is not None:
    try:
        hist_df = load_hist(hist_file)
        st.success(f"Historical rows: {len(hist_df)} from {hist_df['Date'].min().date()} to {hist_df['Date'].max().date()}")
    except Exception as e:
        st.error(str(e))

if fcst_file is not None:
    try:
        fcst_df = load_fcst(fcst_file)
        st.success(f"Forecast rows: {len(fcst_df)} from {fcst_df['Date'].min().date()} to {fcst_df['Date'].max().date()}")
    except Exception as e:
        st.error(str(e))

if hist_df is not None:
    st.markdown("### 2) Tables")
    if fcst_df is not None:
        combined = pd.merge(hist_df, fcst_df, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    else:
        combined = hist_df.copy()

    st.dataframe(combined, use_container_width=True)

    st.markdown("### 3) Charts")
    fig = plt.figure(figsize=(10,4))
    plt.plot(hist_df["Date"], hist_df["Actual_YoY_pct"], label="Actual")
    if fcst_df is not None:
        plt.plot(fcst_df["Date"], fcst_df["Forecast_YoY_pct"], label="Forecast")
    plt.title("Thailand Headline Inflation YoY — History and Forecast")
    plt.xlabel("Year")
    plt.ylabel("Inflation YoY percent")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    # Export combined CSV
    csv_buf = io.StringIO()
    combined.to_csv(csv_buf, index=False)
    st.download_button("Download combined CSV", data=csv_buf.getvalue(), file_name="thai_inflation_results_combined.csv", mime="text/csv")
else:
    st.info("Upload at least the historical CSV to view the table and chart. Optionally add the forecast CSV.")
