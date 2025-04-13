import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Streamlit UI Setup ---
st.set_page_config(page_title="📉 Stock Viewer", layout="wide")
st.title("📊 Stock Price Visualizer with Matplotlib")

# --- Sidebar Inputs ---
st.sidebar.header("Choose Your Options")
stocks = (
    "RELIANCE.NS", "TATAMOTORS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "LT.NS", "HDFC.NS", "SBIN.NS"
)
ticker = st.sidebar.selectbox("Select a Stock", stocks)
start = st.sidebar.date_input("Start Date", date(2015, 1, 1))
end = st.sidebar.date_input("End Date", date.today())

n_years = start.year - end.year
period = n_years * 365
#if period <= 0:
#    st.sidebar.error("End date must be after start date.")
#    st.stop()

# --- Download Data ---
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = load_data(ticker, start, end)

# --- Validate Data ---
if df.empty or 'Open' not in df.columns or 'Close' not in df.columns:
    st.error("📛 Could not fetch valid data. Try different dates or stock.")
    st.stop()

# --- Display Raw Data ---
st.subheader("📅 Recent Stock Data")
st.dataframe(df.tail())

# --- Plotting with Matplotlib ---
st.subheader(f"📉 {ticker} Stock Prices")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Date'], df['Open'], label='Open', color='blue', linewidth=1.5)
ax.plot(df['Date'], df['Close'], label='Close', color='red', linewidth=1.5)
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title(f"{ticker} - Open & Close Prices", fontsize=14)
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)

st.write("🔍 Columns in your dataset:")
st.write(df.columns.tolist())

# 🧼 Clean header if needed
if df.columns[0] == 0 or "" in df.columns:
    df.columns = df.iloc[0]  # Set first row as header
    df = df[1:]              # Drop the first row
    df.reset_index(drop=True, inplace=True)

# 🧽 Strip all column names (whitespace messes things up)
# 🛠 Flatten MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([str(i).strip() for i in col if i]) for col in df.columns.values]
else:
    df.columns = df.columns.astype(str).str.strip()

st.write("🔍 First few rows of the uploaded data:")
st.write(df.head())
