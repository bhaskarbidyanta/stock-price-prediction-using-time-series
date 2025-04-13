import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

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
