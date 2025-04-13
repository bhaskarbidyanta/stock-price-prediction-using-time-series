import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("RELIANCE.NS", "TATAMOTORS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "HINDUNILVR.NS", "LT.NS", "HDFC.NS", "SBIN.NS")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:",1, 4)
period = n_years * 365
