import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Title
st.title("📈 Netflix Stock Price Forecast (Prophet)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Netflix_stock_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Netflix Stock Data")
    st.dataframe(df)

# Preprocess
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Forecast period
period = st.slider("Forecast days into the future", 15, 120, 30)

# Train Prophet model
model = Prophet(seasonality_mode='multiplicative')
model.fit(df_prophet)

# Future prediction
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Plot forecast
st.subheader(f"Forecast for next {period} days")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Show forecast table
if st.checkbox("Show forecast data"):
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period))