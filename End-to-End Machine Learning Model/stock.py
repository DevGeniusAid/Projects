from autots import AutoTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import streamlit as st

# Set plot style and configurations
sns.set()
sns.set(style='whitegrid') 

# Streamlit application title and user input
st.title("Future Price Prediction Model")
df = st.text_input("Let's Predict the Future Prices: ")

if df == "Nasdaq":
    # Load and preprocess data
    data = pd.read_csv("Nasdaq.csv")
    data.dropna(inplace=True)

    # Train AutoTS model
    model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
    model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

    # Generate and display forecast
    prediction = model.predict()
    forecast = prediction.forecast
    st.write(forecast)
