import argparse
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date') 
        data = data[['Adj Close']].dropna()
        logging.info("Data loaded and preprocessed successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

def predict_future(data, start_date, end_date, model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = ARIMAResults.load(model_path)
        last_date = data.index[-1]  
        if pd.to_datetime(start_date) <= last_date:
            raise ValueError(f"Start date {start_date} must be after the last date in the training data ({last_date}).")

        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')  

        forecast = model.predict(start=len(data), end=len(data) + len(future_dates) - 1, dynamic=True)
        forecast.index = future_dates 
        logging.info("Now plotting the expected future values of TSLA")


        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Historical Data')
        plt.plot(forecast, label='Forecast', color='red')
        plt.title('ARIMA Model Forecast')
        plt.legend()

        return forecast
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock prices using a trained ARIMA model.")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv", help="Path to the stock data CSV file.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for prediction (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date for prediction (YYYY-MM-DD).")
    parser.add_argument("--model_path", type=str, default="./../model/arima_model.pkl", help="Path to the trained ARIMA model file.")
    args = parser.parse_args()

    try:
        data = load_data(args.data_path)
        forecast = predict_future(data, args.start_date, args.end_date, args.model_path)
        print("Forecasted values:")
        print(forecast)
        plt.show()

    except Exception as e:
        logging.error(f"Execution failed: {e}")
