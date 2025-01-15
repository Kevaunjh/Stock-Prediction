import argparse
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Price'], index_col='Price')
    data = data[['Adj Close']].dropna()
    return data

# Predict future values
def predict_future(data, start_date, end_date):
    # Load the saved model
    model_path = "./../model/arima_model.pkl"
    model = ARIMAResults.load(model_path)

    # Generate predictions
    forecast = model.predict(start=start_date, end=end_date, dynamic=True)

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Historical Data')
    plt.plot(forecast, label='Forecast', color='red')
    plt.title('ARIMA Model Forecast')
    plt.legend()
    plt.show()

    return forecast

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict stock prices using a trained ARIMA model.")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv", help="Path to the stock data CSV file.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for prediction (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="End date for prediction (YYYY-MM-DD).")
    args = parser.parse_args()

    # Load data
    data = load_data(args.data_path)

    # Predict and plot
    forecast = predict_future(data, args.start_date, args.end_date)

    # Print forecast
    print("Forecasted values:")
    print(forecast)
