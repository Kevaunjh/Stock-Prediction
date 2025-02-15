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
    
def MaximizeIncome(forecast):
    initial_money = 10000
    max_profit = 0
    best_buy_day = -1
    best_sell_day = -1
    
    for buy_day in range(len(forecast) - 1):
        for sell_day in range(buy_day + 1, len(forecast)):
            buy_price = forecast.iloc[buy_day]
            sell_price = forecast.iloc[sell_day]
            effective_buy_price = buy_price * 1.01
            effective_sell_price = sell_price * 0.99
            profit = effective_sell_price - effective_buy_price
            if profit > max_profit:
                max_profit = profit
                best_buy_day = buy_day
                best_sell_day = sell_day
    
    if max_profit > 0:
        buy_price = forecast.iloc[best_buy_day]
        sell_price = forecast.iloc[best_sell_day]
        print(f"Day {best_buy_day + 1}: Buy at ${buy_price:.2f}")
        print(f"Day {best_sell_day + 1}: Sell at ${sell_price:.2f}")
        print(f"Profit from this transaction: ${max_profit:.2f}")
        print(f"Your total balance after the transaction: ${initial_money + max_profit:.2f}")
    else:
        print("No possible way to benefit in money.")

            
    


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
        MaximizeIncome(forecast)
        plt.show()



    except Exception as e:
        logging.error(f"Execution failed: {e}")
