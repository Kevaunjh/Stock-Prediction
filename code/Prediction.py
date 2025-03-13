import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):

    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        data.index = pd.to_datetime(data.index)
        data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']].dropna()
        logging.info("Data loaded successfully. Data shape: %s", data.shape)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def predict_future(data, start_date, end_date, model_path, lookback=60):

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = load_model(model_path)
        

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        if len(scaled_data) < lookback:
            pad_length = lookback - len(scaled_data)
            pad_array = np.repeat(scaled_data[0:1], pad_length, axis=0)
            scaled_data = np.concatenate([pad_array, scaled_data], axis=0)
            logging.info("Data padded: original length %d, padded to %d", len(data), len(scaled_data))
        
        last_lookback = scaled_data[-lookback:] 
        predictions = []
        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        for _ in range(len(future_dates)):
            input_data = last_lookback.reshape((1, lookback, scaled_data.shape[1]))
            predicted_scaled = model.predict(input_data)[0, 0]
            predictions.append(predicted_scaled)
            new_row = last_lookback[-1].copy()
            new_row[4] = predicted_scaled
            last_lookback = np.concatenate([last_lookback[1:], new_row.reshape(1, -1)], axis=0)
        
        target_min = scaler.data_min_[4]
        target_max = scaler.data_max_[4]
        forecast = np.array(predictions).reshape(-1, 1) * (target_max - target_min) + target_min
        
        forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted Price'])
        
        last_month_date = data.index[-1] - pd.DateOffset(months=1)
        last_month_data = data.loc[data.index >= last_month_date]
        
        plt.figure(figsize=(12,6))
        plt.plot(last_month_data.index, last_month_data['Adj Close'], label='Historical Data (Last Month)', color='blue')
        plt.plot(forecast_df.index, forecast_df['Predicted Price'], label='Forecast', color='red')
        
        min_price = forecast_df['Predicted Price'].min()
        min_date = forecast_df['Predicted Price'].idxmin()
        plt.annotate('Min', xy=(min_date, min_price), xytext=(min_date, min_price * 0.95),
                     arrowprops=dict(arrowstyle="->", color='green'),
                     horizontalalignment='center', verticalalignment='bottom')
        
        max_price = forecast_df['Predicted Price'].max()
        max_date = forecast_df['Predicted Price'].idxmax()
        plt.annotate('Max', xy=(max_date, max_price), xytext=(max_date, max_price * 1.05),
                     arrowprops=dict(arrowstyle="->", color='red'),
                     horizontalalignment='center', verticalalignment='top')
        
        plt.title('LSTM Model Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        return forecast_df
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise

def MaximizeIncome(forecast):

    initial_money = 10000
    max_profit = 0
    best_buy_day = -1
    best_sell_day = -1
    
    for buy_day in range(len(forecast) - 1):
        for sell_day in range(buy_day + 1, len(forecast)):
            buy_price = forecast.iloc[buy_day]["Predicted Price"]
            sell_price = forecast.iloc[sell_day]["Predicted Price"]
            effective_buy_price = buy_price * 1.01
            effective_sell_price = sell_price * 0.99
            profit = effective_sell_price - effective_buy_price
            if profit > max_profit:
                max_profit = profit
                best_buy_day = buy_day
                best_sell_day = sell_day
    
    if max_profit > 0:
        print(f"Day {best_buy_day + 1}: Buy at ${forecast.iloc[best_buy_day]['Predicted Price']:.2f}")
        print(f"Day {best_sell_day + 1}: Sell at ${forecast.iloc[best_sell_day]['Predicted Price']:.2f}")
        print(f"Profit from transaction: ${max_profit:.2f}")
        print(f"Total balance after transaction: ${initial_money + max_profit:.2f}")
    else:
        print("No possible way to benefit in money.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict stock prices using a trained LSTM model.")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv", help="Path to the stock data CSV file.")
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", help="Path to the trained LSTM model file.")
    args = parser.parse_args()
    
    try:
        data = load_data(args.data_path)
        last_date = data.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
        start_date = future_dates[0].strftime('%Y-%m-%d')
        end_date = future_dates[-1].strftime('%Y-%m-%d')
        logging.info("Forecasting from %s to %s", start_date, end_date)
        
        forecast = predict_future(data, start_date, end_date, args.model_path)
        print("Forecasted values:")
        print(forecast)
        MaximizeIncome(forecast)
    except Exception as e:
        logging.error("Execution failed: %s", e)
