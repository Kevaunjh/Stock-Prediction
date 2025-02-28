import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tf.config.run_functions_eagerly(True) 

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data, lookback=60):
    try:
        data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
        data = data[['Adj Close']].dropna()
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        logging.info("Data preprocessed successfully.")
        return X, y, scaler
    except KeyError as e:
        logging.error(f"Missing required column in data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built successfully.")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an LSTM model using stock data.")
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", help="Path to save the trained LSTM model.")
    args = parser.parse_args()

    file_path = "./../data/stock_data.csv"

    try:
        data = load_data(file_path)
        X, y, scaler = preprocess_data(data)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if os.path.exists(args.model_path):
            logging.info("Loading existing model for further training.")
            model = load_model(args.model_path)
            model.compile(optimizer='adam', loss='mean_squared_error')  
        else:
            logging.info("No existing model found. Creating a new LSTM model.")
            model = build_lstm_model((X_train.shape[1], 1))
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        
        model.save(args.model_path)
        logging.info(f"Model saved as {args.model_path}")
    
    except Exception as e:
        logging.error(f"Execution failed: {e}")
