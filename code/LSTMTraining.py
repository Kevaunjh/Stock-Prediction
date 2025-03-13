import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tf.config.run_functions_eagerly(True)

def load_data(file_path):
    """
    Loads data from CSV, expecting columns:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close'] (no 'Volume').
    Indexes by 'Date' as a DateTime.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logging.info("Data loaded successfully. Data shape: %s", data.shape)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data, lookback=60):
    """
    Preprocesses data by:
      - Selecting basic features (Open, High, Low, Close, Adj Close).
      - Scaling them between 0 and 1.
      - Creating sequences of length 'lookback' for the LSTM.
    """
    try:
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)
        
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        df = data[features].copy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        
        if len(scaled_data) <= lookback:
            logging.warning("Not enough data points for the specified lookback of %d. Reducing lookback.", lookback)
            lookback = max(1, len(scaled_data) // 2)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, :])
            y.append(scaled_data[i, 4])
        
        X = np.array(X)
        y = np.array(y)
        
        logging.info("Data preprocessed: X shape=%s, y shape=%s", X.shape, y.shape)
        return X, y, scaler
    except KeyError as e:
        logging.error(f"Missing required column in data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def build_lstm_model(input_shape):
    """
    Builds an LSTM model with sufficient capacity to capture trends from basic price features.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built successfully with input shape %s.", input_shape)
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an LSTM model using stock data (no Volume) with basic features."
    )
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", 
                        help="Path to save the trained LSTM model.")
    args = parser.parse_args()

    file_path = "./../data/stock_data.csv"

    try:
        data = load_data(file_path)
        
        X, y, scaler = preprocess_data(data, lookback=60)
        
        split_index = int(0.8 * len(X))
        if split_index == 0:
            logging.warning("Train set is empty after splitting. Using all data for training.")
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, y_train = X[:split_index], y[:split_index]
            X_test, y_test = X[split_index:], y[split_index:]
        
        batch_size = 32
        if len(X_train) < batch_size:
            steps_per_epoch = 10
            ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
            logging.info("Training set is small; using tf.data dataset with %d steps per epoch.", steps_per_epoch)
            train_data = ds_train
        else:
            steps_per_epoch = None
            train_data = (X_train, y_train)
        
        if os.path.exists(args.model_path):
            logging.info("Loading existing model for further training.")
            model = load_model(args.model_path)
            model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            logging.info("No existing model found. Creating a new LSTM model.")
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                min_delta=1e-10,  
                restore_best_weights=True, 
                verbose=1
            ),
            ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-10, verbose=1)
        ]
        

        history = model.fit(
            train_data,
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size if steps_per_epoch is None else None,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
        model.save(args.model_path)
        logging.info(f"Model saved as {args.model_path}")
    
    except Exception as e:
        logging.error(f"Execution failed: {e}")
