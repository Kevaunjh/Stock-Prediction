import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logging.info("Data loaded successfully. Data shape: %s", data.shape)
        
        if data.empty:
            logging.error("Loaded data is empty. Check your CSV file.")
            raise ValueError("Empty dataset loaded.")
            
        logging.info("First 5 rows of data:\n%s", data.head())
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data, lookback=60, train_split=0.8, test_cutoff=5):

    try:
        if len(data) <= lookback:
            logging.error(f"Dataset size ({len(data)}) is too small for the lookback period ({lookback})")
            raise ValueError(f"Dataset needs to be larger than the lookback period ({lookback})")

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)

        if len(data) <= lookback:
            logging.error(f"Dataset size after dropping NaN values ({len(data)}) is too small for the lookback period ({lookback})")
            raise ValueError(f"After cleaning, dataset needs to be larger than the lookback period ({lookback})")
        
        features = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        if 'composite_sentiment' in data.columns:
            features.append('composite_sentiment')
        
        df = data[features].copy()

        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df)

        logging.info(f"Scaled data shape: {scaled_data.shape}")
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, :])
            y.append(scaled_data[i, 4])
        

        if len(X) == 0:
            logging.error("No sequences were created during preprocessing. Check your lookback parameter and data size.")
            raise ValueError("No sequences were created. Dataset may be too small for the chosen lookback period.")
        

        X = np.array(X)
        y = np.array(y)
        

        logging.info(f"X shape after conversion: {X.shape}")
        logging.info(f"y shape after conversion: {y.shape}")
        

        if len(X.shape) != 3:
            logging.warning(f"X does not have the expected 3D shape. Current shape: {X.shape}")
            if len(X) > 0:
                X = X.reshape(X.shape[0], lookback, -1)
                logging.info(f"Reshaped X to: {X.shape}")
        

        if test_cutoff > 0:
            X = X[:-test_cutoff]
            y = y[:-test_cutoff]
            logging.info(f"Removed last {test_cutoff} days from dataset for evaluation")
        

        split_idx = int(len(X) * train_split)
        
        if split_idx == 0:
            logging.error(f"Train split resulted in empty training set. Adjust train_split parameter (currently {train_split}).")
            raise ValueError("Train split resulted in empty training set.")
            
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) == 0:
            logging.error("Training set is empty after splitting.")
            raise ValueError("Empty training set after splitting. Adjust train_split parameter.")
        if len(X_test) == 0:
            logging.warning("Test set is empty after splitting. Creating a minimal test set from training data.")
            if len(X_train) > 1:
                X_test = X_train[-1:].copy()
                y_test = y_train[-1:].copy()
                X_train = X_train[:-1]
                y_train = y_train[:-1]
                logging.info("Created minimal test set from training data.")
        
        logging.info(f"Data preprocessed: Train shapes X={X_train.shape}, y={y_train.shape}, Test shapes X={X_test.shape}, y={y_test.shape}")
        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def build_multi_layer_lstm(input_shape, num_layers=3, units_per_layer=None):
    if units_per_layer is None:
        units_per_layer = [128, 64, 32]
    else:
        if len(units_per_layer) != num_layers:
            units_per_layer = [units_per_layer[0]] * num_layers if len(units_per_layer) > 0 else [64] * num_layers
    
    model = Sequential()
    
    model.add(LSTM(units=units_per_layer[0], 
                  return_sequences=(num_layers > 1),
                  input_shape=input_shape))
    model.add(Dropout(0.2))
    
    for i in range(1, num_layers-1):
        model.add(LSTM(units=units_per_layer[i], return_sequences=True))
        model.add(Dropout(0.2))
    
    if num_layers > 1:
        model.add(LSTM(units=units_per_layer[-1], return_sequences=False))
        model.add(Dropout(0.2))
    
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, model_path, num_layers=3, units_per_layer=None, epochs=100, batch_size=32):

    if X_train.size == 0:
        logging.error("X_train is empty. Cannot train model.")
        raise ValueError("Empty training data.")
    
    if X_test.size == 0:
        logging.warning("X_test is empty. Creating a minimal test set from training data.")
        X_test = X_train[-1:].copy()
        y_test = y_train[-1:].copy()
        X_train = X_train[:-1]
        y_train = y_train[:-1]
    
    if len(X_train.shape) < 3:
        logging.warning(f"X_train does not have the expected 3D shape. Current shape: {X_train.shape}")
        if X_train.size > 0:
            X_train = X_train.reshape(X_train.shape[0], 1, -1)
            logging.info(f"Reshaped X_train to: {X_train.shape}")
    
    if len(X_test.shape) < 3:
        logging.warning(f"X_test does not have the expected 3D shape. Current shape: {X_test.shape}")
        if X_test.size > 0:
            X_test = X_test.reshape(X_test.shape[0], 1, -1)
            logging.info(f"Reshaped X_test to: {X_test.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    logging.info(f"Input shape for LSTM model: {input_shape}")
    
    model = build_multi_layer_lstm(input_shape, num_layers, units_per_layer)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    batch_size = min(batch_size, len(X_train))
    logging.info(f"Using batch size: {batch_size}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('./../data', exist_ok=True)
    plt.savefig('./../data/training_loss.png')
    plt.close()
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):

    if X_test.size == 0 or y_test.size == 0:
        logging.warning("Test data is empty. Skipping evaluation.")
        return None, None
    
    if len(X_test.shape) < 3:
        logging.warning(f"X_test does not have the expected 3D shape. Current shape: {X_test.shape}")
        if X_test.size > 0:
            X_test = X_test.reshape(X_test.shape[0], 1, -1)
            logging.info(f"Reshaped X_test to: {X_test.shape}")
    
    y_pred = model.predict(X_test).flatten()
    
    y_test_scaled = y_test.reshape(-1, 1)
    y_pred_scaled = y_pred.reshape(-1, 1)
    
    y_test_dummy = np.zeros((len(y_test), scaler.scale_.shape[0]))
    y_test_dummy[:, 4] = y_test 
    y_pred_dummy = np.zeros((len(y_pred), scaler.scale_.shape[0]))
    y_pred_dummy[:, 4] = y_pred
    
    y_test_actual = scaler.inverse_transform(y_test_dummy)[:, 4]
    y_pred_actual = scaler.inverse_transform(y_pred_dummy)[:, 4]
    
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    if len(y_test_actual) > 1:
        true_direction = np.sign(np.diff(np.append([y_test_actual[0]], y_test_actual)))
        pred_direction = np.sign(np.diff(np.append([y_pred_actual[0]], y_pred_actual)))
        direction_accuracy = np.mean(true_direction == pred_direction)
    else:
        logging.warning("Not enough test data points to calculate direction accuracy.")
        direction_accuracy = 0
    
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / np.maximum(0.0001, np.abs(y_test_actual)))) * 100
    
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Direction Accuracy': direction_accuracy,
        'MAPE': mape
    }
    
    predictions_df = pd.DataFrame({
        'Actual': y_test_actual,
        'Predicted': y_pred_actual,
        'Error': y_test_actual - y_pred_actual
    })
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(predictions_df)), predictions_df['Actual'], label='Actual')
    plt.plot(range(len(predictions_df)), predictions_df['Predicted'], label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Data Point')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('./../data/prediction_results.png')
    plt.close()
    
    predictions_df.to_csv('./../data/predictions.csv')
    
    logging.info("Model Evaluation Results:")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R²: {r2:.4f}")
    logging.info(f"Direction Accuracy: {direction_accuracy:.4f}")
    logging.info(f"MAPE: {mape:.4f}%")
    
    with open('./../data/model_evaluation_summary.txt', 'w') as f:
        f.write("LSTM Model Evaluation Summary\n")
        f.write("============================\n\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"Direction Accuracy: {direction_accuracy:.4f}\n")
        f.write(f"Mean Absolute Percentage Error: {mape:.4f}%\n\n")
        
        if direction_accuracy > 0.7:
            direction_quality = "excellent"
        elif direction_accuracy > 0.6:
            direction_quality = "good"
        elif direction_accuracy > 0.55:
            direction_quality = "fair"
        else:
            direction_quality = "poor"
        
        if mape < 1:
            mape_quality = "excellent"
        elif mape < 5:
            mape_quality = "good"
        elif mape < 10:
            mape_quality = "fair"
        else:
            mape_quality = "poor"
        
        f.write(f"- The model's ability to predict price direction is {direction_quality} ({direction_accuracy:.1%}).\n")
        f.write(f"- The percentage error in predictions is {mape_quality} ({mape:.2f}%).\n")
        f.write(f"- The model explains {r2:.1%} of the variance in price movements.\n")
        
        f.write("\nTrading Recommendation:\n")
        if direction_accuracy > 0.6 and mape < 10:
            f.write("This model may be suitable for trading assistance, but should be used with caution and alongside other indicators.\n")
        else:
            f.write("This model is not recommended for trading decisions without significant improvement and validation.\n")
    
    return results, predictions_df

def main(data_path, model_path, lookback=60, num_layers=3, units_per_layer=None, epochs=100, batch_size=32, train_split=0.8, test_cutoff=5):

    try:
        data = load_data(data_path)
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data, lookback, train_split, test_cutoff)
        model, history = train_lstm_model(X_train, y_train, X_test, y_test, model_path, num_layers, units_per_layer, epochs, batch_size)
        results, predictions = evaluate_model(model, X_test, y_test, scaler)
        return model, results, predictions
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a multi-layer LSTM model for stock price prediction with sentiment features")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data_with_sentiment_lstm_ready.csv", 
                        help="Path to the sentiment-augmented stock data CSV file")
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", 
                        help="Path to save the trained model")
    parser.add_argument("--lookback", type=int, default=60, help="Number of previous time steps to use for prediction")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--units", type=int, nargs='+', default=[128, 64, 32], help="Number of units in each LSTM layer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--test_cutoff", type=int, default=5, help="Number of days to cut off from the end of the dataset")
    
    args = parser.parse_args()
    
    main(data_path=args.data_path,
         model_path=args.model_path,
         lookback=args.lookback,
         num_layers=args.num_layers,
         units_per_layer=args.units,
         epochs=args.epochs,
         batch_size=args.batch_size,
         train_split=args.train_split,
         test_cutoff=args.test_cutoff)
