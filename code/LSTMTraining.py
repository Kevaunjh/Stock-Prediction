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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
tf.config.run_functions_eagerly(True)

def load_data(file_path):
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

def build_lstm_model(input_shape, existing_model_path=None):
    """Build or load an LSTM model"""
    tf.keras.backend.clear_session()
    
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
    
    if existing_model_path and os.path.exists(existing_model_path):
        try:

            temp_model = load_model(existing_model_path)
            

            dummy_input = np.zeros((1,) + input_shape)
            try:
                temp_model.predict(dummy_input, verbose=0)
                model.set_weights(temp_model.get_weights())
                logging.info(f"Successfully loaded weights from existing model at {existing_model_path}")
            except:
                logging.warning(f"Existing model has incompatible architecture - using new model instead")
        except Exception as e:
            logging.error(f"Error loading existing model: {e}")
            logging.warning("Using freshly created model")
    else:
        logging.info("No existing model found or specified - building new model")
    
    logging.info(f"LSTM model ready with input shape {input_shape}")
    return model

def evaluate_model(y_true_scaled, y_pred_scaled, scaler):
    y_true_dummy = np.zeros((len(y_true_scaled), scaler.scale_.shape[0]))
    y_pred_dummy = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
    y_true_dummy[:, 4] = y_true_scaled
    y_pred_dummy[:, 4] = y_pred_scaled
    
    y_true = scaler.inverse_transform(y_true_dummy)[:, 4]
    y_pred = scaler.inverse_transform(y_pred_dummy)[:, 4]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    true_direction = np.sign(np.diff(np.append([y_true[0]], y_true)))
    pred_direction = np.sign(np.diff(np.append([y_pred[0]], y_pred)))
    direction_accuracy = np.mean(true_direction == pred_direction)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Direction Accuracy': direction_accuracy,
        'MAPE': mape
    }

def cross_validate(X, y, scaler, lookback, model_path=None, n_splits=5, n_repeats=5):
    logging.info(f"Starting {n_splits}x{n_repeats} cross-validation...")
    
    all_metrics = []
    input_shape = (X.shape[1], X.shape[2])
    
    for repeat in range(n_repeats):
        logging.info(f"Starting repeat {repeat+1}/{n_repeats}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42+repeat)
        
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logging.info(f"Training fold {fold+1}/{n_splits}")
            
      
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            model = build_lstm_model(input_shape, model_path)
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    min_delta=1e-10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-10, verbose=1)
            ]
            
            batch_size = min(32, len(X_train))
            steps_per_epoch = None
            if len(X_train) < batch_size:
                steps_per_epoch = 10
                ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
                train_data = ds_train
                val_data = (X_test, y_test)
            else:
                train_data = (X_train, y_train)
                val_data = (X_test, y_test)

            history = model.fit(
                x=X_train,
                y=y_train,
                epochs=100,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            y_pred = model.predict(X_test).flatten()

            metrics = evaluate_model(y_test, y_pred, scaler)
            metrics['fold'] = fold + 1
            metrics['repeat'] = repeat + 1
            metrics['val_loss'] = min(history.history['val_loss'])
            fold_metrics.append(metrics)
            
            logging.info(f"Fold {fold+1} metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                        f"Direction Accuracy={metrics['Direction Accuracy']:.4f}, MAPE={metrics['MAPE']:.4f}")
        
        all_metrics.extend(fold_metrics)
    
    avg_metrics = {
        'MAE': np.mean([m['MAE'] for m in all_metrics]),
        'RMSE': np.mean([m['RMSE'] for m in all_metrics]),
        'R2': np.mean([m['R2'] for m in all_metrics]),
        'Direction Accuracy': np.mean([m['Direction Accuracy'] for m in all_metrics]),
        'MAPE': np.mean([m['MAPE'] for m in all_metrics]),
        'val_loss': np.mean([m['val_loss'] for m in all_metrics])
    }
    
    logging.info("Cross-validation complete.")
    logging.info(f"Average metrics: MAE={avg_metrics['MAE']:.4f}, RMSE={avg_metrics['RMSE']:.4f}, R2={avg_metrics['R2']:.4f}")
    logging.info(f"Direction Accuracy={avg_metrics['Direction Accuracy']:.4f}, MAPE={avg_metrics['MAPE']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(metrics_df['MAE'], bins=10)
    plt.axvline(avg_metrics['MAE'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'MAE Distribution (Avg: {avg_metrics["MAE"]:.4f})')
    
    plt.subplot(2, 2, 2)
    plt.hist(metrics_df['RMSE'], bins=10)
    plt.axvline(avg_metrics['RMSE'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'RMSE Distribution (Avg: {avg_metrics["RMSE"]:.4f})')
    
    plt.subplot(2, 2, 3)
    plt.hist(metrics_df['Direction Accuracy'], bins=10)
    plt.axvline(avg_metrics['Direction Accuracy'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Direction Accuracy Distribution (Avg: {avg_metrics["Direction Accuracy"]:.4f})')
    
    plt.subplot(2, 2, 4)
    plt.hist(metrics_df['MAPE'], bins=10)
    plt.axvline(avg_metrics['MAPE'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'MAPE Distribution (Avg: {avg_metrics["MAPE"]:.4f})')
    
    plt.tight_layout()
    plt.savefig('./../data/cv_metrics_distribution.png')
    plt.show()
    
    return avg_metrics, metrics_df

def train_final_model(X, y, scaler, model_path, lookback=60):
    """Train the final model on all data after cross-validation"""
    

    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_model(input_shape, model_path)
    
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=15,
            min_delta=1e-10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-10, verbose=1)
    ]
    
    batch_size = min(32, len(X))
    history = model.fit(
        X, y,
        epochs=150,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./../data/final_model_training.png')
    plt.show()

    model.save(model_path)
    logging.info(f"Final model saved as {model_path}")

    y_pred = model.predict(X).flatten()
    final_metrics = evaluate_model(y, y_pred, scaler)
    
    logging.info(f"Final model metrics on all data: MAE={final_metrics['MAE']:.4f}, "
                f"RMSE={final_metrics['RMSE']:.4f}, R2={final_metrics['R2']:.4f}")
    
    return model, final_metrics

def main(model_path="./../model/lstm_model.h5", data_path="./../data/stock_data.csv", lookback=10):
    try:

        
        data = load_data(data_path)
        X, y, scaler = preprocess_data(data, lookback=lookback)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if len(X) < 10:
            logging.warning("Dataset too small for cross-validation. Proceeding with simple train/test split.")
            split_index = int(0.8 * len(X))
            X_train, y_train = X[:split_index], y[:split_index]
            X_test, y_test = X[split_index:], y[split_index:]
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_lstm_model(input_shape, model_path)
            model.fit(X_train, y_train, epochs=100, batch_size=min(32, len(X_train)), validation_data=(X_test, y_test))
            model.save(model_path)
        else:
            avg_metrics, metrics_df = cross_validate(X, y, scaler, lookback, model_path)
            
            metrics_df.to_csv('./../data/cross_validation_metrics.csv', index=False)
            
            model, final_metrics = train_final_model(X, y, scaler, model_path, lookback)
            
            pd.DataFrame([final_metrics]).to_csv('./../data/final_model_metrics.csv', index=False)
            
            with open('./../data/model_performance_summary.txt', 'w') as f:
                f.write("LSTM Model Performance Summary\n")
                f.write("==============================\n\n")
                f.write(f"Cross-Validation Results (5x5):\n")
                f.write(f"Mean Absolute Error: {avg_metrics['MAE']:.4f}\n")
                f.write(f"Root Mean Squared Error: {avg_metrics['RMSE']:.4f}\n")
                f.write(f"R² Score: {avg_metrics['R2']:.4f}\n")
                f.write(f"Direction Accuracy: {avg_metrics['Direction Accuracy']:.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {avg_metrics['MAPE']:.4f}%\n\n")
                
                f.write(f"Final Model Performance:\n")
                f.write(f"Mean Absolute Error: {final_metrics['MAE']:.4f}\n")
                f.write(f"Root Mean Squared Error: {final_metrics['RMSE']:.4f}\n")
                f.write(f"R² Score: {final_metrics['R2']:.4f}\n")
                f.write(f"Direction Accuracy: {final_metrics['Direction Accuracy']:.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {final_metrics['MAPE']:.4f}%\n\n")
                
                direction_quality = "excellent" if final_metrics['Direction Accuracy'] > 0.7 else \
                                   "good" if final_metrics['Direction Accuracy'] > 0.6 else \
                                   "fair" if final_metrics['Direction Accuracy'] > 0.55 else "poor"
                                   
                mape_quality = "excellent" if final_metrics['MAPE'] < 1 else \
                              "good" if final_metrics['MAPE'] < 5 else \
                              "fair" if final_metrics['MAPE'] < 10 else "poor"
                
                f.write("Model Performance Interpretation:\n")
                f.write(f"- The model's ability to predict price direction is {direction_quality} ({final_metrics['Direction Accuracy']:.1%}).\n")
                f.write(f"- The percentage error in predictions is {mape_quality} ({final_metrics['MAPE']:.2f}%).\n")
                f.write(f"- The model explains {final_metrics['R2']:.1%} of the variance in price movements.\n")
                
                f.write("\nTrading Recommendation:\n")
                if final_metrics['Direction Accuracy'] > 0.6 and final_metrics['MAPE'] < 10:
                    f.write("This model may be suitable for trading assistance, but should be used with caution and other indicators.\n")
                else:
                    f.write("This model is not recommended for trading decisions without significant improvement and validation.\n")
                
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an LSTM model using stock data with cross-validation."
    )
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", 
                        help="Path to save/load the trained LSTM model.")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv",
                        help="Path to the stock data CSV file.")
    parser.add_argument("--lookback", type=int, default=10, 
                        help="Number of time steps to look back for each sample.")
    
    args = parser.parse_args()
    main(model_path=args.model_path, data_path=args.data_path, lookback=args.lookback)