import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    try:
        data = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            index_col='Date'
        )
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
        data = data[['Adj Close']].dropna()
        logging.info("Data preprocessed successfully.")
        return data
    except KeyError as e:
        logging.error(f"Missing required column in data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_arima(data, order):
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        logging.info("ARIMA model trained successfully.")
        return fitted_model
    except Exception as e:
        logging.error(f"Error training ARIMA model: {e}")
        raise
    
def plot_predictions(data, fitted_model, save_path=None):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Actual')
        plt.plot(fitted_model.fittedvalues, label='Fitted', color='red')
        plt.title('ARIMA Model - Actual vs Fitted')
        plt.legend()
        if save_path:
            os.makedirs(save_path, exist_ok=True) 
            plot_file = os.path.join(save_path, "arima_plot.png")
            plt.savefig(plot_file)
            logging.info(f"Plot saved as {plot_file}")
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting predictions: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an ARIMA model using stock data.")
    parser.add_argument("--model_path", type=str, default="./../model/arima_model.pkl", help="Path to save the trained ARIMA model.")
    parser.add_argument("--plot_path", type=str, default="./../plots/", help="Path to save the plot (optional).")
    parser.add_argument("--order", type=int, nargs=3, default=(5, 1, 0), help="ARIMA order (p, d, q). Example: 5 1 0")
    args = parser.parse_args()

    file_path = "./../data/stock_data.csv"

    try:
        data = load_data(file_path)
        processed_data = preprocess_data(data)

        order = tuple(args.order)
        fitted_model = train_arima(processed_data, order)

        print(fitted_model.summary())
        
        plot_predictions(processed_data, fitted_model, save_path=args.plot_path)

        model_dir = os.path.dirname(args.model_path)
        os.makedirs(model_dir, exist_ok=True)

        fitted_model.save(args.model_path)
        logging.info(f"Model saved as {args.model_path}")

    except Exception as e:
        logging.error(f"Execution failed: {e}")
