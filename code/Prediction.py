import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime

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

def load_metrics():
    try:
        metrics_path = './../data/final_model_metrics.csv'
        cv_metrics_path = './../data/cross_validation_metrics.csv'
        
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path).to_dict('records')[0]
            logging.info("Model metrics loaded successfully")
        else:
            metrics = {"Direction Accuracy": None, "MAPE": None, "R2": None}
            logging.warning("No model metrics found")
            
        if os.path.exists(cv_metrics_path):
            cv_metrics = pd.read_csv(cv_metrics_path)
            cv_avg = cv_metrics.mean().to_dict()
            logging.info("Cross-validation metrics loaded successfully")
        else:
            cv_avg = {}
            logging.warning("No cross-validation metrics found")
            
        return metrics, cv_avg
    except Exception as e:
        logging.error(f"Error loading metrics: {e}")
        return {}, {}

def calculate_recent_trend(data, window=5):
    if data is None or len(data) == 0:
        return 0
        
    if len(data) < window:
        window = len(data)
    
    recent_data = data['Adj Close'].iloc[-window:]

    x = np.arange(len(recent_data))
    y = recent_data.values

    if len(x) > 1: 
        slope, _ = np.polyfit(x, y, 1)
        
        avg_price = np.mean(recent_data)
        normalized_slope = slope / avg_price * 100 
        

        trend_strength = max(min(normalized_slope, 1), -1)
    else:
        trend_strength = 0
        
    return trend_strength

def apply_adaptive_noise(base_prediction, trend_strength, confidence, volatility):
    noise_scale = volatility * (1 - confidence)
    
    directional_bias = trend_strength * confidence
    
    noise = np.random.normal(directional_bias, noise_scale)
    
    return base_prediction * (1 + noise/100) 

def calculate_volatility(data, window=10):
    if len(data) < 2:
        return 0.01 
        
    returns = data['Adj Close'].pct_change().dropna()
    
    if len(returns) < window:
        window = len(returns)
        
    recent_returns = returns.iloc[-window:]
    volatility = recent_returns.std() * 100 
    
    return max(volatility, 0.01)  

def predict_future(data, start_date, end_date, model_path, lookback=10):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        if data is None or len(data) == 0:
            raise ValueError("Empty dataset provided for prediction")
        
        future_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        if len(future_dates) == 0:
            logging.warning("No business days in prediction range")
            return pd.DataFrame(columns=['Predicted Price', 'MAPE', 'Lower Bound', 'Upper Bound'])
            
        model = load_model(model_path)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        if len(scaled_data) < lookback:
            lookback = max(1, len(scaled_data) - 1)
            logging.warning(f"Lookback window reduced to {lookback} due to limited data")
        
        metrics, _ = load_metrics()
        direction_confidence = metrics.get('Direction Accuracy', 0.55) 
        
        trend_strength = calculate_recent_trend(data)
        volatility = calculate_volatility(data)
        logging.info(f"Recent trend strength: {trend_strength:.4f}, Volatility: {volatility:.4f}%")
        
        if len(scaled_data) < lookback:
            logging.error(f"Not enough data for lookback of {lookback}, only {len(scaled_data)} points available")
            lookback = len(scaled_data) - 1
            if lookback < 1:
                raise ValueError("Not enough data points for prediction")
        
        last_lookback = scaled_data[-lookback:]
        
        current_lookback = last_lookback.copy()
        base_predictions = []
        
        for _ in range(len(future_dates)):
            input_data = current_lookback.reshape((1, lookback, scaled_data.shape[1]))
            predicted_scaled = model.predict(input_data, verbose=0)[0, 0]
            base_predictions.append(predicted_scaled)
            
            new_row = current_lookback[-1].copy()
            new_row[4] = predicted_scaled  
            current_lookback = np.concatenate([current_lookback[1:], new_row.reshape(1, -1)], axis=0)
        
        if not base_predictions:
            logging.error("No predictions generated")
            return pd.DataFrame(columns=['Predicted Price', 'MAPE', 'Lower Bound', 'Upper Bound'])
        

        target_min = scaler.data_min_[4]
        target_max = scaler.data_max_[4]
        base_pred_prices = np.array(base_predictions).reshape(-1, 1) * (target_max - target_min) + target_min
        
        current_lookback = last_lookback.copy()
        adjusted_predictions = []
        
        last_actual_price = (current_lookback[-1, 4] * (target_max - target_min) + target_min)
        current_price = last_actual_price
        
        for i in range(len(future_dates)):
            if i > 0 and len(adjusted_predictions) >= 2:
                recent_pred_trend = np.sign(adjusted_predictions[-1] - adjusted_predictions[-2])
                blend_factor = min(i / 3, 0.8)  
                trend_strength = trend_strength * (1 - blend_factor) + recent_pred_trend * blend_factor
            
            input_data = current_lookback.reshape((1, lookback, scaled_data.shape[1]))
            raw_predicted_scaled = model.predict(input_data, verbose=0)[0, 0]
            raw_predicted_price = raw_predicted_scaled * (target_max - target_min) + target_min
            if i == 0:
                base_change_pct = (raw_predicted_price - current_price) / current_price
            else:
                base_change_pct = (raw_predicted_price - adjusted_predictions[-1]) / adjusted_predictions[-1]
            
            confidence_factor = direction_confidence
            
            adjusted_change_pct = base_change_pct * (1 + trend_strength * confidence_factor)
            
            adjusted_change_pct = apply_adaptive_noise(
                adjusted_change_pct, 
                trend_strength, 
                confidence_factor,
                volatility
            )
            
            if i == 0:
                new_price = current_price * (1 + adjusted_change_pct)
            else:
                new_price = adjusted_predictions[-1] * (1 + adjusted_change_pct)
            
            adjusted_predictions.append(new_price)
            
            new_scaled_price = (new_price - target_min) / (target_max - target_min)
            new_row = current_lookback[-1].copy()
            new_row[4] = new_scaled_price
            current_lookback = np.concatenate([current_lookback[1:], new_row.reshape(1, -1)], axis=0)

        if not adjusted_predictions:
            logging.error("No adjusted predictions generated")
            return pd.DataFrame(columns=['Predicted Price', 'MAPE', 'Lower Bound', 'Upper Bound'])
        
        forecast_df = pd.DataFrame(adjusted_predictions, index=future_dates, columns=['Predicted Price'])
        
        mape = metrics.get('MAPE', 5.0)  
        confidence_shrink_factor = 0.2
        intervals = []
        for i in range(len(future_dates)):
            time_factor = 1 + (i / len(future_dates)) * 0.5  
            interval = mape * time_factor * confidence_shrink_factor
            intervals.append(interval)
        
        forecast_df['MAPE'] = intervals
        forecast_df['Lower Bound'] = forecast_df['Predicted Price'] * (1 - forecast_df['MAPE']/100)
        forecast_df['Upper Bound'] = forecast_df['Predicted Price'] * (1 + forecast_df['MAPE']/100)
        
        return forecast_df
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        raise

def simulate_trading(sim_df, initial_cash=10000):

    if sim_df is None or len(sim_df) < 2:
        logging.error("Not enough data for simulation")
        return pd.DataFrame() 
    
    cash = initial_cash
    shares = 0
    log = []  
    
    dates = sim_df.index
    prices = sim_df['Predicted Price'].values

    portfolio_value = cash 
    log.append((dates[0], "Start", prices[0], shares, cash, portfolio_value))
    

    has_bought = False
    has_sold = False
    
    for i in range(1, len(prices)):
        prev_price = prices[i - 1]
        curr_price = prices[i]
        action = "Hold"


        if curr_price < prev_price:
            if cash > 0:
                shares = cash / curr_price
                cash = 0
                action = "Buy"
                has_bought = True
        elif curr_price > prev_price:
            if shares > 0:
                cash = shares * curr_price 
                shares = 0
                action = "Sell"
                has_sold = True
        
        portfolio_value = cash + shares * curr_price
        log.append((dates[i], action, curr_price, shares, cash, portfolio_value))
    
 
    if not has_bought and cash > 0 and len(prices) > 2:

        buy_idx = np.argmin(prices[1:-1]) + 1  
        
        date, _, price, _, _, _ = log[buy_idx]
        shares = cash / price
        cash = 0
        portfolio_value = shares * price
        log[buy_idx] = (date, "Buy", price, shares, cash, portfolio_value)
        has_bought = True
        
        for i in range(buy_idx + 1, len(log)):
            date, action, price, _, _, _ = log[i]
            portfolio_value = cash + shares * price
            log[i] = (date, action, price, shares, cash, portfolio_value)
    
    if has_bought and not has_sold and shares > 0 and len(prices) > 2:
        buy_indices = [i for i, entry in enumerate(log) if entry[1] == "Buy"]
        if buy_indices:
            last_buy_idx = max(buy_indices)
            if last_buy_idx < len(log) - 1:
                remaining_prices = [log[i][2] for i in range(last_buy_idx + 1, len(log))]
                if remaining_prices:
                    best_sell_idx = np.argmax(remaining_prices) + last_buy_idx + 1
                    
                    date, _, price, _, _, _ = log[best_sell_idx]
                    cash = shares * price
                    shares = 0
                    portfolio_value = cash
                    log[best_sell_idx] = (date, "Sell", price, shares, cash, portfolio_value)
                    
                    for i in range(best_sell_idx + 1, len(log)):
                        date, action, price, _, _, _ = log[i]
                        portfolio_value = cash + shares * price
                        log[i] = (date, action, price, shares, cash, portfolio_value)
    
    if shares > 0:
        final_price = prices[-1]
        cash = shares * final_price
        shares = 0
        portfolio_value = cash
        log.append((dates[-1], "Final Sell", final_price, shares, cash, portfolio_value))
    
    if not log:
        logging.error("No trading activity recorded")
        return pd.DataFrame(columns=["Date", "Action", "Price", "Shares", "Cash", "Portfolio Value"])
    
    simulation_df = pd.DataFrame(log, columns=["Date", "Action", "Price", "Shares", "Cash", "Portfolio Value"])
    simulation_df.set_index("Date", inplace=True)
    return simulation_df

def main(data_path="./../data/stock_data.csv",
         model_path="./../model/lstm_model.h5",
         days_to_predict=7,
         lookback=10):
    try:
        data = load_data(data_path)
        
        if data is None or len(data) == 0:
            logging.error("No data available for prediction")
            return None, None
        
        last_actual = data['Adj Close'].iloc[-1]
        
        today = pd.Timestamp.today().normalize()
        forecast_start = today + pd.DateOffset(days=1)
        forecast_end = forecast_start + pd.DateOffset(days=days_to_predict - 1)

        forecast_df = predict_future(data, forecast_start, forecast_end, model_path, lookback)
        
        if forecast_df.empty:
            logging.error("No forecast generated")
            return None, None
        
        today_row = pd.DataFrame({
            'Predicted Price': [last_actual],
            'Lower Bound': [last_actual],
            'Upper Bound': [last_actual],
            'MAPE': [0] 
        }, index=[today])
        sim_df = pd.concat([today_row, forecast_df])
        
        trading_log = simulate_trading(sim_df, initial_cash=10000)
        
        if trading_log.empty:
            logging.error("No trading simulation results")
            return sim_df, None
        
        last_month_date = data.index[-1] - pd.DateOffset(months=1)
        historical = data.loc[data.index >= last_month_date]
        if today not in historical.index:
            historical = pd.concat([historical, pd.DataFrame({'Adj Close': [last_actual]}, index=[today])])

        plt.figure(figsize=(14, 8))
        
        plt.plot(historical.index, historical['Adj Close'], label='Historical Data', color='blue')
        
        plt.plot(sim_df.index, sim_df['Predicted Price'], label='Forecast', color='red', linestyle='-')
        
        plt.fill_between(sim_df.index,
                        sim_df['Lower Bound'],
                        sim_df['Upper Bound'],
                        color='red', alpha=0.2,
                        label='Confidence Interval')
        
        plt.axvline(x=today, color='black', linestyle='--', alpha=0.7, label='Today')
        
        buy_plotted = False
        sell_plotted = False
        
        for idx, row in trading_log.iterrows():
            if row['Action'] == 'Buy' and not buy_plotted:
                plt.scatter(idx, row['Price'], marker='^', color='green', s=100, label='Buy')
                buy_plotted = True
            elif row['Action'] == 'Buy':
                plt.scatter(idx, row['Price'], marker='^', color='green', s=100)
            elif (row['Action'] == 'Sell' or row['Action'] == 'Final Sell') and not sell_plotted:
                plt.scatter(idx, row['Price'], marker='v', color='purple', s=100, label='Sell')
                sell_plotted = True
            elif row['Action'] == 'Sell' or row['Action'] == 'Final Sell':
                plt.scatter(idx, row['Price'], marker='v', color='purple', s=100)
        
        plt.title('TSLA Stock Price Forecast & Trading Simulation')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./../data/forecast_simulation_plot.png')
        plt.show()

        print("\n======== TRADING SIMULATION LOG ========")
        for dt, row in trading_log.iterrows():
            print(f"{dt.date()}: Action={row['Action']}, Price=${row['Price']:.2f}, "
                  f"Cash=${row['Cash']:.2f}, Shares={row['Shares']:.4f}, "
                  f"Portfolio=${row['Portfolio Value']:.2f}")
        print("========================================\n")
        
        final_portfolio = trading_log.iloc[-1]['Portfolio Value']
        profit = final_portfolio - 10000
        print(f"Final Portfolio Value: ${final_portfolio:.2f} (Profit: ${profit:.2f})")
        
        return sim_df, trading_log
    except Exception as e:
        logging.error(f"Prediction execution failed: {e}")
        print(f"\nERROR: {e}")
        print("Please check the logs for details.")
        return None, None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict future stock prices with direction-based forecast and simulate trading decisions."
    )
    parser.add_argument("--model_path", type=str, default="./../model/lstm_model.h5", 
                        help="Path to the trained LSTM model.")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv",
                        help="Path to the stock data CSV file.")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of business days to predict.")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Number of time steps to look back for each prediction.")
    
    args = parser.parse_args()
    main(model_path=args.model_path, data_path=args.data_path, 
         days_to_predict=args.days, lookback=args.lookback)