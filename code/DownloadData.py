import os
import yfinance as yf
import pandas as pd
from datetime import datetime

file_path = './../data/stock_data.csv'

# Define ticker and date range
ticker = 'TSLA'
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

if os.path.exists(file_path):
    print("File exists. Loading data from file...")

    # Load the data
    data = pd.read_csv(file_path)

    # Ensure 'Date' is parsed as datetime and set it as the index
    data['Date'] = pd.to_datetime(data['Date'], utc=True).dt.tz_convert(None)
    data.set_index('Date', inplace=True)

    print("Cleaned Data:")
    print(data.head())
else:
    print("File does not exist. Downloading data...")

    # Download data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Remove time from the Date index
    data.index = data.index.date

    # Save to CSV with the cleaned Date
    data.to_csv(file_path, index_label="Date")
    print("Downloaded Data:")
    print(data.head())

print(f"Data is saved in {file_path}")
