import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def clean_csv_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if i == 0 and stripped.startswith("Date"):
            cleaned_lines.append(line)
        elif stripped and stripped[0].isdigit():
            cleaned_lines.append(line)

    if not cleaned_lines or not cleaned_lines[0].startswith("Date"):
        header = "Date,Adj Close,Close,High,Low,Open\n"
        cleaned_lines.insert(0, header)

    with open(file_path, 'w') as f:
        f.writelines(cleaned_lines)

    df = pd.read_csv(file_path, parse_dates=['Date'])

    if "Volume" in df.columns:
        df.drop(columns=["Volume"], inplace=True)

    desired_order = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open']
    existing_cols = [col for col in desired_order if col in df.columns]
    df = df[existing_cols]

    df.to_csv(file_path, index=False)
    return df

def main():
    file_path = "./../data/stock_data.csv"
    news_file_path = "./../data/stock_data_with_sentiment.csv"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        os.remove(news_file_path)
        print(f"Existing file '{file_path}' has been deleted.")
        print(f"Existing file '{news_file_path}' has been deleted.")
    
    ticker = 'TSLA'
    print("Downloading data for the last 60 calendar days...")
    data = yf.download(ticker, period='60d')

    last_30_data = data.tail(50)
    last_30_data.index = last_30_data.index.date
    last_30_data.to_csv(file_path, index_label="Date")

    df = clean_csv_file(file_path)

    print("Cleaned Data (Last 30 Trading Days):")
    print(df.head())
    print(f"Data is saved in {file_path}")

if __name__ == "__main__":
    main()
