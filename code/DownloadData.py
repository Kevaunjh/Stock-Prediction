import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def clean_csv_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        cleaned_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and stripped[0].isdigit():
            cleaned_lines.append(line)
    header = "Date,Adj Close,Close,High,Low,Open\n"
    cleaned_lines.insert(0, header)
    
    with open(file_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    df = pd.read_csv(file_path)
    if "Volume" in df.columns:
        df.drop(columns=["Volume"], inplace=True)
    
    desired_order = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open']
    df = df[desired_order]
    df.to_csv(file_path, index=False)
    return df
file_path = './../data/stock_data.csv'
ticker = 'TSLA'
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

if os.path.exists(file_path):
    print("File exists. Cleaning up the CSV file...")
    df = clean_csv_file(file_path)
    print("Cleaned Data:")
    print(df.head())
else:
    print("File does not exist. Downloading data...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = data.index.date
    data.to_csv(file_path, index_label="Date")
    df = clean_csv_file(file_path)
    print("Downloaded and cleaned Data:")
    print(df.head())

print(f"Data is saved in {file_path}")
