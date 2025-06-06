import yfinance as yf
import pandas as pd
import utils

tickers = [
    # Technology
    'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN', 'TSLA', 'NVDA', 'IBM', 'ORCL', 'INTC',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB',
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABBV', 'TMO', 'CVS', 'BMY', 'AMGN', 'GILD',
    # Consumer
    'KO', 'PEP', 'WMT', 'COST', 'MCD', 'NKE', 'SBUX', 'DIS', 'HD', 'LOW',
    # Industrials
    'BA', 'CAT', 'DE', 'MMM', 'HON', 'LMT', 'GE', 'RTX', 'FDX', 'UPS',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'PSX', 'EOG', 'VLO', 'MPC', 'KMI'
]

data = yf.download(tickers, start='2000-01-01', end='2025-6-1', interval='1d')
df = pd.DataFrame()

print(data.columns.values)
# Downoad ohlcv data
for i in range(1, 60):
    df = pd.concat(df, utils.parse_data(data))
    print(ticker)
    
data.to_csv('data.csv')