import yfinance as yf
import pandas as pd

# List taken from GPT-4o
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

def main():
    '''
    Save ohlcv data for every ticker
    '''
    for ticker in tickers:
        data = yf.download(ticker, start='2000-01-01', end='2025-6-1', interval='1d') 
        data.to_csv(f'Data/{ticker.lower()}_ohlcv.csv', index=True)
    
if __name__ == '__main__':
    main()