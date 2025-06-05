import yfinance as yf


garbage_collector = yf.download('AAPL', start='2015-01-01', end='2024-12-31', interval='1d')
print('i', garbage_collector.head())