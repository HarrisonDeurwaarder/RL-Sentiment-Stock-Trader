import pandas as pd


def parse_data(df):
    ticker = df.iloc(1)[0]
    
    new_df = pd.DataFrame()
    new_df[['date', 'close', 'high', 'low', 'open', 'volume']] = df[['Price', 'Close', 'High', 'Low', 'Open', 'Volume']][2:]
    new_df['ticker'] = [df['date'][0]] * len(df)
    
    return new_df