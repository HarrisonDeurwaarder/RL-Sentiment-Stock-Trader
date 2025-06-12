import datetime as dt
import torch
import pandas as pd


class Environment:
    
    def __init__(self, 
                 ticker: str,
        ) -> None:
        self.ticker = ticker
        
        self.shares = [0]
        
        self.date = dt.datetime(2000, 1, 1)
        self.df = pd.DataFrame()
        self.index = 0
        self.assemble_env()
        
        
    def __repr__(self,) -> str:
        return f'Environment(ticker={self.ticker}, next_date={self.date})'
    
    
    def assemble_env(self,) -> None:
        '''
        Loads the dataframe from an imported CSV file
        '''
        # Extract the historical data from a single company
        data = pd.read_csv(f'Data/{self.ticker}_ohlcv.csv')
        self.df = data.drop([0, 1])
        self.df = self.df.rename(columns={'Price': 'date', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'})
    
    
    def get_states(self, 
                   index: int, 
                   max_memory: int = 20,
        ) -> torch.Tensor:
        '''
        Given an index, returns the current state and memory states
        '''
        return self.df.iloc[max(index - max_memory+1, 0): index+1] if index > len(self.df) else None
    
    
    def get_reward(self,
                   index: int,
        ) -> float:
        '''
        Gets the reward of the current state
        '''
    
    
    @staticmethod
    def get_port_val(shares: int,
                     high: float,
                     low: float,
        ) -> float:
        '''
        Gets the value of the portfolio given the average price and num shares
        '''
        avg_price = (high + low) * 0.5
        return shares * avg_price
    
    
    def step(self,
             action: float,
             max_memory: int = 20,
        ) -> torch.Tensor:
        '''
        Handles a step in the RL environment (executes action, returns next state, computes reward)
        '''
        is_over = self.index > len(self.df)-1

        # Get the past (max_memory) states, or the most if that's not possible
        states = self.get_states(index=self.index, max_memory=max_memory)
        next_states = self.get_states(index=self.index+1, max_memory=max_memory)
        
        self.index += 1
        return states, is_over