import datetime as dt
import torch
import numpy as np
import pandas as pd
from typing import Tuple


class Environment:
    
    def __init__(self, 
                 ticker: str,
                 total_capital: float = 1000,
                 horizon: int = 2048) -> None:
        self.ticker = ticker
        
        # Value that scales how many shares are able to be bought
        self.start_capital = total_capital
        self.horizon = horizon
        
        self.start = 0
        self.end = horizon
        self.index = 0
        
        self.df = pd.DataFrame()
        self.assemble_env()
        
        
    def __repr__(self,) -> str:
        return f'Environment(ticker={self.ticker}, index={self.index}, capital={self.capital}, rollout_start={self.start}, rollout_end={self.end})'
    
    
    def __len__(self,) -> int:
        return len(self.df)
    
    
    def assemble_env(self,) -> None:
        '''
        Loads the dataframe from an imported CSV file
        '''
        # Extract the historical data from a single company
        data = pd.read_csv(f'Data/{self.ticker}_ohlcv.csv')
        self.df = data.drop([0, 1])
        self.df = self.df.rename(columns={'Price': 'date', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'})
        self.df[['close', 'high', 'low', 'open', 'volume']] = self.df[['close', 'high', 'low', 'open', 'volume']].astype(float)
        self.df = self.df.reset_index(drop=True)
    
    
    def reset(self,) -> pd.DataFrame:
        '''
        Resets the environment back to the start of the rollout and returns the first state
        '''
        self.index = self.start
        self.df['portfolio-vol'] = [0.0] * len(self.df)
        self.df['capital'] = [self.start_capital] + [0.0] * (len(self.df)-1)
        start_states = self.get_states()
        self.index += 1
        
        return start_states
    
    
    def next_rollout(self,) -> bool:
        '''
        Cycles to the next set of states in the horizon, indicates whether or not this is the final rollout
        '''
        self.start += self.horizon
        self.end += self.horizon
        
        print(self.end + self.horizon <= len(self.df))
        return self.end + self.horizon <= len(self.df)
    
    
    def get_states(self,
                   max_memory: int = 20,) -> pd.DataFrame:
        '''
        Given an index, returns the current and memory states
        '''
        return self.df.iloc[max(self.index - max_memory+1, 0): self.index+1].drop('date', axis=1) if self.index < len(self.df) else None
    
    
    def get_reward(self,) -> float:
        '''
        Gets the reward of the current state
        '''
        # Get the value of the portfolio by averaging the high/low of the day
        port_val = lambda shares, high, low: (high + low) * 0.5 * shares
        
        val = port_val(self.df['portfolio-vol'][self.index], self.df['high'][self.index], self.df['low'][self.index])
        next_val = port_val(self.df['portfolio-vol'][self.index+1], self.df['high'][self.index+1], self.df['low'][self.index+1])
        
        # Computes the log return of an action
        return np.log(val / (next_val + 1e-4) + 1e-4)
        
        
    def trade(self, 
              action: float,) -> None:
        '''
        Convert a trading score (action) into bought / sold shares for the next state
        '''
        # Spending cap based on the predicted action scalar [-1, 1]
        trade_cap = action * self.df['capital'][self.index-1]
        # Clip the new volume to prevent negative holdings
        new_volume = max(trade_cap // self.df['open'][self.index] + self.df['portfolio-vol'][self.index-1], 0)
        # Circle back and scale based on the actual $ spent
        self.df.loc[self.index, 'capital'] = self.df.loc[self.index, 'capital'] - self.df.loc[self.index, 'open'] * new_volume
        self.df.loc[self.index, 'portfolio-vol'] = new_volume
    
    
    def step(self,
             action: float,
             sample: bool = False,
             max_memory: int = 20,) -> Tuple[pd.DataFrame, float, bool]:
        '''
        Handles a step in the RL environment (executes action, returns next state, computes reward)
        '''
        is_over = self.index > self.end-1
        if sample:
            action = torch.normal(action[0], action[1])
        
        self.trade(action=action,)

        # Get the past (max_memory) states, or the most if that's not possible
        state = self.get_states(max_memory=max_memory,)
        reward = self.get_reward()
        
        self.index += 1
        return state, reward, is_over