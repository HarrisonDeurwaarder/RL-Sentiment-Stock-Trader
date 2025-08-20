import torch
import numpy as np
import pandas as pd
from typing import Tuple


class Environment:
    
    def __init__(self, 
                 ticker: str,
                 capital_dist: torch.distributions.Distribution = torch.distributions.Uniform(100.0, 1000.0),
                 horizon: int = 2048,
                 episode_cutoff: int = 64,
                 max_nav_portion: float = 1.0,) -> None:
        self.ticker = ticker
        
        # Value that scales how many shares are able to be bought
        self.dist = capital_dist
        self.start_capital = capital_dist.sample().item()
        self.horizon = horizon
        self.max_nav = max_nav_portion
        self.episode_cutoff = episode_cutoff
        
        self.start = 0
        self.end = horizon
        self.index = 0
        
        self.df = pd.DataFrame()
        self.assemble_env()
        
        
    def __repr__(self,) -> str:
        return f'Environment(ticker={self.ticker}, index={self.index}, capital={self.df["capital"][self.index-1]}, capital_dist={self.dist}, rollout_start={self.start}, rollout_end={self.end}, max_nav={self.max_nav}, episode_cutoff={self.episode_cutoff})'
    
    
    def __len__(self,) -> int:
        return len(self.df)
    
    
    def assemble_env(self,) -> None:
        '''
        Loads the dataframe from an imported CSV file
        '''
        # Extract the historical data from a single company
        data = pd.read_csv(f'Data/{self.ticker.lower()}_ohlcv.csv')
        self.df = data.drop([0, 1])
        self.df = self.df.rename(columns={'Price': 'date', 'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'})
        self.df[['close', 'high', 'low', 'open', 'volume']] = self.df[['close', 'high', 'low', 'open', 'volume']].astype(float)
        self.df = self.df.reset_index(drop=True)
    
    
    def reset(self,) -> pd.DataFrame:
        '''
        Resets the environment back to the start of the rollout and returns the first state
        '''
        self.index = self.start+1
        self.df['portfolio-vol'] = [0.0] * len(self.df)
        self.df['capital'] = [self.start_capital] + [0.0] * (len(self.df)-1)
        start_states = self.get_states()
        
        return start_states
    
    
    def soft_reset(self,) -> None:
        '''
        Marks the end of an episode. Resets the capital and portfolio-vol, but leaves the index
        '''
        self.df.loc[self.index, 'portfolio-vol'] = 0.0
        self.df.loc[self.index, 'capital'] = self.dist.sample().item()
    
    
    def next_rollout(self,) -> bool:
        '''
        Cycles to the next set of states in the horizon, indicates whether or not this is the final rollout
        '''
        self.start += self.horizon
        self.end += self.horizon
        
        return self.end + self.horizon >= len(self.df)
    
    
    def get_states(self,
                   max_memory: int = 20,) -> pd.DataFrame:
        '''
        Given an index, returns the current and memory states
        '''
        return self.df.iloc[max(self.index - max_memory+1, 0): self.index+1].drop('date', axis=1) if self.index < len(self.df) else None
    
    
    def get_nav(self,
                index: int,) -> float:
        '''
        Returns the net asset value at an index, given the current price
        '''
        return self.df['capital'][index] + self.df['open'][self.index] * self.df['portfolio-vol'][index]
    
    
    def get_reward(self,) -> float:
        '''
        Returns the reward of the current state
        '''
        nav, prev_nav = self.get_nav(self.index), self.get_nav(self.index-1)
        
        # Arithmetic return
        return nav - prev_nav
    
    
    def get_episode_return(self,) -> float:
        '''
        Returns the arithmetic return over an episode
        '''
        nav, start_nav = self.get_nav(self.index), self.get_nav(self.index-self.episode_cutoff)
        if self.index > 5:
            print(self.df['capital'][self.index-5:self.index+5])
        # Arithmetic return
        return (nav - start_nav) / (start_nav + 1e-4)
        
        
    def trade(self, 
              action: float,) -> float:
        '''
        Derives a target value based on action and trades to reach it. Discourages high or low holdings
        '''
        # Compress the action value on (0, w_max)
        w_nav = self.max_nav / (1 + np.exp(-action))
        # Current value of shares
        curr_value = self.df['open'][self.index] * self.df['portfolio-vol'][self.index-1]
        # Net asset value = price*shares + uninvested capital
        nav = self.df['capital'][self.index-1] + self.df['open'][self.index] * self.df['portfolio-vol'][self.index-1]
        # Portion of NAV to invest
        target_value = w_nav * nav
        delta_value = curr_value - target_value
        # Update new capital/volume based on target
        delta_vol = delta_value / self.df['open'][self.index] - self.df['portfolio-vol'][self.index-1]
        
        self.df.loc[self.index, 'capital'] += delta_value
        self.df.loc[self.index, 'portfolio-vol'] += delta_vol
    
    
    def step(self,
             action: float,
             max_memory: int = 20,) -> Tuple[pd.DataFrame, float, bool]:
        '''
        Handles a step in the RL environment (executes action, returns next state, computes reward)
        '''
        is_over = self.index >= self.end
        
        self.trade(action=action,)

        # Get the past (max_memory) states, or the most if that's not possible
        state = self.get_states(max_memory=max_memory,)
        reward = self.get_reward()
        
        # Episode reset, excluding rollout beginning
        if self.index % self.episode_cutoff == 0 and self.index != self.start:
            self.soft_reset()
        
        self.index += 1
        return state, reward, is_over