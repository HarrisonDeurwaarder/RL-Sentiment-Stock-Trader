from typing import Dict, List
from torch.utils.data.dataset import Dataset
import torch


class Rollout(Dataset):
    '''
    Contains the rollout of states of length HORIZON to train the policy
    '''
    def __init__(self, 
                 states: List[torch.Tensor], 
                 actions: List[torch.Tensor], 
                 rewards: List[float],) -> None:
        self.states = states[:-1]
        self.next_states = states[1:]
        self.actions = actions
        self.rewards = rewards
        
    def __len__(self,) -> int:
        return len(self.states)
    
    def __getitem__(self, 
                    index: int,) -> Dict[str, torch.Tensor]:
        return {
            'states': self.states[index],
            'next_states': self.next_states[index],
            'actions': self.actions[index],
            'rewards': self.rewards[index]
        }