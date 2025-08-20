from typing import Dict, List, Union
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch


class Rollout(Dataset):
    '''
    Contains the rollout of states of length HORIZON to train the policy
    '''
    def __init__(self, 
                 states: List[torch.Tensor], 
                 actions: List[torch.Tensor], 
                 log_probs: List[torch.Tensor],
                 rewards: List[float],
                 device: torch.DeviceObjType) -> None:
        self.states = states[:-1]
        self.next_states = states[1:]
        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards
        
        self.device = device
        
    def __len__(self,) -> int:
        return len(self.states)
    
    def __getitem__(self, 
                    index: int,) -> List[torch.Tensor]:
        return [
            self.states[index],
            self.next_states[index],
            self.actions[index],
            self.log_probs[index],
            self.rewards[index],
        ]
        
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]],) -> Dict[str, Union[torch.Tensor]]:
        '''
        Pads the state_t and state_t+1 using torch.nn.utils.rnn.pack_padded_sequence
        '''
        # Format 
        states, next_states, actions, log_probs, rewards = zip(*batch)
        # Derive the lengths of the paddings for torch.nn.utils.rnn.pack_padded_sequence and pad_sequences
        lengths = [state.shape[0] for state in states]
        # Padded transition variables
        states = pad_sequence(states,
                            batch_first=True,)
        next_states = pad_sequence(next_states,
                                batch_first=True,)
        
        packed_states = pack_padded_sequence(states,
                                            lengths=lengths,
                                            batch_first=True,
                                            enforce_sorted=False,)
        packed_next_states = pack_padded_sequence(next_states,
                                                lengths=lengths,
                                                batch_first=True,
                                                enforce_sorted=False,)
        return {
            'states': packed_states.float().to(self.device),
            'next_states': packed_next_states.float().to(self.device),
            'actions': torch.stack(actions).float().to(self.device),
            'log_probs': torch.stack(log_probs).float().to(self.device),
            'rewards': torch.tensor(rewards).float().to(self.device),
        }