from environment import *
from model import *
from utils import *
from data import *
from torch.utils.data.dataloader import DataLoader
import torch
from datetime import datetime as dt
torch.autograd.set_detect_anomaly(True)

# Ensures the agent is adaptable to the start amount
CAPITAL_DIST = torch.distributions.Uniform(low=100, high=1000)
PATH = f'Models/{dt.now()}'
DISCOUNT_FACTOR = 0.99
CLIPPING_PARAM = 0.2
TRAINING_EPOCHS = 4
ITERATIONS = 1
HORIZON = 1024
EPISODE_CUTOFF = 16
BATCH = 64
MEMORY = 10
MAX_NAV_TRADE = 1.0
assert HORIZON % BATCH == 0


def main() -> None:
    
    # Default to cpu if gpu is unavailable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO]: Using device: {str(device).upper()}')
    
    actor = Actor().to(device)
    critic = Critic().to(device)
        
    for _ in range(ITERATIONS):
        for ticker in tickers:
            
            # Assemble the environment and define the rollout indices
            env = Environment(ticker=ticker, 
                            capital_dist=CAPITAL_DIST,
                            horizon=HORIZON,
                            max_nav_portion=MAX_NAV_TRADE,)
            # Until no more full rollouts can be taken from the data
            rollouts_exhausted = False
            while not rollouts_exhausted:
                print(f'[INFO]: Next rollout collection started (ticker={ticker.upper()})... ')
                
                # To be converted to a torch Dataset for training
                states = []
                actions = []
                log_probs = []
                rewards = []
                start = dt.now()
                
                # Append root state
                states.append(
                    torch.tensor(env.reset().values).float()
                )
                
                # Until rollout queue is exhausted
                rollout_over = False
                while not rollout_over:
                    # Compute rollout
                    
                    action, dist = actor(states[-1].float().to(device))
                    next_state, reward, rollout_over = env.step(action=action.item(),
                                                max_memory=MEMORY,)
                    states.append(
                        torch.tensor(next_state.values)
                    )
                    actions.append(action.detach())
                    log_probs.append(dist.log_prob(action).detach())
                    rewards.append(reward)
                    
                data = Rollout(states=states,
                            actions=actions,
                            log_probs=log_probs,
                            rewards=rewards,
                            device=device,)
                data = DataLoader(data, 
                                batch_size=BATCH, 
                                shuffle=True,
                                collate_fn=data.collate_fn,)
                
                # Training loop
                for _ in range(TRAINING_EPOCHS):
                    for batch in data:
                        actor.train(critic=critic,
                                    rewards=batch['rewards'],
                                    states=batch['states'],
                                    old_actions=batch['actions'], 
                                    old_log_probs=batch['log_probs'],
                                    discount_factor=DISCOUNT_FACTOR,
                                    clipping_parameter=CLIPPING_PARAM,)
                        critic.train(rewards=batch['rewards'],
                                    states=batch['states'],
                                    next_states=batch['next_states'],
                                    discount_factor=DISCOUNT_FACTOR)
                
                
                print(f'[INFO]: Rollout completed in {dt.now() - start}s\n[INFO]: Mean reward: {torch.tensor(rewards).mean():.5f}')
                # Verifies that more rollouts can be extracted
                rollouts_exhausted = env.next_rollout()
                
    print('[INFO]: Training complete.')
    
    # Save actor
    torch.save(actor, PATH)
    print(f'[INFO]: Agent saved to {PATH}')


if __name__ == '__main__':
    main()