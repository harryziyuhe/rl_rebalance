import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
from typing import List, Tuple, Dict
import os
import sys

# Add the current directory to sys.path to ensure dqn imports work if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import PortfolioEnvironment and helper from dqn.py
try:
    from dqn import PortfolioEnvironment, compute_minimum_variance_weights
except ImportError:
    # Fallback if running from root
    from rl_agents.dqn import PortfolioEnvironment, compute_minimum_variance_weights

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    
    Shared feature extractor with separate heads for Actor (policy) and Critic (value).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (outputs logits for actions)
        self.actor_fc = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (outputs scalar value)
        self.critic_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        
        # Actor: return logits
        action_logits = self.actor_fc(x)
        
        # Critic: return value
        state_value = self.critic_fc(x)
        
        return action_logits, state_value

class PPOAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        lr: float = 3e-4, 
        gamma: float = 0.99, 
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01,
        k_epochs: int = 4,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.k_epochs = k_epochs
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Select action and return action index and its log probability.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            logits, _ = self.policy_old(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
        return action.item(), dist.log_prob(action).item()
    
    def update(self, memory):
        """
        Update policy using PPO algorithm
        """
        # Convert memory to tensors
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values = self.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Get entropy
            _, _, dist_entropy = self.get_dist_entropy(old_states, old_actions)

            # Final loss: actor loss + critic loss - entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Step the scheduler
        self.scheduler.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def get_dist_entropy(self, state: torch.Tensor, action: torch.Tensor):
        logits, state_value = self.policy(state)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return action_logprobs, state_value, dist_entropy

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        # Kept for compatibility if needed, but we use get_dist_entropy now
        return self.get_dist_entropy(state, action)[:2]

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train_ppo(
    agent: PPOAgent,
    env: PortfolioEnvironment,
    n_episodes: int = 100,
    max_steps: int = 1000,
    update_timestep: int = 2000,
    verbose: bool = True
) -> Dict:
    
    memory = Memory()
    timestep = 0
    
    history = {
        'rewards': [],
        'turnovers': [],
        'tracking_errors': []
    }
    
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        ep_reward = 0
        ep_turnover = 0
        ep_tracking_error = 0
        steps = 0
        
        for t in range(max_steps):
            timestep += 1
            
            # Run old policy
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Save data in memory
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(torch.tensor(log_prob))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            ep_reward += reward
            ep_turnover += info.get('turnover', 0)
            ep_tracking_error += info.get('tracking_error', 0)
            steps += 1
            
            # Update if its time
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
                
        history['rewards'].append(ep_reward)
        history['turnovers'].append(ep_turnover)
        history['tracking_errors'].append(ep_tracking_error / max(steps, 1))
        
        # if verbose and i_episode % 10 == 0:
        if verbose:

            print(f"Episode {i_episode}/{n_episodes}")
            print(f"  Reward: {ep_reward:.4f}")
            print(f"  Turnover: {ep_turnover:.4f}")
            print(f"  Avg Tracking Error: {ep_tracking_error/max(steps, 1):.6f}")
            print()
            
    return history

if __name__ == "__main__":
    print("=" * 80)
    print("PPO Portfolio Rebalancing Training")
    print("=" * 80)
    
    # Paths
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    returns_path = os.path.join(data_dir, 'returns.parquet')
    cov_path = os.path.join(data_dir, 'cov_oas_window252.pkl')
    
    print("Loading data...")
    if not os.path.exists(returns_path) or not os.path.exists(cov_path):
        print("Data files not found. Please ensure data/returns.parquet and data/cov_oas_window252.pkl exist.")
        sys.exit(1)
        
    returns = pd.read_parquet(returns_path)
    with open(cov_path, 'rb') as f:
        cov_dict = pickle.load(f)
        
    train_start = '2010-01-01'
    train_end = '2019-12-31'
    returns_train = returns.loc[train_start:train_end]
    
    print("Computing targets...")
    target_weights_list = []
    for date in returns_train.index:
        if date in cov_dict:
            weights = compute_minimum_variance_weights(cov_dict[date])
        else:
            weights = np.ones(len(returns_train.columns)) / len(returns_train.columns)
        target_weights_list.append(weights)
        
    target_weights = pd.DataFrame(
        target_weights_list, 
        index=returns_train.index, 
        columns=returns_train.columns
    )
    
    print("Creating Environment...")
    env = PortfolioEnvironment(
        returns=returns_train,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=0.5
    )
    
    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        entropy_coef=0.01,
        k_epochs=4,
        device=device
    )
    
    print("Starting Training...")
    history = train_ppo(
        agent, 
        env, 
        n_episodes=200, 
        max_steps=len(returns_train), 
        update_timestep=4000,
        verbose=True
    )
    
    print("Saving Model...")
    agent.save('ppo_portfolio_model.pth')
    with open('ppo_training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
        
    print("Done!")

