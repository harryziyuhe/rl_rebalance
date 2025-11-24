import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import teammate's classes and utilities
# Reuse exact env and network architecture for apples-to-apples comparison
from dqn import PortfolioEnvironment, DQN as QNetwork, compute_minimum_variance_weights


class MonteCarloAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, gamma: float = 0.99, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        # Same architecture as teammate's DQN for fair comparison
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Per-episode memory: list of (state, action, reward)
        self.episode_memory: List[Tuple[np.ndarray, int, float]] = []

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        self.episode_memory.append((state, action, reward))

    def update_policy(self) -> float:
        if not self.episode_memory:
            return 0.0

        states, actions, rewards = zip(*self.episode_memory)

        # Compute discounted returns G_t backwards
        G = 0.0
        returns: List[float] = []
        for r in reversed(rewards):
            G = float(r) + self.gamma * G
            returns.insert(0, G)

        # Tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(list(actions)).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Predicted Q-values for actions taken
        q_values = self.policy_net(states_t)  # [T, action_dim]
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [T]

        # Supervised regression to Monte Carlo returns
        loss = nn.MSELoss()(q_taken, returns_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Clear episode memory
        self.episode_memory.clear()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)


def _compute_target_weights(returns_df: pd.DataFrame, covariances: dict) -> pd.DataFrame:
    """
    Build a DataFrame of minimum-variance weights aligned to returns_df.index,
    using the provided rolling covariance dictionary keyed by date.
    """
    n_assets = len(returns_df.columns)
    target_weights_list: List[np.ndarray] = []
    for date in returns_df.index:
        # Check validity of covariance matrix before computing
        if (date in covariances and 
            isinstance(covariances[date], np.ndarray) and 
            np.isfinite(covariances[date]).all()):
            
            weights = compute_minimum_variance_weights(covariances[date])
            # Double check output is finite
            if not np.isfinite(weights).all():
                weights = np.ones(n_assets, dtype=float) / n_assets
        else:
            weights = np.ones(n_assets, dtype=float) / n_assets
        
        target_weights_list.append(weights)

    target_weights = pd.DataFrame(
        target_weights_list,
        index=returns_df.index,
        columns=returns_df.columns
    )
    return target_weights


def train_mmc():
    # Resolve data directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    returns_path = os.path.join(data_dir, 'returns.parquet')
    cov_path = os.path.join(data_dir, 'cov_oas_window252.pkl')

    print(f"Loading data from {os.path.abspath(data_dir)}...")
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Could not find {returns_path}")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"Could not find {cov_path}")

    returns_df = pd.read_parquet(returns_path)
    with open(cov_path, 'rb') as f:
        covariances = pickle.load(f)

    # Compute min-var targets aligned to returns
    print("Computing minimum-variance target weights...")
    target_weights = _compute_target_weights(returns_df, covariances)
    print("Targets computed.")

    # Initialize environment with same interface as teammate's DQN setup
    env = PortfolioEnvironment(
        returns=returns_df,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=1.0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Derive dimensions from env
    init_state = env.reset()
    state_dim = int(len(init_state))
    action_dim = int(env.n_actions)

    agent = MonteCarloAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        device=device
    )

    n_episodes = 10000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    initial_balance = 100000.0

    history = {'rewards': [], 'losses': [], 'final_balances': []}

    print("Starting Monte Carlo training...")
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        current_balance = initial_balance

        while not done:
            if not np.isfinite(state).all():
                print(f"Warning: NaN detected in state at episode {episode+1}. Ending episode.")
                break
                
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            
            if not np.isfinite(reward):
                # Fallback reward if calculation fails
                reward = -10.0 
            
            # Track portfolio balance
            # info['portfolio_return'] is the raw return w_t * r_{t+1}
            # We apply it to the current balance
            current_balance *= (1.0 + info['portfolio_return'])

            agent.store_transition(state, action, reward)
            state = next_state
            total_reward += float(reward)

        # Monte Carlo update at end of episode
        loss = agent.update_policy()

        # Epsilon schedule
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        history['rewards'].append(total_reward)
        history['losses'].append(loss)
        history['final_balances'].append(current_balance)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{n_episodes} | Return: {total_reward:.4f} | Loss: {loss:.6f} | Balance: ${current_balance:,.2f} | Epsilon: {epsilon:.2f}")

    # Save artifacts
    save_model_path = os.path.join(current_dir, 'mmc_portfolio_model.pth')
    agent.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    history_path = os.path.join(current_dir, 'mmc_training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    train_mmc()


