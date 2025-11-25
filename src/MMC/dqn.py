"""
Deep Q-Network (DQN) for Portfolio Rebalancing

This module implements a Deep Q-Network to learn optimal portfolio rebalancing policies
under transaction costs. The agent learns when and how much to rebalance towards a
time-varying minimum-variance target portfolio.

===================================================================================
STATE REPRESENTATION
===================================================================================
The state vector contains the following features:

1. Current Portfolio Weights (n_assets dimensions):
   - Current allocation across all assets
   - Normalized to sum to 1.0

2. Target Portfolio Weights (n_assets dimensions):
   - Minimum-variance portfolio weights from shrunk rolling covariance
   - The optimal allocation we want to move towards

3. Weight Deviation (n_assets dimensions):
   - Difference between current and target weights: current - target
   - Indicates how far we are from optimal allocation

4. Portfolio Metrics (scalar features):
   - Tracking error: distance from target (L2 norm of weight deviation)
   - Recent portfolio volatility (rolling standard deviation)
   - Recent portfolio returns
   - Days since last rebalance

Total State Dimension: 3 * n_assets + 4

===================================================================================
ACTION SPACE
===================================================================================
Discrete action space representing rebalancing intensity:

Action 0: No rebalance (hold current positions)
Action 1: Small rebalance (move 25% towards target)
Action 2: Medium rebalance (move 50% towards target)
Action 3: Large rebalance (move 75% towards target)
Action 4: Full rebalance (move 100% to target)

Each action determines what fraction of the gap between current and target
weights to close.

===================================================================================
REWARD FUNCTION
===================================================================================
The reward balances three components:

R = wR - TC(Δwt) - λ(dt)²

Where:
1. Portfolio Return (wR):
   - Weighted return of the portfolio (w^T * r)
   - Rewards positive portfolio performance

2. Transaction Cost TC(Δwt):
   - Cost proportional to turnover: cost_rate * Σ|Δw_i|
   - Penalizes trading due to transaction costs

3. Tracking Error Penalty λ(dt)²:
   - λ * ||w_current - w_target||²
   - Penalizes deviation from the minimum-variance target

===================================================================================
NEURAL NETWORK ARCHITECTURE
===================================================================================
The DQN uses a fully connected neural network to approximate Q(s, a):

Input Layer: state_dim dimensions
Hidden Layer 1: 256 neurons + ReLU + Dropout(0.2)
Hidden Layer 2: 128 neurons + ReLU + Dropout(0.2)
Hidden Layer 3: 64 neurons + ReLU
Output Layer: n_actions neurons (Q-values for each action)

The network learns to predict the expected cumulative discounted reward for
taking each action in a given state.

===================================================================================
OPTIMIZATION DETAILS
===================================================================================
Loss Function: Huber Loss (smooth L1)
- Less sensitive to outliers than MSE
- Loss = Huber(Q(s,a) - target)
- target = r + � * max_a' Q_target(s', a')

Optimizer: Adam
- Learning rate: 1e-4
- Gradient clipping to prevent exploding gradients

Training Details:
- Experience Replay Buffer: 100,000 transitions
- Batch Size: 64
- Target Network: Updated every 1000 steps (soft update with �=0.001)
- �-greedy exploration: � decays from 1.0 to 0.01 over training
- Discount factor �: 0.99

===================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import pickle
from typing import Tuple, List
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(nn.Module):
    """
    Deep Q-Network Neural Network Architecture

    A fully connected neural network that approximates the Q-value function Q(s, a),
    which represents the expected cumulative discounted reward for taking action a
    in state s and following the optimal policy thereafter.

    Parameters:
    -----------
    state_dim : int
        Dimension of the state space (3 * n_assets + 4)
    action_dim : int
        Number of possible actions (5 rebalancing intensities)
    hidden_dims : List[int]
        Sizes of hidden layers [default: [256, 128, 64]]
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        self.fc4 = nn.Linear(hidden_dims[2], action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU activations
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Parameters:
        -----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)

        Returns:
        --------
        torch.Tensor
            Q-values for each action, shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))

        q_values = self.fc4(x)

        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN

    Stores transitions (s, a, r, s', done) and samples random minibatches for training.
    This breaks the temporal correlation in the training data and improves stability.

    Parameters:
    -----------
    capacity : int
        Maximum number of transitions to store
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class PortfolioEnvironment:
    """
    Portfolio Rebalancing Environment

    Simulates portfolio dynamics with rebalancing decisions under transaction costs.

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (dates � assets)
    target_weights : pd.DataFrame
        Target portfolio weights from minimum-variance optimization (dates � assets)
    transaction_cost : float
        Proportional transaction cost rate (e.g., 0.001 = 0.1%)
    lambda_tracking : float
        Penalty weight for tracking error (lambda in the formula)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        target_weights: pd.DataFrame,
        transaction_cost: float = 0.001,
        lambda_tracking: float = 1.0
    ):
        self.returns = returns
        self.target_weights = target_weights
        self.transaction_cost = transaction_cost
        self.lambda_tracking = lambda_tracking

        self.dates = returns.index.intersection(target_weights.index)
        self.returns = returns.loc[self.dates]
        self.target_weights = target_weights.loc[self.dates]

        self.n_assets = len(returns.columns)
        self.n_actions = 5  # 0%, 25%, 50%, 75%, 100% rebalancing

        # Action mapping: what fraction of the gap to close
        self.action_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

        self.reset()

    def reset(self, start_idx: int = 0) -> np.ndarray:
        """
        Reset the environment to initial state

        Parameters:
        -----------
        start_idx : int
            Starting index in the data

        Returns:
        --------
        np.ndarray
            Initial state vector
        """
        self.current_idx = start_idx
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.days_since_rebalance = 0

        self.portfolio_volatility = 0.0
        self.portfolio_return = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Construct the state vector

        Returns:
        --------
        np.ndarray
            State vector containing:
            - Current weights (n_assets)
            - Target weights (n_assets)
            - Weight deviations (n_assets)
            - Tracking error (1)
            - Portfolio volatility (1)
            - Portfolio return (1)
            - Days since rebalance (1)
        """
        target_w = self.target_weights.iloc[self.current_idx].values
        weight_dev = self.current_weights - target_w
        tracking_error = np.sqrt(np.sum(weight_dev ** 2))

        state = np.concatenate([
            self.current_weights,
            target_w,
            weight_dev,
            [tracking_error],
            [self.portfolio_volatility],
            [self.portfolio_return],
            [self.days_since_rebalance / 100.0]  # Normalize days
        ])

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Parameters:
        -----------
        action : int
            Action index (0-4)

        Returns:
        --------
        next_state : np.ndarray
            State after taking the action
        reward : float
            Immediate reward
        done : bool
            Whether the episode is finished
        info : dict
            Additional information (turnover, tracking_error, etc.)
        """
        target_w = self.target_weights.iloc[self.current_idx].values

        rebalance_fraction = self.action_fractions[action]
        weight_change = rebalance_fraction * (target_w - self.current_weights)
        new_weights_pre_return = self.current_weights + weight_change

        turnover = np.sum(np.abs(weight_change))

        transaction_cost_incurred = turnover * self.transaction_cost

        self.current_idx += 1
        done = (self.current_idx >= len(self.dates) - 1)

        if not done:
            period_returns = self.returns.iloc[self.current_idx].values
            portfolio_return = np.sum(new_weights_pre_return * period_returns)
            self.current_weights = new_weights_pre_return * (1 + period_returns)
            self.current_weights = self.current_weights / np.sum(self.current_weights)
            self.portfolio_return = portfolio_return
            target_w_new = self.target_weights.iloc[self.current_idx].values
            tracking_error = np.sqrt(np.sum((self.current_weights - target_w_new) ** 2))

            reward = (
                portfolio_return                                    # wR
                - transaction_cost_incurred                         # TC(Δwt)
                - self.lambda_tracking * (tracking_error ** 2)      # λ(dt)²
            )

            if turnover > 0.001:  # Threshold for considering it a rebalance
                self.days_since_rebalance = 0
            else:
                self.days_since_rebalance += 1

            next_state = self._get_state()
        else:
            reward = 0.0
            next_state = self._get_state()
            tracking_error = 0.0
            portfolio_return = 0.0

        info = {
            'turnover': turnover,
            'tracking_error': tracking_error,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost_incurred
        }

        return next_state, reward, done, info


class DQNAgent:
    """
    DQN Agent for Portfolio Rebalancing

    Implements Double DQN with experience replay and target networks.

    Parameters:
    -----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Number of possible actions
    learning_rate : float
        Learning rate for Adam optimizer
    gamma : float
        Discount factor for future rewards
    epsilon_start : float
        Initial exploration rate
    epsilon_end : float
        Final exploration rate
    epsilon_decay : float
        Decay rate for epsilon
    buffer_size : int
        Size of replay buffer
    batch_size : int
        Minibatch size for training
    target_update_freq : int
        How often to update target network (in steps)
    device : str
        Device to run on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Parameters:
        -----------
        state : np.ndarray
            Current state
        training : bool
            If True, use epsilon-greedy exploration

        Returns:
        --------
        int
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """
        Perform one training step on a minibatch

        Returns:
        --------
        float
            Training loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q(s, a) using policy network
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using target network (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {filepath}")


def compute_minimum_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute minimum variance portfolio weights

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix (n_assets � n_assets)

    Returns:
    --------
    np.ndarray
        Portfolio weights that minimize variance
    """
    n = len(cov_matrix)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        w = inv_cov @ ones
        w = w / np.sum(w)
        w = np.maximum(w, 0)
        w = w / np.sum(w)
        return w
    except np.linalg.LinAlgError:
        return np.ones(n) / n


def train_dqn(
    agent: DQNAgent,
    env: PortfolioEnvironment,
    n_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    verbose: bool = True
) -> dict:
    """
    Train the DQN agent

    Parameters:
    -----------
    agent : DQNAgent
        The DQN agent to train
    env : PortfolioEnvironment
        The portfolio environment
    n_episodes : int
        Number of training episodes
    max_steps_per_episode : int
        Maximum steps per episode
    verbose : bool
        Print training progress

    Returns:
    --------
    dict
        Training history (episode_rewards, episode_losses, etc.)
    """
    episode_rewards = []
    episode_losses = []
    episode_turnovers = []
    episode_tracking_errors = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_turnover = 0.0
        episode_tracking_error = 0.0
        steps = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.train_step()

            episode_reward += reward
            episode_loss += loss
            episode_turnover += info['turnover']
            episode_tracking_error += info['tracking_error']
            steps += 1

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / max(steps, 1))
        episode_turnovers.append(episode_turnover)
        episode_tracking_errors.append(episode_tracking_error / max(steps, 1))

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Avg Loss: {episode_loss / max(steps, 1):.6f}")
            print(f"  Turnover: {episode_turnover:.4f}")
            print(f"  Avg Tracking Error: {episode_tracking_error / max(steps, 1):.6f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Steps: {steps}")
            print()

    history = {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'episode_turnovers': episode_turnovers,
        'episode_tracking_errors': episode_tracking_errors
    }

    return history


if __name__ == "__main__":
    """
    Main training script

    Loads data, creates environment and agent, and trains the DQN on
    portfolio rebalancing from 2010-01-01 to 2019-12-31
    """

    print("=" * 80)
    print("DQN Portfolio Rebalancing Training")
    print("=" * 80)
    print()

    print("Loading data...")
    returns = pd.read_parquet('../data/returns.parquet')
    prices = pd.read_parquet('../data/prices.parquet')

    with open('../data/cov_oas_window252.pkl', 'rb') as f:
        cov_dict = pickle.load(f)

    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print()

    train_start = '2010-01-01'
    train_end = '2019-12-31'

    returns_train = returns.loc[train_start:train_end]
    print(f"Training period: {returns_train.index.min()} to {returns_train.index.max()}")
    print(f"Training samples: {len(returns_train)}")
    print()

    print("Computing minimum variance target weights...")
    target_weights_list = []

    for date in returns_train.index:
        if date in cov_dict:
            cov_matrix = cov_dict[date]
            weights = compute_minimum_variance_weights(cov_matrix)
        else:
            weights = np.ones(len(returns_train.columns)) / len(returns_train.columns)
        target_weights_list.append(weights)

    target_weights = pd.DataFrame(
        target_weights_list,
        index=returns_train.index,
        columns=returns_train.columns
    )
    print("Target weights computed.")
    print()

    print("Creating portfolio environment...")
    env = PortfolioEnvironment(
        returns=returns_train,
        target_weights=target_weights,
        transaction_cost=0.001,  # 0.1% transaction cost (TC rate)
        lambda_tracking=1.0      # Tracking error penalty (λ)
    )

    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of assets: {env.n_assets}")
    print()

    print("Creating DQN agent...")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    print()

    print("Starting training...")
    print()
    history = train_dqn(
        agent=agent,
        env=env,
        n_episodes=100,
        max_steps_per_episode=len(returns_train),
        verbose=True
    )
    print("Saving trained model...")
    agent.save('dqn_portfolio_model.pth')

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to training_history.pkl")

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
