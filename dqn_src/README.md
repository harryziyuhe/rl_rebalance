# Deep Q-Network (DQN) for Portfolio Rebalancing

This directory contains a DQN implementation for learning optimal portfolio rebalancing policies under transaction costs.

## Overview

The DQN agent learns to balance two competing objectives:
1. **Minimizing tracking error** - staying close to the minimum-variance target portfolio
2. **Minimizing transaction costs** - avoiding excessive trading

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python dqn.py
```

This will:
- Load data from `../data/` (returns and covariance matrices)
- Train on data from 2010-01-01 to 2019-12-31
- Save the trained model to `dqn_portfolio_model.pth`
- Save training history to `training_history.pkl`

### Visualizing Results

```bash
python visualize_training.py
```

This generates `training_history.png` with training metrics.

## Model Architecture

### State Representation (3n + 4 dimensions)

For n assets, the state vector contains:

1. **Current Portfolio Weights** (n dims): Current allocation across assets
2. **Target Portfolio Weights** (n dims): Minimum-variance optimal weights
3. **Weight Deviations** (n dims): Current - Target weights
4. **Tracking Error** (1 dim): L2 distance from target
5. **Portfolio Volatility** (1 dim): Recent volatility
6. **Portfolio Return** (1 dim): Recent return
7. **Days Since Rebalance** (1 dim): Time since last rebalance

### Action Space (5 discrete actions)

- **Action 0**: No rebalance (0% towards target)
- **Action 1**: Small rebalance (25% towards target)
- **Action 2**: Medium rebalance (50% towards target)
- **Action 3**: Large rebalance (75% towards target)
- **Action 4**: Full rebalance (100% to target)

### Reward Function

```
R = wR - TC(Δwt) - λ(dt)²
```

Where:
- **wR**: Portfolio return (weighted return)
- **TC(Δwt)**: Transaction cost = cost_rate × Σ|Δw_i|
- **λ(dt)²**: Tracking error penalty = λ × ||w_current - w_target||²

Default parameters:
- λ (lambda_tracking) = 1.0
- cost_rate (transaction_cost) = 0.001 (0.1%)

### Neural Network

```
Input (state_dim)
  → FC(256) + ReLU + Dropout(0.2)
  → FC(128) + ReLU + Dropout(0.2)
  → FC(64) + ReLU
  → Output (5 Q-values)
```

### Optimization

- **Algorithm**: Double DQN with experience replay
- **Loss**: Huber Loss (smooth L1)
- **Optimizer**: Adam (lr=1e-4)
- **Replay Buffer**: 100,000 transitions
- **Batch Size**: 64
- **Target Network Update**: Every 1,000 steps
- **Exploration**: ε-greedy (ε: 1.0 → 0.01)
- **Discount Factor**: γ = 0.99

## Key Features

### 1. Double DQN
Reduces overestimation bias by using the policy network to select actions and the target network to evaluate them.

### 2. Experience Replay
Breaks temporal correlations by training on random minibatches from a replay buffer.

### 3. Target Network
Stabilizes training by using a slowly-updated copy of the policy network for computing targets.

### 4. Portfolio Dynamics
Properly models:
- Weight drift due to asset returns
- Transaction costs
- Weight renormalization after returns

## Files

- `dqn.py` - Main DQN implementation and training script
- `visualize_training.py` - Training visualization utilities
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Model Parameters You Can Tune

In `dqn.py`, you can adjust:

### Environment Parameters
```python
env = PortfolioEnvironment(
    transaction_cost=0.001,  # Transaction cost rate (TC)
    lambda_tracking=1.0      # Tracking error penalty weight (λ)
)
```

### Agent Parameters
```python
agent = DQNAgent(
    learning_rate=1e-4,       # Learning rate
    gamma=0.99,               # Discount factor
    epsilon_start=1.0,        # Initial exploration
    epsilon_end=0.01,         # Final exploration
    epsilon_decay=0.995,      # Exploration decay rate
    buffer_size=100000,       # Replay buffer size
    batch_size=64,            # Minibatch size
    target_update_freq=1000   # Target network update frequency
)
```

### Training Parameters
```python
history = train_dqn(
    n_episodes=100,                        # Number of episodes
    max_steps_per_episode=len(returns_train)  # Steps per episode
)
```

## Expected Behavior

During training, you should see:
1. **Initial exploration** - High epsilon, random actions, learning the environment
2. **Learning phase** - Decreasing loss, improving rewards
3. **Convergence** - Stable policy, balanced turnover/tracking error

The agent should learn to:
- Rebalance less frequently when close to target
- Rebalance more aggressively when far from target
- Adapt rebalancing to market volatility
- Balance transaction costs with tracking error

## Next Steps

After training, you can:
1. Evaluate on out-of-sample data (2020+)
2. Compare to baseline strategies (periodic, threshold-based)
3. Analyze learned policy behaviors
4. Tune hyperparameters for better performance
5. Extend to support long/short positions or leverage constraints
