import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import agent and env
from ppo_agent import PPOAgent
# Import env from dqn (or ppo_agent if it re-exports/wraps it, but ppo_agent imports from dqn)
try:
    from dqn import PortfolioEnvironment, compute_minimum_variance_weights
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(current_dir, '..'))
    from rl_agents.dqn import PortfolioEnvironment, compute_minimum_variance_weights

def plot_training_history(history_path='ppo_training_history.pkl'):
    """
    Load and plot PPO training history
    """
    if not os.path.exists(history_path):
        print(f"History file {history_path} not found.")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    rewards = history.get('rewards', [])
    turnovers = history.get('turnovers', [])
    tracking_errors = history.get('tracking_errors', [])
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('PPO Portfolio Agent Training History', fontsize=16)

    # Window for moving average
    window = 10

    # 1. Rewards
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.6, label='Episode Reward', color='blue')
    if len(rewards) >= window:
        ma_rewards = pd.Series(rewards).rolling(window=window).mean()
        ax1.plot(ma_rewards, color='darkblue', linewidth=2, label=f'{window}-Ep MA')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Turnover
    ax2 = axes[1]
    ax2.plot(turnovers, alpha=0.6, label='Episode Turnover', color='green')
    if len(turnovers) >= window:
        ma_turnover = pd.Series(turnovers).rolling(window=window).mean()
        ax2.plot(ma_turnover, color='darkgreen', linewidth=2, label=f'{window}-Ep MA')
    ax2.set_title('Cumulative Turnover')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Turnover')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Tracking Error
    ax3 = axes[2]
    ax3.plot(tracking_errors, alpha=0.6, label='Avg Tracking Error', color='purple')
    if len(tracking_errors) >= window:
        ma_te = pd.Series(tracking_errors).rolling(window=window).mean()
        ax3.plot(ma_te, color='indigo', linewidth=2, label=f'{window}-Ep MA')
    ax3.set_title('Average Tracking Error')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'ppo_training_history.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Training history plot saved to {plot_path}")
    # plt.show() # Commented out for headless environments

def run_backtest(model_path='ppo_portfolio_model.pth'):
    """
    Run a backtest of the trained PPO agent on the training data
    to visualize portfolio performance.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    # --- Load Data (Same as training for now) ---
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    returns_path = os.path.join(data_dir, 'returns.parquet')
    cov_path = os.path.join(data_dir, 'cov_oas_window252.pkl')
    
    print("Loading data for backtest...")
    returns = pd.read_parquet(returns_path)
    with open(cov_path, 'rb') as f:
        cov_dict = pickle.load(f)
        
    train_start = '2010-01-01'
    train_end = '2019-12-31'
    returns_train = returns.loc[train_start:train_end]
    
    # Compute Targets
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
    
    # Setup Env
    env = PortfolioEnvironment(
        returns=returns_train,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=1.0
    )
    
    # Load Agent
    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading PPO agent from {model_path} on {device}...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    agent.load(model_path)
    
    # Run Episode
    print("Running backtest episode...")
    state = env.reset()
    done = False
    
    portfolio_values = [1.0] # Normalized start at 1.0
    daily_returns = []
    actions = []
    turnovers = []
    
    while not done:
        # Deterministic action selection (use mode of distribution or just sample?)
        # For PPO, usually we use mean/mode for evaluation. 
        # But our select_action samples.
        # Let's modify select_action locally or just sample. 
        # Given discrete actions, sampling is fine, but technically we might want the argmax of logits.
        
        # Get logits directly to pick max probability action
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            logits, _ = agent.policy(state_t)
            action = logits.argmax().item()
        
        next_state, reward, done, info = env.step(action)
        
        # Track metrics
        # info['portfolio_return'] is the return for this step
        ret = info['portfolio_return']
        daily_returns.append(ret)
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        actions.append(action)
        turnovers.append(info['turnover'])
        
        state = next_state

    # Plot Performance
    dates = returns_train.index
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Portfolio Value
    ax1 = axes[0]
    # Ensure dates and values match length
    plot_dates = dates[:len(portfolio_values)]
    ax1.plot(plot_dates, portfolio_values, label='PPO Agent', color='blue') 
    # Add Equal Weight Baseline for comparison?
    # Simple equal weight daily rebalance approximation (or buy and hold)
    # Let's just plot the agent for now.
    ax1.set_title('Portfolio Growth (Normalized)')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Actions/Rebalancing Intensity
    ax2 = axes[1]
    ax2.scatter(dates[:len(actions)], actions, alpha=0.5, s=10, color='orange')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax2.set_title('Rebalancing Actions over Time')
    ax2.set_ylabel('Rebalance Intensity')
    ax2.grid(True, alpha=0.3)
    
    # 3. Turnover
    ax3 = axes[2]
    ax3.plot(dates[:len(turnovers)], turnovers, color='green', linewidth=0.5)
    ax3.set_title('Daily Turnover')
    ax3.set_ylabel('Turnover')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    perf_plot_path = 'ppo_performance.png'
    plt.savefig(perf_plot_path, dpi=300)
    print(f"Performance plot saved to {perf_plot_path}")

if __name__ == "__main__":
    # 1. Plot Training History
    plot_training_history()
    
    # 2. Run Backtest and Plot Performance
    run_backtest()

