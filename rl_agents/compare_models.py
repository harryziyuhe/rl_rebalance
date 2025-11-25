import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import os
import sys

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ppo_agent import PPOAgent
from mmc_agent import MonteCarloAgent
from dqn import DQNAgent, PortfolioEnvironment, compute_minimum_variance_weights

# ---------------------------------------------------------
# Comparison Script
# ---------------------------------------------------------

def run_comparison():
    print("=" * 60)
    print("Comparing PPO, DQN, and Baselines (Periodic / Band)")
    print("=" * 60)

    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    returns_path = os.path.join(data_dir, 'returns.parquet')
    cov_path = os.path.join(data_dir, 'cov_oas_window252.pkl')
    
    if not os.path.exists(returns_path) or not os.path.exists(cov_path):
        print("Data not found.")
        return

    returns = pd.read_parquet(returns_path)
    with open(cov_path, 'rb') as f:
        cov_dict = pickle.load(f)
    
    # We'll test on the SAME period we trained on (for now) to see how well they learned
    # Or you can split: train 2010-2017, test 2018-2019.
    # Let's use the full 2010-2019 for this visual comparison as per training.
    start_date = '2010-01-01'
    end_date = '2019-12-31'
    
    returns_test = returns.loc[start_date:end_date]
    print(f"Period: {start_date} to {end_date}")
    
    # Compute Targets
    target_weights_list = []
    for date in returns_test.index:
        if date in cov_dict:
            weights = compute_minimum_variance_weights(cov_dict[date])
        else:
            weights = np.ones(len(returns_test.columns)) / len(returns_test.columns)
        target_weights_list.append(weights)
        
    target_weights = pd.DataFrame(
        target_weights_list, 
        index=returns_test.index, 
        columns=returns_test.columns
    )

    # -----------------------------------------------------
    # Initialize Environment
    # -----------------------------------------------------
    env = PortfolioEnvironment(
        returns=returns_test,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=1.0
    )
    
    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions
    
    # -----------------------------------------------------
    # 1. PPO Agent Evaluation
    # -----------------------------------------------------
    print("\nRunning PPO Agent...")
    # Models are expected to be in the same directory as this script
    ppo_model_path = os.path.join(current_dir, 'ppo_portfolio_model.pth')
    if os.path.exists(ppo_model_path):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ppo_agent = PPOAgent(state_dim, action_dim, device=device)
        ppo_agent.load(ppo_model_path)
        
        state = env.reset()
        ppo_values = [1.0]
        ppo_actions = []
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(device)
                logits, _ = ppo_agent.policy(state_t)
                action = logits.argmax().item()
            
            ppo_actions.append(action)
            next_state, _, done, info = env.step(action)
            ppo_values.append(ppo_values[-1] * (1 + info['portfolio_return']))
            state = next_state
        print(f"PPO Action Dist: {pd.Series(ppo_actions).value_counts(normalize=True).sort_index().to_dict()}")
    else:
        print(f"Warning: {ppo_model_path} not found. Skipping PPO.")
        ppo_values = []

    # -----------------------------------------------------
    # 2. DQN Agent Evaluation
    # -----------------------------------------------------
    print("\nRunning DQN Agent...")
    dqn_model_path = os.path.join(current_dir, 'dqn_portfolio_model.pth')
    if os.path.exists(dqn_model_path):
        dqn_agent = DQNAgent(state_dim, action_dim)
        # If device mismatch (cuda vs cpu), handle inside agent or force load map_location
        # dqn_agent uses 'cuda' if available by default.
        try:
            dqn_agent.load(dqn_model_path)
            
            state = env.reset()
            dqn_values = [1.0]
            dqn_actions = []
            done = False
            while not done:
                action = dqn_agent.select_action(state, training=False)
                dqn_actions.append(action)
                next_state, _, done, info = env.step(action)
                dqn_values.append(dqn_values[-1] * (1 + info['portfolio_return']))
                state = next_state
            print(f"DQN Action Dist: {pd.Series(dqn_actions).value_counts(normalize=True).sort_index().to_dict()}")
        except Exception as e:
            print(f"Error loading DQN: {e}")
            dqn_values = []
    else:
        print(f"Warning: {dqn_model_path} not found. Skipping DQN.")
        dqn_values = []

    # -----------------------------------------------------
    # 3. MCMC Agent Evaluation
    # -----------------------------------------------------
    print("\nRunning MCMC Agent...")
    mmc_model_path = os.path.join(current_dir, 'mmc_portfolio_model.pth')
    if os.path.exists(mmc_model_path):
        # MCMC agent uses the same network structure as DQN (QNetwork from dqn.py)
        # Assuming MonteCarloAgent stores network state dict
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        mmc_agent = MonteCarloAgent(state_dim, action_dim, device=device)
        try:
            # MonteCarloAgent.load loads state_dict into policy_net
            # Need to ensure load method exists and works, assuming standard
            # Let's check if load expects full path or what
            # The snippet of mmc_agent.py shows:
            # def save(self, path): torch.save(self.policy_net.state_dict(), path)
            # So we need a load method. The snippet didn't show a load method in MonteCarloAgent class!
            # We can manually load it.
            mmc_agent.policy_net.load_state_dict(torch.load(mmc_model_path, map_location=device))
            
            state = env.reset()
            mmc_values = [1.0]
            mmc_actions = []
            done = False
            while not done:
                action = mmc_agent.select_action(state, epsilon=0.0) # Greedy
                mmc_actions.append(action)
                next_state, _, done, info = env.step(action)
                mmc_values.append(mmc_values[-1] * (1 + info['portfolio_return']))
                state = next_state
            print(f"MCMC Action Dist: {pd.Series(mmc_actions).value_counts(normalize=True).sort_index().to_dict()}")
        except Exception as e:
            print(f"Error loading MCMC: {e}")
            mmc_values = []
    else:
        print(f"Warning: {mmc_model_path} not found. Skipping MCMC.")
        mmc_values = []

    # -----------------------------------------------------
    # 3. Baselines (Periodic & Equal Weight)
    # -----------------------------------------------------
    print("\nCalculating Baselines...")
    
    # Equal Weight (Buy & Hold / Daily Rebal approximation)
    # Actually env starts with 1/N. If we take action 4 (100% rebal) every step -> MinVar
    # If we take action 0 (0% rebal) every step -> Buy & Hold (mostly)
    
    # Let's simulate MinVariance Target (Theoretical limit ignoring costs/tracking error constraints slightly)
    # Or just use the env with fixed action.
    
    # Periodic (Monthly) - approximated by rebalancing every 21 days
    state = env.reset()
    periodic_values = [1.0]
    done = False
    steps = 0
    while not done:
        if steps % 21 == 0:
            action = 4 # Full rebalance
        else:
            action = 0 # Hold
        
        next_state, _, done, info = env.step(action)
        periodic_values.append(periodic_values[-1] * (1 + info['portfolio_return']))
        state = next_state
        steps += 1
        
    # Min-Var Daily (Theoretical ideal without costs)
    # Just rebalance 100% every day
    state = env.reset()
    minvar_values = [1.0]
    done = False
    while not done:
        action = 4 # Full rebalance every day
        next_state, _, done, info = env.step(action)
        minvar_values.append(minvar_values[-1] * (1 + info['portfolio_return']))
        state = next_state

    # Buy and Hold (Action 0)
    state = env.reset()
    bnh_values = [1.0]
    done = False
    while not done:
        action = 0 # No rebalance
        next_state, _, done, info = env.step(action)
        bnh_values.append(bnh_values[-1] * (1 + info['portfolio_return']))
        state = next_state

    # -----------------------------------------------------
    # Comparison Stats
    # -----------------------------------------------------
    print("\nComparison Statistics (Final Normalized Value):")
    if ppo_values: print(f"  PPO: {ppo_values[-1]:.4f}")
    if dqn_values: print(f"  DQN: {dqn_values[-1]:.4f}")
    if mmc_values: print(f"  MCMC: {mmc_values[-1]:.4f}")
    if periodic_values: print(f"  Periodic: {periodic_values[-1]:.4f}")
    if minvar_values: print(f"  Min-Var (Daily): {minvar_values[-1]:.4f}")
    if bnh_values: print(f"  Buy & Hold: {bnh_values[-1]:.4f}")

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    print("\nGenerating Comparison Plot...")
    dates = returns_test.index
    
    plt.figure(figsize=(12, 8))
    
    # Ensure lengths match
    plot_dates = dates[:len(ppo_values)-1] if len(ppo_values) > 1 else dates
    
    if ppo_values:
        plt.plot(plot_dates, ppo_values[:-1], label='PPO Agent', linewidth=2)
    
    if dqn_values:
        # Cut to match dates if needed
        plt.plot(plot_dates, dqn_values[:len(plot_dates)], label='DQN Agent', linewidth=2, linestyle='--')

    if mmc_values:
        plt.plot(plot_dates, mmc_values[:len(plot_dates)], label='MCMC Agent', linewidth=2, linestyle='-.')
        
    if periodic_values:
        plt.plot(plot_dates, periodic_values[:len(plot_dates)], label='Periodic (Monthly)', linewidth=1.5, linestyle=':')
        
    if minvar_values:
         plt.plot(plot_dates, minvar_values[:len(plot_dates)], label='Daily Min-Var (High Cost)', linewidth=1, alpha=0.5)

    plt.title('Portfolio Performance Comparison (2010-2019)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = 'model_comparison.png'
    plt.savefig(out_path, dpi=300)
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    run_comparison()

