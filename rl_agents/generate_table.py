import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

# Import Utils & Baselines
from src.eval import metrics_from_pnl
from src.baselines import simulate_periodic, simulate_band

# Import Agents
from rl_agents.dqn import DQNAgent, PortfolioEnvironment, compute_minimum_variance_weights
from rl_agents.ppo_agent import PPOAgent
from rl_agents.mmc_agent import MonteCarloAgent
# Try importing TileQ
try:
    from src.TilesQ.tiles_q import TileQAgent
    TILEQ_AVAILABLE = True
except ImportError:
    TILEQ_AVAILABLE = False

def evaluate_strategy(pnl, tc_bps):
    return metrics_from_pnl(np.array(pnl), np.array(tc_bps))

def run_agent_backtest(agent, env, agent_type='dqn', device='cpu'):
    state = env.reset()
    pnl = []
    tc_bps = []
    
    done = False
    while not done:
        # Select Action
        if agent_type == 'dqn':
            action = agent.select_action(state, training=False)
        elif agent_type == 'ppo':
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(device)
                logits, _ = agent.policy(state_t)
                action = logits.argmax().item()
        elif agent_type == 'mmc':
            action = agent.select_action(state, epsilon=0.0)
        elif agent_type == 'tileq':
            action = agent.select_action(state, training=False)
        
        # Step
        next_state, _, done, info = env.step(action)
        
        # Calculate Net Return and Cost
        gross_ret = info['portfolio_return']
        cost = info['transaction_cost']
        
        # Assuming portfolio value is ~1.0 for cost calc relative to return
        net_ret = gross_ret - cost
        cost_bps = cost * 10000
        
        pnl.append(net_ret)
        tc_bps.append(cost_bps)
        
        state = next_state
        
    return pnl, tc_bps

def main():
    print("Generating Performance Table (Train Period: 2010-2019)...")
    
    # 1. Load Data
    data_dir = os.path.join(root_dir, 'data')
    returns = pd.read_parquet(os.path.join(data_dir, 'returns.parquet'))
    with open(os.path.join(data_dir, 'cov_oas_window252.pkl'), 'rb') as f:
        cov_dict = pickle.load(f)
        
    train_start = '2010-01-01'
    train_end = '2019-12-31'
    returns_train = returns.loc[train_start:train_end]
    
    # 2. Compute Targets
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
    
    # 3. Initialize Environment
    env = PortfolioEnvironment(
        returns=returns_train,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=1.0
    )
    
    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    results = []
    
    # --- Baselines ---
    print("Running Baselines...")
    
    # Band 2%
    res_band_2 = simulate_band(returns_train, target_weights, band=0.02, transaction_cost=0.001)
    m_band2 = evaluate_strategy(res_band_2['return'], res_band_2['tc'] * 10000) # tc is cost, *10000 for bps
    results.append(('Band_2%', m_band2))
    
    # Band 4%
    res_band_4 = simulate_band(returns_train, target_weights, band=0.04, transaction_cost=0.001)
    m_band4 = evaluate_strategy(res_band_4['return'], res_band_4['tc'] * 10000)
    results.append(('Band_4%', m_band4))
    
    # Monthly
    res_monthly = simulate_periodic(returns_train, target_weights, rebalance_freq='M', transaction_cost=0.001)
    m_monthly = evaluate_strategy(res_monthly['return'], res_monthly['tc'] * 10000)
    results.append(('Monthly', m_monthly))
    
    # --- DQN ---
    print("Running DQN...")
    dqn_path = os.path.join(current_dir, 'dqn_portfolio_model.pth')
    if os.path.exists(dqn_path):
        dqn_agent = DQNAgent(state_dim, action_dim)
        try:
            dqn_agent.load(dqn_path)
            pnl, tc = run_agent_backtest(dqn_agent, env, 'dqn')
            m_dqn = evaluate_strategy(pnl, tc)
            results.append(('DQN', m_dqn))
        except Exception as e:
            print(f"DQN Error: {e}")
    
    # --- MMC ---
    print("Running MMC...")
    mmc_path = os.path.join(current_dir, 'mmc_portfolio_model.pth')
    if os.path.exists(mmc_path):
        mmc_agent = MonteCarloAgent(state_dim, action_dim, device=device)
        try:
            mmc_agent.policy_net.load_state_dict(torch.load(mmc_path, map_location=device))
            pnl, tc = run_agent_backtest(mmc_agent, env, 'mmc')
            m_mmc = evaluate_strategy(pnl, tc)
            results.append(('MMC', m_mmc))
        except Exception as e:
             print(f"MMC Error: {e}")

    # --- PPO ---
    print("Running PPO...")
    ppo_path = os.path.join(current_dir, 'ppo_portfolio_model.pth')
    if os.path.exists(ppo_path):
        ppo_agent = PPOAgent(state_dim, action_dim, device=device)
        try:
            ppo_agent.load(ppo_path)
            pnl, tc = run_agent_backtest(ppo_agent, env, 'ppo', device)
            m_ppo = evaluate_strategy(pnl, tc)
            results.append(('PPO', m_ppo))
        except Exception as e:
            print(f"PPO Error: {e}")
            
    # --- TileQ ---
    if TILEQ_AVAILABLE:
        print("Running TileQ...")
        tileq_path = os.path.join(root_dir, 'src', 'TilesQ', 'tileq_portfolio_agent.pkl')
        if os.path.exists(tileq_path):
            # Init with same params as tiles_q.py
            state_low = np.full(state_dim, -1.0, dtype=float)
            state_high = np.full(state_dim, 1.0, dtype=float)
            tileq_agent = TileQAgent(
                state_dim=state_dim, 
                action_dim=action_dim,
                num_tilings=8,
                tiles_per_dim=8,
                n_features=4096,
                state_low=state_low,
                state_high=state_high
            )
            try:
                tileq_agent.load(tileq_path)
                pnl, tc = run_agent_backtest(tileq_agent, env, 'tileq')
                m_tileq = evaluate_strategy(pnl, tc)
                results.append(('TileQ', m_tileq))
            except Exception as e:
                print(f"TileQ Error: {e}")
    
    # --- Output Table ---
    print("\n\nPerformance Table (Train Period)")
    print(f"{'Strategy':<12} {'Ann Ret':<8} {'Ann Vol':<8} {'Sharpe':<8} {'Max DD':<8} {'Avg TC (bps)':<12}")
    print("-" * 65)
    
    for name, m in results:
        print(f"{name:<12} {m['ann_ret']:.4f}   {m['ann_vol']:.4f}   {m['sharpe']:.4f}   {m['max_dd']:.4f}   {m['avg_tc_bps']:.4f}")

if __name__ == "__main__":
    main()

