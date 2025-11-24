"""
Model Testing and Evaluation Script

This script loads a trained DQN model and evaluates its performance on test data.
It can also be used to test individual components before full training.

Usage:
    python test_model.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
from dqn import DQNAgent, PortfolioEnvironment, compute_minimum_variance_weights


def test_environment():
    """Test that the environment works correctly"""
    print("Testing Portfolio Environment...")

    n_assets = 5
    n_days = 100

    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2020-01-01', periods=n_days)
    )

    target_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(n_assets), n_days),
        columns=[f'Asset_{i}' for i in range(n_assets)],
        index=pd.date_range('2020-01-01', periods=n_days)
    )

    env = PortfolioEnvironment(
        returns=returns,
        target_weights=target_weights,
        transaction_cost=0.001
    )

    state = env.reset()
    print(f"  Initial state shape: {state.shape}")
    print(f"  State dimension: {len(state)}")
    for i in range(5):
        action = np.random.randint(0, env.n_actions)
        next_state, reward, done, info = env.step(action)
        print(f"  Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")

    print("  ✓ Environment test passed!\n")


def test_agent():
    """Test that the agent can be created and make predictions"""
    print("Testing DQN Agent...")

    state_dim = 20
    action_dim = 5

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4
    )

    dummy_state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(dummy_state, training=False)
    print(f"  Selected action: {action}")

    with torch.no_grad():
        state_tensor = torch.FloatTensor(dummy_state).unsqueeze(0)
        q_values = agent.policy_net(state_tensor)
        print(f"  Q-values shape: {q_values.shape}")
        print(f"  Q-values: {q_values.numpy().flatten()}")

    print("  ✓ Agent test passed!\n")


def evaluate_model(model_path='dqn_portfolio_model.pth', test_start='2020-01-01', test_end='2024-12-31'):
    """
    Evaluate a trained model on test data

    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint
    test_start : str
        Start date for test period
    test_end : str
        End date for test period
    """
    print(f"Evaluating model on test period: {test_start} to {test_end}")

    returns = pd.read_parquet('../data/returns.parquet')
    with open('../data/cov_oas_window252.pkl', 'rb') as f:
        cov_dict = pickle.load(f)

    returns_test = returns.loc[test_start:test_end]
    print(f"Test samples: {len(returns_test)}")

    target_weights_list = []
    for date in returns_test.index:
        if date in cov_dict:
            cov_matrix = cov_dict[date]
            weights = compute_minimum_variance_weights(cov_matrix)
        else:
            weights = np.ones(len(returns_test.columns)) / len(returns_test.columns)
        target_weights_list.append(weights)

    target_weights = pd.DataFrame(
        target_weights_list,
        index=returns_test.index,
        columns=returns_test.columns
    )

    env = PortfolioEnvironment(
        returns=returns_test,
        target_weights=target_weights,
        transaction_cost=0.001
    )

    state_dim = 3 * env.n_assets + 4
    action_dim = env.n_actions

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)

    # Evaluate
    state = env.reset()
    total_reward = 0.0
    total_turnover = 0.0
    total_tracking_error = 0.0
    steps = 0

    actions_taken = []

    while True:
        action = agent.select_action(state, training=False)
        actions_taken.append(action)

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_turnover += info['turnover']
        total_tracking_error += info['tracking_error']
        steps += 1

        state = next_state

        if done:
            break

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward per Step: {total_reward/steps:.4f}")
    print(f"Total Turnover: {total_turnover:.4f}")
    print(f"Average Tracking Error: {total_tracking_error/steps:.6f}")
    print(f"\nAction Distribution:")
    for action_id in range(action_dim):
        count = actions_taken.count(action_id)
        pct = 100 * count / len(actions_taken)
        print(f"  Action {action_id} ({env.action_fractions[action_id]*100:.0f}% rebalance): {count} times ({pct:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("DQN Model Testing")
    print("="*60)
    print()

    test_environment()
    test_agent()

    import os
    if os.path.exists('dqn_portfolio_model.pth'):
        print("Found trained model. Running evaluation...\n")
        try:
            evaluate_model()
        except Exception as e:
            print(f"Evaluation failed: {e}")
            print("Make sure you have test data available in the specified date range.")
    else:
        print("No trained model found. Run 'python dqn.py' to train first.")

    print("\nAll tests completed!")
