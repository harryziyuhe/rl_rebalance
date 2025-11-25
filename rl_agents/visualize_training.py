"""
Training Visualization Script

This script loads and visualizes the training history from the DQN portfolio rebalancing model.
It generates plots showing:
- Episode rewards over time
- Training loss progression
- Cumulative turnover
- Average tracking error

Usage:
    python visualize_training.py
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history_path='training_history.pkl'):
    """
    Load and plot training history

    Parameters:
    -----------
    history_path : str
        Path to the training history pickle file
    """
    # Load history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    episode_rewards = history['episode_rewards']
    episode_losses = history['episode_losses']
    episode_turnovers = history['episode_turnovers']
    episode_tracking_errors = history['episode_tracking_errors']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Portfolio Rebalancing Training History', fontsize=16, fontweight='bold')

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, linewidth=1.5, alpha=0.7, label='Episode Reward')
    # Add moving average
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg,
                linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = axes[0, 1]
    ax2.plot(episode_losses, linewidth=1.5, alpha=0.7, color='orange', label='Avg Loss per Episode')
    if len(episode_losses) >= window:
        moving_avg_loss = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_losses)), moving_avg_loss,
                linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Turnover
    ax3 = axes[1, 0]
    ax3.plot(episode_turnovers, linewidth=1.5, alpha=0.7, color='green', label='Episode Turnover')
    if len(episode_turnovers) >= window:
        moving_avg_turnover = np.convolve(episode_turnovers, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(episode_turnovers)), moving_avg_turnover,
                linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Turnover')
    ax3.set_title('Cumulative Turnover per Episode')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Average Tracking Error
    ax4 = axes[1, 1]
    ax4.plot(episode_tracking_errors, linewidth=1.5, alpha=0.7, color='purple', label='Avg Tracking Error')
    if len(episode_tracking_errors) >= window:
        moving_avg_te = np.convolve(episode_tracking_errors, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(episode_tracking_errors)), moving_avg_te,
                linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Avg Tracking Error')
    ax4.set_title('Average Tracking Error per Episode')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved to 'training_history.png'")
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary Statistics")
    print("="*60)
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(episode_rewards):.4f}")
    print(f"  Std: {np.std(episode_rewards):.4f}")
    print(f"  Min: {np.min(episode_rewards):.4f}")
    print(f"  Max: {np.max(episode_rewards):.4f}")
    print(f"\nLoss:")
    print(f"  Mean: {np.mean(episode_losses):.6f}")
    print(f"  Final 10 episodes avg: {np.mean(episode_losses[-10:]):.6f}")
    print(f"\nTurnover:")
    print(f"  Mean: {np.mean(episode_turnovers):.4f}")
    print(f"  Total: {np.sum(episode_turnovers):.4f}")
    print(f"\nTracking Error:")
    print(f"  Mean: {np.mean(episode_tracking_errors):.6f}")
    print(f"  Final 10 episodes avg: {np.mean(episode_tracking_errors[-10:]):.6f}")
    print("="*60)


if __name__ == "__main__":
    plot_training_history("/Users/georgiikuznetsov/Desktop/rl_rebalance/rl_agents/training_history.pkl")
