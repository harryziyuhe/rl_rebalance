import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pickle
import os

def plot_training_results():
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(current_dir, 'mmc_training_history_100.pkl')
    
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found. Run training first.")
        return

    print("Loading training history...")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # Helper to get key with legacy fallback
    def get_metric(key, legacy_key, default_val=None):
        if key in history:
            return history[key]
        elif legacy_key in history:
            return history[legacy_key]
        elif default_val is not None:
            n = len(history.get('episode_rewards', history.get('rewards', [])))
            return [default_val] * n
        else:
            return [] # Should ideally not happen if history is valid

    episode_rewards = get_metric('episode_rewards', 'rewards')
    episode_losses = get_metric('episode_losses', 'losses')
    episode_turnovers = get_metric('episode_turnovers', 'turnovers', default_val=0.0)
    episode_tracking_errors = get_metric('episode_tracking_errors', 'tracking_errors', default_val=0.0)

    # Handle legacy history files without balances
    if 'final_balances' in history:
        balances = pd.Series(history['final_balances'])
    else:
        balances = None

    # Create figure with subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monte Carlo Agent Training Performance (100 Episodes)', fontsize=16, fontweight='bold')
    
    window = 10

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, linewidth=1.5, alpha=0.7, label='Episode Reward')
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
    save_path = os.path.join(current_dir, 'mmc_training_plot_100.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"4-panel plot saved successfully to {save_path}")

    # Also plot portfolio value if available
    if balances is not None:
        plot_portfolio_value(balances, current_dir, window)

def plot_portfolio_value(balances, output_dir, window):
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    plt.axhline(y=100000, color='gray', linestyle='--', label='Initial Investment ($100k)')
    
    # Plot balances
    plt.plot(balances, color='green', alpha=0.6, label='Final Portfolio Value')
    
    # Add smoothed trend
    balances_smooth = balances.rolling(window=window, min_periods=1).mean()
    plt.plot(balances_smooth, color='darkgreen', linewidth=2, label=f'Trend (MA {window})')
    
    plt.title('Portfolio Value Growth over Training (100 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format Y-axis as currency
    ax = plt.gca()
    formatter = mticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mmc_portfolio_value_100.png')
    plt.savefig(save_path, dpi=300)
    print(f"Portfolio value plot saved successfully to {save_path}")

if __name__ == "__main__":
    plot_training_results()
