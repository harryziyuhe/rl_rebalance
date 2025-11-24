import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pickle
import os

def plot_training_results():
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(current_dir, 'mmc_training_history.pkl')
    
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found. Run training first.")
        return

    print("Loading training history...")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    rewards = pd.Series(history['rewards'])
    losses = pd.Series(history['losses'])
    # Handle legacy history files without balances
    if 'final_balances' in history:
        balances = pd.Series(history['final_balances'])
    else:
        balances = None

    # Rolling average for smoothness
    window = 100
    rewards_smooth = rewards.rolling(window=window, min_periods=1).mean()
    losses_smooth = losses.rolling(window=window, min_periods=1).mean()

    # Plotting 1: Reward & Loss
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (Smoothed)', color=color)
    ax1.plot(rewards_smooth, color=color, label='Reward (MA 100)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Loss (MSE) (Smoothed)', color=color)
    ax2.plot(losses_smooth, color=color, linestyle='--', label='Loss (MA 100)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Monte Carlo Agent Training Performance (10,000 Episodes)')
    fig.tight_layout()
    
    save_path = os.path.join(current_dir, 'mmc_training_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")

    # Plotting 2: Portfolio Value (if available)
    if balances is not None:
        plot_portfolio_value(balances, current_dir)

def plot_portfolio_value(balances, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    plt.axhline(y=100000, color='gray', linestyle='--', label='Initial Investment ($100k)')
    
    # Plot balances
    plt.plot(balances, color='green', alpha=0.6, label='Final Portfolio Value')
    
    # Add smoothed trend
    window = 100
    balances_smooth = balances.rolling(window=window, min_periods=1).mean()
    plt.plot(balances_smooth, color='darkgreen', linewidth=2, label=f'Trend (MA {window})')
    
    plt.title('Portfolio Value Growth over Training')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format Y-axis as currency
    ax = plt.gca()
    formatter = mticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mmc_portfolio_value.png')
    plt.savefig(save_path, dpi=300)
    print(f"Portfolio value plot saved successfully to {save_path}")

if __name__ == "__main__":
    plot_training_results()

