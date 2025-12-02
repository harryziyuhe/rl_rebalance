import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

def plot_metric(series, title, ylabel, filename, color, window=None):
    plt.figure(figsize=(10, 6))
    
    # Determine window size dynamically if not provided
    # Match the reference image logic: ~10% of data for short runs, but max out for long runs
    if window is None:
        if len(series) < 200:
            window = max(5, int(len(series) * 0.1))
        else:
            window = 100

    # Raw data
    plt.plot(series, color=color, alpha=0.4, label='Raw Data')
    
    # Smoothed data (Red line like the reference image)
    if len(series) >= window:
        smoothed = series.rolling(window=window, min_periods=1).mean()
        plt.plot(smoothed, color='tab:red', linewidth=2, label=f'Moving Avg ({window} eps)')
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

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

    # Extract metrics
    rewards = pd.Series(history.get('rewards', []))
    losses = pd.Series(history.get('losses', []))
    turnovers = pd.Series(history.get('turnovers', []))
    tracking_errors = pd.Series(history.get('tracking_errors', []))

    print(f"Found {len(rewards)} episodes.")

    # 1. Reward Plot (Blue raw, Red MA)
    plot_metric(
        rewards, 
        'MMC Training Reward per Episode', 
        'Total Reward', 
        os.path.join(current_dir, 'mmc_reward.png'),
        'tab:blue'
    )

    # 2. Loss Plot (Orange raw, Red MA)
    plot_metric(
        losses, 
        'MMC Training Loss (MSE)', 
        'Loss', 
        os.path.join(current_dir, 'mmc_loss.png'),
        'orange'
    )

    # 3. Turnover Plot (Green raw, Red MA)
    if not turnovers.empty:
        plot_metric(
            turnovers, 
            'MMC Portfolio Turnover per Episode', 
            'Turnover', 
            os.path.join(current_dir, 'mmc_turnover.png'),
            'tab:green'
        )
    else:
        print("No turnover data found in history.")

    # 4. Tracking Error Plot (Purple raw, Red MA)
    if not tracking_errors.empty:
        plot_metric(
            tracking_errors, 
            'MMC Average Tracking Error per Episode', 
            'Tracking Error', 
            os.path.join(current_dir, 'mmc_tracking_error.png'),
            'tab:purple'
        )
    else:
        print("No tracking error data found in history.")

if __name__ == "__main__":
    plot_training_results()
