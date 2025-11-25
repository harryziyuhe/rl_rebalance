import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history_path, model_name):
    # Load training history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    episode_rewards = history['episode_rewards']
    episode_turnovers = history['episode_turnovers']
    episode_tracking_errors = history['episode_tracking_errors']

    window = 10  # moving average window

    # Create figure with 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'{model_name} Training History', fontsize=16, fontweight='bold')

    # Plot 1 — Rewards
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.7, label='Episode Reward')
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_rewards)), ma, color='red', linewidth=2,
                label=f'{window}-episode MA')
    ax.set_title("Episode Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 2 — Episode Turnover
    ax = axes[1]
    ax.plot(episode_turnovers, color='green', alpha=0.7, label='Turnover')
    if len(episode_turnovers) >= window:
        ma = np.convolve(episode_turnovers, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_turnovers)), ma, color='red', linewidth=2,
                label=f'{window}-episode MA')
    ax.set_title("Turnover per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 3 — Tracking Error
    ax = axes[2]
    ax.plot(episode_tracking_errors, color='purple', alpha=0.7, label='Tracking Error')
    if len(episode_tracking_errors) >= window:
        ma = np.convolve(episode_tracking_errors, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(episode_tracking_errors)), ma, color='red', linewidth=2,
                label=f'{window}-episode MA')
    ax.set_title("Average Tracking Error")
    ax.set_xlabel("Episode")
    ax.set_ylabel("TE")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_training_history.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved: {model_name}_training_history.png")

if __name__ == "__main__":
    plot_training_history()

