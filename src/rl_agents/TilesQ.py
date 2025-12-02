import numpy as np
import pandas as pd
import pickle
import random
import os

class PortfolioEnvironment:
    """
    Portfolio Rebalancing Environment

    Simulates portfolio dynamics with rebalancing decisions under transaction costs.

    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns (dates � assets)
    target_weights : pd.DataFrame
        Target portfolio weights from minimum-variance optimization (dates � assets)
    transaction_cost : float
        Proportional transaction cost rate (e.g., 0.001 = 0.1%)
    lambda_tracking : float
        Penalty weight for tracking error (lambda in the formula)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        target_weights: pd.DataFrame,
        transaction_cost: float = 0.001,
        lambda_tracking: float = 1.0
    ):
        self.returns = returns
        self.target_weights = target_weights
        self.transaction_cost = transaction_cost
        self.lambda_tracking = lambda_tracking

        self.dates = returns.index.intersection(target_weights.index)
        self.returns = returns.loc[self.dates]
        self.target_weights = target_weights.loc[self.dates]

        self.n_assets = len(returns.columns)
        self.n_actions = 5  # 0%, 25%, 50%, 75%, 100% rebalancing

        # Action mapping: what fraction of the gap to close
        self.action_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]

        self.reset()

    def reset(self, start_idx: int = 0) -> np.ndarray:
        """
        Reset the environment to initial state

        Parameters:
        -----------
        start_idx : int
            Starting index in the data

        Returns:
        --------
        np.ndarray
            Initial state vector
        """
        self.current_idx = start_idx
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.days_since_rebalance = 0

        self.portfolio_volatility = 0.0
        self.portfolio_return = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Construct the state vector

        Returns:
        --------
        np.ndarray
            State vector containing:
            - Current weights (n_assets)
            - Target weights (n_assets)
            - Weight deviations (n_assets)
            - Tracking error (1)
            - Portfolio volatility (1)
            - Portfolio return (1)
            - Days since rebalance (1)
        """
        target_w = self.target_weights.iloc[self.current_idx].values
        weight_dev = self.current_weights - target_w
        tracking_error = np.sqrt(np.sum(weight_dev ** 2))

        state = np.concatenate([
            self.current_weights,
            target_w,
            weight_dev,
            [tracking_error],
            [self.portfolio_volatility],
            [self.portfolio_return],
            [self.days_since_rebalance / 100.0]  # Normalize days
        ])

        return state.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment

        Parameters:
        -----------
        action : int
            Action index (0-4)

        Returns:
        --------
        next_state : np.ndarray
            State after taking the action
        reward : float
            Immediate reward
        done : bool
            Whether the episode is finished
        info : dict
            Additional information (turnover, tracking_error, etc.)
        """
        target_w = self.target_weights.iloc[self.current_idx].values

        rebalance_fraction = self.action_fractions[action]
        weight_change = rebalance_fraction * (target_w - self.current_weights)
        new_weights_pre_return = self.current_weights + weight_change

        turnover = np.sum(np.abs(weight_change))

        transaction_cost_incurred = turnover * self.transaction_cost

        self.current_idx += 1
        done = (self.current_idx >= len(self.dates) - 1)

        if not done:
            period_returns = self.returns.iloc[self.current_idx].values
            portfolio_return = np.sum(new_weights_pre_return * period_returns)
            self.current_weights = new_weights_pre_return * (1 + period_returns)
            self.current_weights = self.current_weights / np.sum(self.current_weights)
            self.portfolio_return = portfolio_return
            target_w_new = self.target_weights.iloc[self.current_idx].values
            tracking_error = np.sqrt(np.sum((self.current_weights - target_w_new) ** 2))

            reward = (
                portfolio_return                                    # wR
                - transaction_cost_incurred                         # TC(Δwt)
                - self.lambda_tracking * (tracking_error ** 2)      # λ(dt)²
            )

            if turnover > 0.001:  # Threshold for considering it a rebalance
                self.days_since_rebalance = 0
            else:
                self.days_since_rebalance += 1

            next_state = self._get_state()
        else:
            reward = 0.0
            next_state = self._get_state()
            tracking_error = 0.0
            portfolio_return = 0.0

        info = {
            'turnover': turnover,
            'tracking_error': tracking_error,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost_incurred
        }

        return next_state, reward, done, info

def compute_minimum_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute minimum variance portfolio weights

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix (n_assets * n_assets)

    Returns:
    --------
    np.ndarray
        Portfolio weights that minimize variance
    """
    n = len(cov_matrix)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n)
        w = inv_cov @ ones
        w = w / np.sum(w)
        w = np.maximum(w, 0)
        w = w / np.sum(w)
        return w
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    
class TileCoder:
    def __init__(
            self,
            state_dim: int,
            num_tilings: int = 8,
            tiles_per_dim: int = 8,
            n_features: int = 4096,
            low: np.ndarray | None = None,
            high: np.ndarray | None = None
    ):
        self.staet_dim = state_dim
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_features = n_features

        if low is None:
            self.low = -np.ones(state_dim, dtype = float)
        if high is None:
            self.high = np.ones(state_dim, dtype = float)
        
        self.low = low.astype(float)
        self.high = high.astype(float)
        self.range = self.high - self.low
        self.tile_width = self.range / self.tiles_per_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype = float)
        s = np.minimum(self.high, np.maximum(self.low, s))

        feature_indices = []
        for tiling in range(self.num_tilings):
            offset = (tiling / self.num_tilings) * self.tile_width
            rel = (s - self.low + offset) / self.range
            rel = np.clip(rel, 0.0, 0.999999)
            bins = (rel * self.tiles_per_dim).astype(int)  # 0 .. tiles_per_dim-1

            # Hash (tiling, bins) to a feature index
            key = (tiling, *bins.tolist())
            idx = hash(key) % self.n_features
            feature_indices.append(idx)

        return np.array(feature_indices, dtype=int)
    
class TileQAgent:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            alpha: float = 1e-3,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            num_tilings: int = 8,
            tiles_per_dim: int = 8,
            n_features: int = 4096,
            state_low: np.ndarray | None = None,
            state_high: np.ndarray | None = None
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        if state_low is None:
            state_low = -np.ones(state_dim, dtype = float)
        if state_high is None:
            state_high = np.ones(state_dim, dtype = float)

        self.tile_coder = TileCoder(
            state_dim = state_dim,
            num_tilings = num_tilings,
            tiles_per_dim = tiles_per_dim,
            n_features = n_features,
            low = state_low,
            high = state_high
        )

        self.n_features = n_features
        self.weights = np.zeros((action_dim, n_features), dtype = float)
        self.steps = 0

    def _q_values(self, state: np.ndarray) -> np.ndarray:
        feats = self.tile_coder.encode(state)
        return np.sum(self.weights[:, feats], axis = 1)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q_vals = self._q_values(state)
        return int(np.argmax(q_vals))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        feats = self.tile_coder.encode(state)
        q_sa = np.sum(self.weights[action, feats])

        if done:
            target = reward
        else:
            q_next = self._q_values(next_state)
            target = reward + self.gamma * np.max(q_next)
        
        td_err = target - q_sa
        
        self.weights[action, feats] += (self.alpha * td_err / self.tile_coder.num_tilings)

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        obj = {
            "weights": self.weights,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "n_features": self.n_features,
            "action_dim": self.action_dim
        }
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        print(f"TileQAgent saved to {filepath}")

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        self.weights = obj["weights"]
        self.epsilon = obj["epsilon"]
        self.steps = obj["steps"]
        print(f"TileQAgent loaded from {filepath}")

def train_tile_q(
        agent: TileQAgent,
        env: PortfolioEnvironment,
        n_episodes: int = 100,
        max_steps_per_episode: int | None = None,
        verbose: bool = True
) -> dict:
    episode_rewards = []
    episode_turnovers = []
    episode_tracking_errors = []

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_turnover = 0.0
        ep_tracking = 0.0
        steps = 0

        while True:
            action = agent.select_action(state, training = True)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            ep_reward += reward
            ep_turnover += info.get("turnover", 0.0)
            ep_tracking += info.get("tracking_error", 0.0)
            steps += 1

            state = next_state

            if done:
                break
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break

        episode_rewards.append(ep_reward)
        episode_turnovers.append(ep_turnover)
        episode_tracking_errors.append(ep_tracking / max(steps, 1))

        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}")
            print(f"  Total Reward: {ep_reward:.4f}")
            print(f"  Turnover: {ep_turnover:.4f}")
            print(f"  Avg Tracking Error: {episode_tracking_errors[-1]:.6f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Steps: {steps}")
            print()
        
    history = {
        "episode_rewards": episode_rewards,
        "episode_turnovers": episode_turnovers,
        "episode_tracking_errors": episode_tracking_errors,
    }
    return history
        
if __name__ == "__main__":
    print("=" * 80)
    print("Tile-coded Q-learning Portfolio Rebalancing Training")
    print("=" * 80)
    print()

    print("Loading data...")
    returns = pd.read_parquet("data/returns.parquet")
    prices = pd.read_parquet("data/prices.parquet")

    with open("data/cov_oas_window252.pkl", "rb") as f:
        cov_dict = pickle.load(f)

    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print()

    train_start = "2010-01-01"
    train_end = "2019-12-31"

    returns_train = returns.loc[train_start:train_end]
    print(f"Training period: {returns_train.index.min()} to {returns_train.index.max()}")
    print(f"Training samples: {len(returns_train)}")
    print()

    # ------------------------------------------------------------------
    # Build minimum-variance targets (or you can load precomputed targets)
    # ------------------------------------------------------------------
    print("Computing minimum variance target weights...")
    target_weights_list = []

    for date in returns_train.index:
        if date in cov_dict:
            cov_matrix = cov_dict[date]
            weights = compute_minimum_variance_weights(cov_matrix)
        else:
            weights = np.ones(len(returns_train.columns)) / len(returns_train.columns)
        target_weights_list.append(weights)

    target_weights = pd.DataFrame(
        target_weights_list,
        index=returns_train.index,
        columns=returns_train.columns,
    )
    print("Target weights computed.")
    print()

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    print("Creating portfolio environment...")
    env = PortfolioEnvironment(
        returns=returns_train,
        target_weights=target_weights,
        transaction_cost=0.001,  # 0.1% per unit turnover
        lambda_tracking=1.0,     # tracking error penalty
    )

    # Infer state dimension from env._get_state()
    init_state = env.reset()
    state_dim = int(init_state.shape[0])
    action_dim = env.n_actions

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of assets: {env.n_assets}")
    print()

    # Approximate state bounds for tile coder
    # (weights & targets in [0,1], deviations in [-1,1], tracking error, vol, returns ~[-0.2,0.2], days_since_rebalance/100 in [0,1])
    state_low = np.full(state_dim, -1.0, dtype=float)
    state_high = np.full(state_dim, 1.0, dtype=float)

    # ------------------------------------------------------------------
    # Create TileQAgent
    # ------------------------------------------------------------------
    print("Creating TileQ agent...")
    agent = TileQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        num_tilings=8,
        tiles_per_dim=8,
        n_features=4096,
        state_low=state_low,
        state_high=state_high,
    )
    print()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Starting TileQ training...")
    history = train_tile_q(
        agent=agent,
        env=env,
        n_episodes=100,
        max_steps_per_episode=len(returns_train),
        verbose=True,
    )

    print("Saving TileQ agent and training history...")
    agent.save("model/tileq_portfolio_model.pkl")
    with open("model/tileq_training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print()
    print("=" * 80)
    print("TileQ Training Complete!")
    print("=" * 80)