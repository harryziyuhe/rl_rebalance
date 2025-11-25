from __future__ import annotations
import numpy as np
import pandas as pd
import pickle


def simulate_periodic(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        rebalance_freq: str,
        tc_rate: float
) -> dict[str, pd.Series]:

    dates = returns.index
    N = returns.shape[1]

    # Start fully in cash or zeros? RL env starts uniform, but baselines usually start flat.
    w = np.ones(N, dtype=float) / N

    port_ret = []
    tc_series = []
    weights_list = []

    # Determine rebalance dates
    rebal_dates = pd.Series(1, index=dates).resample(rebalance_freq).first().index
    rebal_set = set(rebal_dates)

    for t in range(len(dates) - 1):

        today = dates[t]
        w_star = targets.iloc[t].values.astype(float)

        # --- Rebalance only on schedule ---
        if today in rebal_set:
            desired = w_star
            delta = desired - w
        else:
            delta = np.zeros(N)

        # --- Update weight after trades ---
        w = w + delta
        weights_list.append(w.copy())

        # --- Transaction cost (linear only) ---
        tc_cost = tc_rate * np.sum(np.abs(delta))
        tc_series.append(tc_cost)

        # --- Portfolio return next day ---
        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))

    weights_df = pd.DataFrame(
        np.vstack(weights_list),
        index = dates[1:],
        columns = returns.columns
    )

    df = weights_df
    df["pnl"] = port_ret
    df["tc"] = tc_series

    return df

def simulate_band(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        band: float,
        tc_rate: float
) -> dict[str, pd.Series]:

    dates = returns.index
    N = returns.shape[1]

    w = np.ones(N, dtype=float) / N
    weights_list = []
    port_ret = []
    tc_series = []

    for t in range(len(dates) - 1):

        w_star = targets.iloc[t].values.astype(float)
        dev = w - w_star

        # Target adjustment based on deviation exceeding the band
        desired = w.copy()
        for i in range(N):
            if dev[i] > band:
                desired[i] = w_star[i] + band
            elif dev[i] < -band:
                desired[i] = w_star[i] - band

        desired = desired
        delta = desired - w

        # Update weights
        w = w + delta
        weights_list.append(w.copy())

        # Linear transaction cost
        tc_cost = tc_rate * np.sum(np.abs(delta))
        tc_series.append(tc_cost)

        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))

    weights_df = pd.DataFrame(
        np.vstack(weights_list),
        index = dates[1:],
        columns = returns.columns
    )

    df = weights_df
    df["pnl"] = port_ret
    df["tc"] = tc_series

    return df

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
    print("Running periodic monthly baseline...")
    res_periodic_M = simulate_periodic(
        returns=returns_train,
        targets=target_weights,
        rebalance_freq="M",
        transaction_cost=0.001
    )

    print("Running band baseline (±2%)...")
    res_band_2 = simulate_band(
        returns=returns_train,
        targets=target_weights,
        band=0.02,
        transaction_cost=0.001
    )

    print("Running band baseline (±4%)...")
    res_band_4 = simulate_band(
        returns=returns_train,
        targets=target_weights,
        band=0.04,
        transaction_cost=0.001
    )

    rows = []

    strategies = [name for name, _ in rows]
    metrics = pd.DataFrame([m for _, m in rows], index=strategies)

    print()
    print("Baseline performance (train period):")
    print(metrics.round(4))
    print()