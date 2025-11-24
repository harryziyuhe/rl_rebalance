from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from eval import *

def _project_bounds(w: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clamp weights to [low, high] (optionally asset-specific)."""
    return np.minimum(high, np.maximum(low, w))


# -----------------------------------------------------------------------------
# Periodic rebalancing baseline
# -----------------------------------------------------------------------------
def simulate_periodic(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        rebalance_freq: str,
        transaction_cost: float,
) -> dict[str, pd.Series]:
    """
    Periodic full rebalance to target weights on a given calendar frequency.

    Parameters
    ----------
    returns : DataFrame
        Asset returns (dates x assets)
    targets : DataFrame
        Target weights (dates x assets), aligned with returns
    rebalance_freq : str
        Pandas offset alias, e.g. 'M' for month-end, 'W-FRI' for weekly
    transaction_cost : float
        Linear TC rate (e.g. 0.001 = 10 bps per 1.0 of turnover)

    Returns
    -------
    dict with keys 'return', 'tc'
    """
    dates = returns.index
    N = returns.shape[1]

    # Start from equal weights (consistent with RL env)
    w = np.ones(N, dtype=float) / N

    port_ret = []
    tc_series = []

    # Rebalance dates from the calendar frequency
    rebal_dates = pd.Series(1, index=dates).resample(rebalance_freq).first().index
    rebal_set = set(rebal_dates)

    # Simple bounds [0, 1]; min-var targets should already respect these
    weight_low = np.zeros(N, dtype=float)
    weight_high = np.ones(N, dtype=float)

    for t in range(len(dates) - 1):
        today = dates[t]
        w_star = targets.iloc[t].values.astype(float)

        if today in rebal_set:
            # Full rebalance to target (with bounds, but no leverage / turnover caps)
            desired = _project_bounds(w_star, weight_low, weight_high)
            delta = desired - w
        else:
            delta = np.zeros(N, dtype=float)

        # Apply trade
        w = w + delta

        # Linear transaction cost
        turnover = np.sum(np.abs(delta))
        tc_cost = transaction_cost * turnover

        # Realize next-day return
        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))
        tc_series.append(tc_cost)

    out = {
        "return": pd.Series(port_ret, index=dates[1:]),
        "tc": pd.Series(tc_series, index=dates[1:])
    }
    return out


# -----------------------------------------------------------------------------
# Band rebalancing baseline
# -----------------------------------------------------------------------------
def simulate_band(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        band: float,
        transaction_cost: float,
) -> dict[str, pd.Series]:
    """
    Band rebalancing: only trade an asset if its deviation from target
    exceeds +/- band. Move it back to the band boundary.

    Parameters
    ----------
    returns : DataFrame
        Asset returns (dates x assets)
    targets : DataFrame
        Target weights (dates x assets), aligned with returns
    band : float
        Band half-width in *weight* units (e.g. 0.02 = 2%)
    transaction_cost : float
        Linear TC rate

    Returns
    -------
    dict with keys 'return', 'tc'
    """
    dates = returns.index
    N = returns.shape[1]

    w = np.ones(N, dtype=float) / N  # start equal-weight
    port_ret, tc_series = [], []

    weight_low = np.zeros(N, dtype=float)
    weight_high = np.ones(N, dtype=float)

    for t in range(len(dates) - 1):
        w_star = targets.iloc[t].values.astype(float)
        dev = w - w_star
        desired = w.copy()

        # If deviation > band, push part-way to target band boundary
        for i in range(N):
            if dev[i] > band:
                # overweight vs target → sell down to (target + band)
                desired[i] = w_star[i] + band
            elif dev[i] < -band:
                # underweight → buy up to (target - band)
                desired[i] = w_star[i] - band

        desired = _project_bounds(desired, weight_low, weight_high)
        delta = desired - w
        w = w + delta

        turnover = np.sum(np.abs(delta))
        tc_cost = transaction_cost * turnover

        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))
        tc_series.append(tc_cost)

    return {
        "return": pd.Series(port_ret, index=dates[1:]),
        "tc": pd.Series(tc_series, index=dates[1:])
    }

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

    rows.append(
        ("Periodic_M",
         metrics_from_pnl(res_periodic_M["return"].values,
                          tc_bps=10000 * res_periodic_M["tc"].values))
    )
    rows.append(
        ("Band_2pct",
         metrics_from_pnl(res_band_2["return"].values,
                          tc_bps=10000 * res_band_2["tc"].values))
    )
    rows.append(
        ("Band_4pct",
         metrics_from_pnl(res_band_4["return"].values,
                          tc_bps=10000 * res_band_4["tc"].values))
    )

    strategies = [name for name, _ in rows]
    metrics = pd.DataFrame([m for _, m in rows], index=strategies)

    print()
    print("Baseline performance (train period):")
    print(metrics.round(4))
    print()