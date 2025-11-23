from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd

def _project_bounds(w: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(high, np.maximum(low, w))

def _project_leverage(w: np.ndarray, lev_cap: float) -> np.ndarray:
    gross = np.sum(np.abs(w))
    if gross <= lev_cap + 1e-12:
        return w
    return w * (lev_cap / gross)

def _project_turnover(delta: np.ndarray, cap: float) -> np.ndarray:
    turn = np.sum(np.abs(delta))
    if turn <= cap + 1e-12:
        return delta
    return delta * (cap / turn)

def simulate_periodic(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        rebalance_freq: str,
        weight_low: np.ndarray,
        weight_high: np.ndarray,
        leverage_cap: float,
        daily_turnover_cap: float,
        tc_c: np.ndarray,
        tc_k: np.ndarray
) -> dict[str, pd.Series]:
    dates = returns.index
    N = returns.shape[1]
    w = np.zeros(N)
    port_ret = []
    tc_series = []

    rebal_dates = pd.Series(1, index=dates).resample(rebalance_freq).first().index
    rebal_set = set(rebal_dates)

    for t in range(len(dates) - 1):
        today = dates[t]
        tomorrow = dates[t + 1]
        w_star = targets.iloc[t].values.astype(float)

        if today in rebal_set:
            desired = _project_bounds(w_star, weight_low, weight_high)
            desired = _project_leverage(desired, leverage_cap)
            delta = desired - w
            delta = _project_turnover(delta, daily_turnover_cap)
        else:
            delta = np.zeros(N)

        w = w + delta

        from .costs import tc_linear_sqrt
        sigma_t = returns.rolling(20).std().iloc[t].fillna(0.0).values.astype(float)
        tc_cost = tc_linear_sqrt(delta, tc_c, tc_k, sigma_t)

        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))
        tc_series.append(tc_cost)
    
    out = {
        "return": pd.Series(port_ret, index = dates[1:]),
        "tc": pd.Series(tc_series, index = dates[1:])
    }
    return out

def simulate_band(
        returns: pd.DataFrame,
        targets: pd.DataFrame,
        band: float,
        weight_low: np.ndarray,
        weight_high: np.ndarray,
        leverage_cap: float,
        daily_turnover_cap: float,
        tc_c: np.ndarray,
        tc_k: np.ndarray
) -> dict[str, pd.Series]:
    dates = returns.index
    N = returns.shape[1]
    w = np.zeros(N)
    port_ret, tc_series = [], []

    for t in range(len(dates) - 1):
        w_star = targets.iloc[t].values.astype(float)
        dev = w - w_star
        desired = w.copy()

        for i in range(N):
            if dev[i] > band:
                desired[i] = w_star[i] + band
            elif dev[i] < -band:
                desired[i] = w_star[i] - band
        
        desired = _project_bounds(desired, weight_low, weight_high)
        desired = _project_leverage(desired, leverage_cap)
        delta = desired - w
        delta = _project_turnover(delta, daily_turnover_cap)
        w =  w + delta

        from .costs import tc_linear_sqrt
        sigma_t = returns.rolling(20).std().iloc[t].fillna(0.0).values.astype(float)
        tc_cost = tc_linear_sqrt(delta, tc_c, tc_k, sigma_t)

        r_tp1 = returns.iloc[t + 1].values.astype(float)
        port_ret.append(float(np.dot(w, r_tp1) - tc_cost))
        tc_series.append(tc_cost)

    return {
        "return": pd.Series(port_ret, index = dates[1:]),
        "tc": pd.Series(tc_series, index = dates[1:])
    }
