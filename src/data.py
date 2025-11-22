from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import OAS, LedoitWolf

@dataclass
class FetchConfig:
    tickers = ["SPY", "QQQ", "EFA", "EWJ", "EEM", "IAU"]
    start = "2005-02-01"
    end = None
    out_dir = "C:/Users/zih028/Documents/GitHub/rl_rebalance/data"
    fname_prices = "prices.parquet"
    fname_returns = "returns.parquet"

def _ensure_dir(path) -> None:
    os.makedirs(path, exist_ok=True)

def fetch_prices(cfg: FetchConfig) -> pd.DataFrame:
    """
    Fetch Adjusted Close for given tickers.
    """
    data = yf.download(cfg.tickers,
                       start = cfg.start,
                       end = cfg.end,
                       auto_adjust = True,
                       progress = False)
    return data["Close"].dropna()

def to_log_returns(prices) -> pd.DataFrame:
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets

def rolling_shrunk_cov(returns, 
                       window = 252, 
                       shrink = "oas", 
                       min_obs = 60):
    """
    Compute rolling shrunk covariance matrices.
    """
    covs = []
    fit_cls = OAS if shrink == "oas" else LedoitWolf

    for end_idx in range(len(returns)):
        end_date = returns.index[end_idx]
        start_idx = max(0, end_idx - window + 1)
        window_slice = returns.iloc[start_idx: end_idx + 1]
        if len(window_slice) < min_obs:
            covs.append((end_date, np.full((returns.shape[1], returns.shape[1]), np.nan)))
            continue
        model = fit_cls().fit(window_slice.values)
        covs.append((end_date, model.covariance_))

    cov_series = pd.Series({d: c for d, c in covs}).sort_index()
    return cov_series

def save_frames(cfg: FetchConfig, prices, returns) -> None:
    _ensure_dir(cfg.out_dir)
    prices.to_parquet(os.path.join(cfg.out_dir, cfg.fname_prices))
    returns.to_parquet(os.path.join(cfg.out_dir, cfg.fname_returns))

def load_prices(cfg: FetchConfig) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(cfg.out_dir, cfg.fname_prices))

def load_returns(cfg: FetchConfig) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(cfg.out_dir, cfg.fname_returns))

if __name__ == "__main__":
    cfg = FetchConfig()
    px = fetch_prices(cfg)
    rets = to_log_returns(px)
    save_frames(cfg, px, rets)
    cov_series = rolling_shrunk_cov(rets)
    cov_path = os.path.join(cfg.out_dir, "cov_oas_window252.pkl")
    cov_series.to_pickle(cov_path)
