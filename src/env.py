from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

@dataclass
class EnvConfig:
    action_grid: Iterable[float]
    daily_overturn_cap: float
    weight_lower: np.ndarray
    weight_upper: np.ndarray
    leverage_cap: float = 1.0
    risk_penalty: float = 0.0
    te_penalty: float = 10.0
    exec_next_open: bool = True

class RebalanceEnv:
    def __init__(
            self,
            returns: pd.DataFrame,
            sigma_diag: pd.DataFrame,
            covmats: dict[pd.Timestamp, np.ndarray],
            targets: pd.DataFrame,
            tc_c: np.ndarray,
            tc_k: np.ndarray,
            config: EnvConfig
    ):
        self.assets = list(returns.columns)
        self.N = len(self.assets)

        self.rets= returns.sort_index()
        self.sig = sigma_diag.reindex_like(self.rets)
        self.targets = targets.reindex_like(self.rets)
        self.covmats = covmats
        self.cfg = config

        self.tc_c = np.asarray(tc_c, dtype = float)
        self.tc_k = np.asarray(tc_k, dtype = float)
        
        self.action_grid = np.asarray(list(config.action_grid), dtype=float)
        self.num_actions = len(self.action_grid)

        self.dates = self.rets.index.to_list()
        self.T = len(self.dates)
        self.t = 0
        self.w = np.zeros(self.N, dtype = float)
        self._pending_trade = np.zeros(self.N, dtype = float)
    
    def _project_bounds(self, w: np.ndarray) -> np.ndarray:
        return np.minimum(self.cfg.weight_upper, np.maximum(self.cfg.weight_lower, w))
    
    def _project_leverage(self, w: np.ndarray) -> np.ndarray:
        gross = np.sum(np.abs(w))
        if gross <= self.cfg.leverage_cap + 1e-12:
            return w
        return w * (self.cfg.leverage_cap / gross)
    
    def _project_turnover_cap(self, delta_w: np.ndarray, cap: float) -> np.ndarray:
        turn = np.sum(np.abs(delta_w))
        if turn <= cap + 1e-12:
            return delta_w
        return delta_w * (cap / turn)
    
    def _apply_trade(self, w: np.ndarray, delta_w: np.ndarray) -> np.ndarray:
        w_new = w + delta_w
        w_new = self._project_bounds(w_new)
        w_new = self._project_leverage(w_new)
        return w_new
    
    def _reward(
            self,
            w_before: np.ndarray,
            w_after: np.ndarray,
            w_star: np.ndarray,
            ret_next: np.ndarray,
            sigma_today: np.ndarray,
            cov_today: np.ndarray,
            tc_cost: float
    ) -> float:
        port_ret = np.dot(w_after, ret_next)
        risk_term = self.cfg.risk_penalty * float(w_after.T @ cov_today @ w_after)
        te_term = self.cfg.te_penalty * float(np.sum((w_after - w_star) ** 2))

        return port_ret - tc_cost - risk_term - te_term
    
    def reset(self, w0: np.ndarray | None = None, start_idx: int = 0):
        self.t = start_idx
        self.w = np.zeros(self.N) if w0 is None else np.asarray(w0, dtype=float)
        self._pending_trade[:] = 0.0
        return self._obs(), {"date": self.dates[self.t], "w": self.w.copy()}
    
    def _obs(self) -> np.ndarray:
        w_star = self.targets.iloc[self.t].values.astype(float)
        deviations = self.w - w_star
        cov = self.covmats[self.dates[self.t]]
        sleeve_vol = np.sqrt(float(self.w.T @ cov @ self.w))
        obs = np.concatenate([deviations, [sleeve_vol]])
        return obs
    
    def step(self, action_idx: np.ndarray):
        if self.t >= self.T - 1:
            raise RuntimeError("Episode already finished")
        
        date_t = self.dates[self.t]
        date_tp1 = self.dates[self.t + 1]

        w_star_t = self.targets.iloc[self.t].values.astype(float)
        ret_tp1 = self.rets.iloc[self.t + 1].values.astype(float)
        sigma_t = self.sig.iloc[self.t].values.astype(float)
        cov_t = self.covmats[date_t]

        raw_delta = self.action_grid[np.asarray(action_idx, dtype = int)]
        delta = self._project_turnover_cap(raw_delta, self.cfg.daily_overturn_cap)
        w_after = self._apply_trade(self.w, delta)
        
        from .costs import tc_linear_sqrt
        tc_cost = tc_linear_sqrt(delta_w = delta, c = self.tc_c, k = self.tc_k, sigma = sigma_t)

        reward = self._reward(self.w, w_after, w_star_t, ret_tp1, sigma_t, cov_t, tc_cost)

        self.w = w_after.copy()
        self.t += 1
        done = (self.t >= self.T - 1)
        obs = None if done else self._obs()
        info = {
            "date": date_tp1, 
            "w": self.w.copy(),
            "tc": tc_cost, 
            "delta": delta.copy()
        }

        return obs, float(reward), done, info
