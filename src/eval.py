from __future__ import annotations
from re import M
import numpy as np
import pandas as pd

def metrics_from_pnl(
        pnl: np.ndarray,
        tc_bps: np.ndarray | None = None
) -> dict[str, float]:
    r = pnl
    ann = 252
    mu = r.mean() * ann
    sd = r.std(ddof = 1) * np.sqrt(ann)
    sharpe = mu / (sd + 1e-12)
    eq = (1 + r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    max_dd = dd.min()
    out = {
        "ann_ret": float(mu),
        "ann_vol": float(sd),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd)
    }
    if tc_bps is not None:
        out["avg_tc_bps"] = float(np.mean(tc_bps))
    return out

def run_env_policy(env, policy_fn: callable[[dict], np.ndarray]) -> dict:
    obs = env.reset()
    pnl, tc_bps, turnover = [], [], []
    weights = [obs["w"].copy()]

    while True:
        act_idx = policy_fn(obs)
        obs1, rew, done, info = env.step(act_idx)
        port_ret = info["port_ret"]
        pnl.append(port_ret)
        tc_bps.append(info["tc_bps"])
        turnover.append(np.abs(info["delta_w"]).sum())
        weights.append(obs1["w"].copy())
        obs = obs1
        if done:
            break
    
    return {
        "pnl": np.array(pnl),
        "tc_bps": np.array(tc_bps),
        "turnover": np.array(turnover),
        "weights": np.array(weights[:-1])
    }