from __future__ import annotations

import numpy as np

def linear_sqrt_cost(
        delta_w: np.ndarray,
        lin_bps: np.ndarray,
        k_impact: np.ndarray,
        sigma_daily: np.ndarray
) -> float:
    abs_dw = np.abs(delta_w)
    tc_bps = lin_bps * abs_dw + k_impact * sigma_daily * np.sqrt(abs_dw + 1e-12)
    tc_frac = (tc_bps / 1e4).sum()
    return float(tc_frac)

def project_turnover_cap(delta_w: np.ndarray, cap: float) -> np.ndarray:
    tw = np.abs(delta_w).sum()
    if tw <= cap + 1e-12:
        return delta_w
    scale = cap / (tw + 1e-12)
    return delta_w * scale

def apply_bounds_and_simplex(w_current: np.ndarray,
                             delta_w: np.ndarray,
                             lb: np.ndarray,
                             ub: np.ndarray) -> np.ndarray:
    w_new = w_current + delta_w
    w_new = np.clip(w_new, lb, ub)
    s = w_new.sum()
    if s == 0:
        w_new = np.full_like(w_new, 1.0 / len(w_new))
    else:
        w_new = w_new / s
    
    return w_new