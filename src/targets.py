from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def _project_simplex_with_bounds(x0: np.ndarray,
                                 lb: np.ndarray,
                                 ub: np.ndarray) -> np.ndarray:
    x = np.clip(x0, lb, ub)
    s = x.sum()
    if np.isclose(s, 1.0):
        return x
    diff = 1.0 - s
    cap = (ub - x) if diff > 0 else (x - lb)
    weight = cap / (np.sum(cap) + 1e-12)
    x = x + diff * weight
    return np.clip(x, lb, ub)

def minvar_target(
        Sigma: np.ndarray,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        w_init: np.ndarray | None = None
) -> np.ndarray:
    n = Sigma.shape[0]
    if lb is None: lb = np.zeros(n)
    if ub is None: ub = np.ones(n)
    if w_init is None:
        iv = 1.0 / (np.sqrt(np.clip(np.diag(Sigma), 1e-12, None)))
        w_init = iv / iv.sum()
        w_init = _project_simplex_with_bounds(w_init, lb, ub)
    
    if not np.isfinite(Sigma).all():
        return _project_simplex_with_bounds(w_init, lb, ub)
    
    def obj(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)
    
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    )

    bounds = tuple((float(lb[i]), float(ub[i])) for i in range(n))
    res = minimize(obj, w_init, method = "SLSQP", bounds = bounds, constraints = cons, options={"maxiter": 500})
    if not res.success:
        return _project_simplex_with_bounds(w_init, lb, ub)
    w= res.x
    w = _project_simplex_with_bounds(w, lb, ub)
    return w

def rolling_min_var_targets(
        cov_series: pd.Series,
        tickers: list[str],
        lb_map: dict[str, float] | None = None,
        ub_map: dict[str, float] | None = None
) -> pd.DataFrame:
    n = len(tickers)
    if lb_map is None: lb_map = {t: 0.0 for t in tickers}
    if ub_map is None: ub_map = {t: 1.0 for t in tickers}
    lb = np.array([lb_map[t] for t in tickers], dtype = float)
    ub = np.array([ub_map[t] for t in tickers], dtype = float)

    rows = []
    for dt, Sigma in cov_series.items():
        if Sigma is None or not isinstance(Sigma, np.ndarray) or Sigma.shape != (n, n):
            rows.append((dt, np.full(n, np.nan)))
            continue
        w = minvar_target(Sigma, lb = lb, ub = ub)
        rows.append((dt, w))

    W = pd.DataFrame({dt: w for dt, w in rows}).T
    W.columns = tickers
    return W.dropna()