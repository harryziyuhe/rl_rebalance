from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pnl(pnl: np.ndarray, label: str = "strategy"):
    eq = (1 + pnl).cumprod()
    plt.figure()
    plt.plot(eq, label = label)
    plt.title("Equity Curve")
    plt.legend()
    plt.tight_layout()

def plot_drawdown(pnl: np.ndarray):
    eq = (1 + pnl).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) + 1
    plt.figure()
    plt.plot(dd)
    plt.title("Drawdown")
    plt.tight_layout()

def plt_turnover_te_frontier(
        turnover: np.ndarray,
        te: np.ndarray,
        labels: np.ndarray | None = None   
):
    plt.figure()
    if labels is None:
        plt.scatter(te, turnover)
    else:
        for x, y, s, in zip(te, turnover, labels):
            plt.scatter(x, y)
            plt.annotate(s, (x, y))
    plt.xlabel("Tracking Error")
    plt.ylabel("Turnover")
    plt.title("Turnover vs Tracking Error Frontier")
    plt.tight_layout()

def policy_heatmap_2d(
        dev_i: np.ndarray,
        dev_j: np.ndarray,
        actions: np.ndarray,
        bins: int = 21,
        title: str = "Policy Heatmap"
):
    H, xedges, yedges = np.histogram2d(dev_i, dev_j, bins = bins)
    A = np.full((bins, bins), np.nan)
    xi = np.clip(np.digitize(dev_i, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(dev_j, yedges) - 1, 0, bins - 1)
    for i in range(bins):
        for j in range(bins):
            mask = (xi == i) & (yi == j)
            if mask.any():
                A[j,i] = actions[mask].mean()
    plt.figure()
    plt.imshow(A, origin = "lower", aspect = "auto",
               extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label = "Average Action")
    plt.xlabel("Deviation asset i")
    plt.ylabel("Deviation asset j")
    plt.title(title)
    plt.tight_layout()