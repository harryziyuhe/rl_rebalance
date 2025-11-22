# rl_rebalance
Reinforcement Learning Rebalancing to a Moving Minimum-Variance Target

The goal of the project is model dynamic portfolio rebalancing under transaction costs. The target allocation each day is the minimum-variance portfolio from a shrunk rolling covariance. Rebalancing is costly, so moving to target everyday is sub-optimal. I train a reinforcement learning agent to learn state-dependent no-trade regions and partial-rebalance actions that minimize expected shortfall and turnover, subject to bounds and daily turnover budget. I compare RL to period, band, and myopic baselines and visualize learned policies, turnover-tracking-error frontiers, and out-of-sample performances.
