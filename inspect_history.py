import pickle
import numpy as np
import os

def inspect_pickle(path, name):
    if not os.path.exists(path):
        print(f"{name}: File not found at {path}")
        return
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"--- {name} ({path}) ---")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, (list, np.ndarray)):
                    if len(v) > 0:
                        print(f"  {k}: len={len(v)}, mean={np.mean(v):.4f}, last={v[-1]:.4f}")
                    else:
                        print(f"  {k}: empty")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0 and isinstance(data[0], (int, float)):
                print(f"  Mean: {np.mean(data):.4f}, Last: {data[-1]:.4f}")
        else:
            print(f"Type: {type(data)}")
    except Exception as e:
        print(f"Error reading {name}: {e}")
    print()

inspect_pickle('ppo_training_history.pkl', 'Root PPO History')
inspect_pickle('rl_agents/ppo_training_history.pkl', 'RL_Agents PPO History')
inspect_pickle('rl_agents/training_history.pkl', 'RL_Agents General History')

