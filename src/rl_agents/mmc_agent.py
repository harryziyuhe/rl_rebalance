import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# need to make sure dqn.py is in the same folder
from .dqn import PortfolioEnvironment, DQN as QNetwork, compute_minimum_variance_weights

class MonteCarloAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, device="cpu"):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma=gamma
        self.device=device
        
        # using the same network structure as the DQN agent so we can compare them fairly
        self.policy_net=QNetwork(state_dim, action_dim).to(device)
        self.optimizer=optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # list to store everything that happens in an episode
        self.episode_memory=[]

    def select_action(self, state, epsilon=0.0):
        # epsilon-greedy strategy
        rand_val=random.random()
        if rand_val<epsilon:
            # pick random action
            action=random.randint(0, self.action_dim-1)
            return action
        else:
            # use the network
            with torch.no_grad():
                # convert state to tensor
                state_t=torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values=self.policy_net(state_t)
                # get the index of the max q value
                action=int(q_values.argmax(dim=1).item())
                return action

    def store_transition(self, state, action, reward):
        # just append to the list
        self.episode_memory.append((state, action, reward))

    def update_policy(self):
        # if memory is empty, don't do anything
        if len(self.episode_memory)==0:
            return 0.0

        # unpack the memory manually
        states=[]
        actions=[]
        rewards=[]
        for step in self.episode_memory:
            states.append(step[0])
            actions.append(step[1])
            rewards.append(step[2])

        # Calculate returns (G_t)
        # we have to go backwards from the end of the episode
        G=0.0
        returns=[]
        
        # reverse the rewards list to iterate backwards
        for r in reversed(rewards):
            G=float(r)+self.gamma*G
            # insert at the front to keep the order correct
            returns.insert(0, G)

        # convert everything to pytorch tensors
        states_t=torch.FloatTensor(np.array(states)).to(self.device)
        actions_t=torch.LongTensor(actions).to(self.device)
        returns_t=torch.FloatTensor(returns).to(self.device)

        # get what the network predicts for these states
        q_values=self.policy_net(states_t) 
        q_taken=q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # calculate loss: difference between predicted Q and actual Return G
        criterion=nn.MSELoss()
        loss=criterion(q_taken, returns_t)

        self.optimizer.zero_grad()
        loss.backward()
        
        # clip grads to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()

        # clear memory for the next episode
        self.episode_memory=[]
        
        return float(loss.item())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)


# helper function to get the target weights
def get_targets(returns_df, covariances):
    n_assets=len(returns_df.columns)
    target_weights_list=[]
    
    # loop through every date
    for date in returns_df.index:
        # check if we have data for this date
        if date in covariances:
            cov=covariances[date]
            # make sure cov matrix is valid
            if isinstance(cov, np.ndarray) and np.isfinite(cov).all():
                w=compute_minimum_variance_weights(cov)
                # verify weights are okay
                if np.isfinite(w).all():
                    target_weights_list.append(w)
                else:
                    # fallback to equal weights
                    target_weights_list.append(np.ones(n_assets)/n_assets)
            else:
                target_weights_list.append(np.ones(n_assets)/n_assets)
        else:
            # fallback if no date found
            target_weights_list.append(np.ones(n_assets)/n_assets)
        
    # make it a dataframe
    target_weights=pd.DataFrame(
        target_weights_list,
        index=returns_df.index,
        columns=returns_df.columns
    )
    return target_weights


def train_mmc():
    # fix paths
    current_file_path=os.path.abspath(__file__)
    current_dir=os.path.dirname(current_file_path)
    # assuming data is two levels up (src/MMC/../../data)
    data_dir=os.path.join(current_dir, '..', '..', 'data')
    
    returns_file=os.path.join(data_dir, 'returns.parquet')
    cov_file=os.path.join(data_dir, 'cov_oas_window252.pkl')

    print(f"Looking for data in: {data_dir}")
    
    if not os.path.exists(returns_file):
        print("Error: returns.parquet not found")
        return
    if not os.path.exists(cov_file):
        print("Error: cov_oas_window252.pkl not found")
        return

    # load data
    print("Loading datasets...")
    returns_df=pd.read_parquet(returns_file)
    with open(cov_file, 'rb') as f:
        covariances=pickle.load(f)

    # get the min var weights
    print("Calculating target weights (this might take a second)...")
    target_weights=get_targets(returns_df, covariances)
    print("Done calculating targets.")

    # setup env
    env=PortfolioEnvironment(
        returns=returns_df,
        target_weights=target_weights,
        transaction_cost=0.001,
        lambda_tracking=1.0
    )

    # setup agent
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # get dims from env
    temp_state=env.reset()
    state_dim=len(temp_state)
    action_dim=env.n_actions

    agent=MonteCarloAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3, # standard lr
        gamma=0.99,
        device=device
    )

    # Hyperparams
    n_episodes=50 # Reduced for testing
    epsilon=1.0
    epsilon_min=0.01
    epsilon_decay=0.9995 # slow decay since we have lots of episodes
    
    # history storage
    history={
        'rewards': [], 
        'losses': [], 
        'turnovers': [],
        'tracking_errors': []
    }

    print("Starting training loop...")
    
    for i in range(n_episodes):
        state=env.reset()
        done=False
        total_reward=0.0
        total_turnover=0.0
        total_tracking_error=0.0
        steps=0

        while not done:
            # check for bad states
            if np.isnan(state).any():
                print(f"Got NaNs in state at episode {i}. Breaking.")
                break
                
            # get action
            action=agent.select_action(state, epsilon)
            
            # take step
            next_state, reward, done, info=env.step(action)
            
            # check for weird rewards
            if not np.isfinite(reward):
                reward=-10.0 # penalty for breaking things
            
            # store for MC update later
            agent.store_transition(state, action, reward)
            
            state=next_state
            total_reward+=float(reward)
            total_turnover+=info.get('turnover', 0.0)
            total_tracking_error+=info.get('tracking_error', 0.0)
            steps+=1

        # episode is over, update policy
        loss=agent.update_policy()

        # decay epsilon
        if epsilon>epsilon_min:
            epsilon=epsilon*epsilon_decay
        else:
            epsilon=epsilon_min

        # save stats
        history['rewards'].append(total_reward)
        history['losses'].append(loss)
        history['turnovers'].append(total_turnover)
        # Average tracking error per step is usually more meaningful, but let's store total for now or average?
        # DQN stores average: episode_tracking_errors.append(episode_tracking_error / max(steps, 1))
        # Let's match DQN
        history['tracking_errors'].append(total_tracking_error / max(steps, 1))

        # print progress every now and then
        if (i+1)%500==0:
            print(f"Ep {i+1} | Reward: {total_reward:.4f} | Loss: {loss:.5f} | Eps: {epsilon:.3f}")

    # save everything
    print("Saving model and history...")
    save_path=os.path.join(current_dir, 'mmc_portfolio_model.pth')
    agent.save(save_path)
    
    history_path=os.path.join(current_dir, 'mmc_training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
        
    print("All done.")

if __name__ == "__main__":
    train_mmc()