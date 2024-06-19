import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

class D4RL_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        data = d4rl.qlearning_dataset(gym.make(args.env))
        self.device = args.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).view(-1).float().to(self.device)
        self.is_finished = torch.from_numpy(data['terminals']).view(-1).float().to(self.device)

        if "antmaze" in args.env:
            reward_tune = "iql_antmaze"
        else:
            reward_tune = "iql_locomotion"
        
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = torch.where(reward > 0, 0.0, -1.0)
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("dql dataloard loaded")
        
        self.len = self.states.shape[0]
        print(self.len, "data loaded")

    def __getitem__(self, index):
        data = {'s': self.states[index % self.len],
                'a': self.actions[index % self.len],
                'r': self.rewards[index % self.len],
                's_':self.next_states[index % self.len],
                'd': self.is_finished[index % self.len],
            }
        return data

    def __add__(self, other):
        pass
    def __len__(self):
        return self.len