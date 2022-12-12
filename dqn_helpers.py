import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time


import torch

import torch.nn.functional as F

import torch.nn as nn

class Net(nn.Module):
    def __init__(self, state_dimen, action_dimen):
        super(Net, self).__init__()
        # hidden_nodes1 = 1024
        # hidden_nodes2 = 512
        hidden_nodes1 = 512
        hidden_nodes2 = 256
        # hidden_nodes1 = 2048
        # hidden_nodes2 = 1024
        self.func1 = nn.Linear(state_dimen, hidden_nodes1)
        self.func2 = nn.Linear(hidden_nodes1, hidden_nodes2)
        self.func3 = nn.Linear(hidden_nodes2, action_dimen)

    def forward(self, state):
        x = state
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        res = self.func3(x)
        return res

class Replay:
    def __init__(self, state_dimen, action_dimen, buffer_size, init_length, env):

        self.state_dimen = state_dimen
        self.action_dimen = action_dimen
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.env = env

        self._store = []
        self._init_buffer(init_length)

    def buffer_sample(self, N): #sample from buffer
        sample = random.sample(self._store, N)
        return sample

    def buffer_add(self, experience): #add to buffer
        self._store.append(experience)
        if len(self._store) > self.buffer_size:
            self._store.pop(0)

    def _init_buffer(self, n): #initialize buffer

        state,_ = self.env.reset()
        for _ in range(n):
            action = self.env.action_space.sample()
            next_state, reward, done, _, _ = self.env.step(action)
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done}
            self._store.append(exp)
            state = next_state

            if done:
                state,_ = self.env.reset()
                done = False
