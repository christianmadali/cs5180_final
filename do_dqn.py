import pprint
import gym
import highway_env
import os

import torch

import torch.nn.functional as F

import torch.nn as nn

def run(filename):
    t = ""
    with open(filename) as f:
        t = f.read()
    return t

env = gym.make("intersection-v0")
pprint.pprint(env.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

exec(run('dqn.py'))

# plot
plt.figure()
plt.plot(avg_loss)
plt.grid()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("double_dqn_loss.png", dpi=150)
plt.show()

plt.figure()
plt.plot(avg_reward_list)
plt.grid()
plt.xlabel("*40 epochs")
plt.ylabel("reward")
plt.savefig(time_str + "/double_dqn_train_reward.png", dpi=150)
plt.show()
