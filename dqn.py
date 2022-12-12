import gym
import highway_env

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

import dqn_helpers as helper


class DDQN(nn.Module):
    def __init__(
        self,
        env,
        state_dimen,
        action_dimen,
        lr=0.002,
        gamma=0.98,
        batch_size=5,
        timestamp="",
    ):
        super(DDQN, self).__init__()

        # use code from helper to initialize
        self.replay_buffer = helper.Replay(self.state_dimen, self.action_dimen, 1000, 100, env)
        self.target_net = helper.Net(self.state_dimen, self.action_dimen).to(device)
        self.estimate_net = helper.Net(self.state_dimen, self.action_dimen).to(device)

        self.env = env
        self.env.reset()
        self.timestamp = timestamp

        self.state_dimen = state_dimen
        self.action_dimen = action_dimen

        self.eval_env = copy.deepcopy(env)

        self.gamma = gamma

        self.learn_step_counter = 0

        self.batch_size = batch_size


        self.optimizer = torch.optim.Adam(self.estimate_net.parameters(), lr=lr)

    def action_select(self, state, epsilon=0.9): #epsilon greedy action selection

        state = torch.FloatTensor(state).to(device).reshape(-1)  # get a 1D array
        #epsilon greedy
        if np.random.rand() <= epsilon:
            action_val = self.estimate_net(state)
            action = torch.argmax(action_val).item()
        else:
            action = np.random.randint(0, self.action_dimen)
        return action

    def evaluate(self):

        state,_ = self.eval_env.reset()
        count = 0
        total_reward = 0
        done = False

        while not done:
            action = self.action_select(state, epsilon=1)
            next_state, reward, done, _, _ = self.eval_env.step(action)
            total_reward += reward
            count += 1
            state = next_state

        return total_reward, count

    def train(self, num_epochs):
        # init lists
        epoch_reward = 0
        loss_list = []
        avg_reward_list = []

        for epoch in tqdm(range(int(num_epochs))):
            done = False
            state,_ = self.env.reset()
            avg_loss = 0
            step = 0
            while not done:
                step += 1
                action = self.action_select(state)
                next_state, reward, done, _, _ = self.env.step(action)
                # experience
                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                }
                #add to buffer
                self.replay_buffer.buffer_add(exp)
                # set next state
                state = next_state

                # sample
                exp_batch = self.replay_buffer.buffer_sample(self.batch_size)

                #create batch variables and reshape
                state_batch = torch.FloatTensor([exp["state"] for exp in exp_batch]).to(device).reshape(self.batch_size, -1)
                # print(state_batch)
                action_batch = torch.LongTensor([exp["action"] for exp in exp_batch]).to(device).reshape(self.batch_size, -1)
                reward_batch = torch.FloatTensor([exp["reward"] for exp in exp_batch]).to(device).reshape(self.batch_size, -1)
                # print(reward_batch)
                next_state_batch = torch.FloatTensor([exp["next_state"] for exp in exp_batch]).to(device).reshape(self.batch_size, -1)
                done_batch = torch.FloatTensor([1 - exp["done"] for exp in exp_batch]).to(device).reshape(self.batch_size, -1)


                estimate_Q = self.estimate_net(state_batch).gather(1, action_batch)

                max_action_idx = self.estimate_net(next_state_batch).detach().argmax(1)
                # target
                target_Q = reward_batch + done_batch * self.gamma * self.target_net(next_state_batch).gather(1, max_action_idx.unsqueeze(1))

                loss = F.mse_loss(estimate_Q, target_Q)
                # print(loss)
                avg_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.learn_step_counter % 10 == 0: #target network
                    self.target_net.load_state_dict(self.estimate_net.state_dict())
                self.learn_step_counter += 1

            reward, count = self.evaluate()
            # cumulative reward
            epoch_reward += reward
            # print(reward)
            # print(epoch_reward)


            period = 50
            if epoch % period == 0:
                # log
                avg_loss /= step
                epoch_reward /= period
                avg_reward_list.append(epoch_reward)
                loss_list.append(avg_loss)

                print(
                    "\nepoch: [{}/{}], \t loss: {:f}, \t reward: {:f}, \tsteps: {}".format(
                        epoch + 1, num_epochs, avg_loss, epoch_reward, count
                    )
                )

                epoch_reward = 0
                try:
                    os.makedirs(self.timestamp)
                except OSError:
                    pass
                np.save(self.timestamp + "/double_dqn_loss.npy", loss_list)
                np.save(self.timestamp + "/double_dqn_avg_reward.npy", avg_reward_list)
                torch.save(
                    self.estimate_net.state_dict(), self.timestamp + "/double_dqn.pkl"
                )

        self.env.close()
        return loss_list, avg_reward_list


if __name__ == "__main__":

    # timestamp for saving
    name_time = time.localtime()  # get struct_time
    time_str = time.strftime(
        "%m%d_%H_%M", name_time
    )

    double_dqn_object = DDQN(
        env,
        state_dimen=105,
        action_dimen=3,
        lr=0.001,
        gamma=0.99,
        batch_size=64,
        timestamp=time_str,
    )

    avg_loss, avg_reward_list = double_dqn_object.train(2000) #train policy

    torch.save(double_dqn_object.estimate_net.state_dict(), time_str + "/double_dqn_network.pkl")
    np.save(time_str + "/double_dqn_loss.npy", avg_loss)
    np.save(time_str + "/double_dqn_avg_reward.npy", avg_reward_list)
