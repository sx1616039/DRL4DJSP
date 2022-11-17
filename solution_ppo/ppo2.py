import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from env_solution import JobEnv


class Actor(nn.Module):
    def __init__(self, num_input, num_output, node_num=100):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.action_head = nn.Linear(node_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_input, num_output=1, node_num=100):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.state_value = nn.Linear(node_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO:
    def __init__(self, j_env, unit_num=100, memory_size=5, batch_size=32, clip_ep=0.2):
        super(PPO, self).__init__()
        self.env = j_env
        self.memory_size = memory_size
        self.batch_size = batch_size  # update batch size
        self.epsilon = clip_ep

        self.state_dim = self.env.state_num
        self.action_dim = self.env.action_num
        self.case_name = self.env.case_name
        self.gamma = 1  # reward discount
        self.A_LR = 1e-3  # learning rate for actor
        self.C_LR = 3e-3  # learning rate for critic
        self.A_UPDATE_STEPS = 16  # actor update steps
        self.max_grad_norm = 0.5
        self.training_step = 0

        self.actor_net = Actor(self.state_dim, self.action_dim, node_num=1*unit_num)
        self.critic_net = Critic(self.state_dim, node_num=unit_num)
        self.actor_optimizer = optimizer.Adam(self.actor_net.parameters(), self.A_LR)
        self.critic_net_optimizer = optimizer.Adam(self.critic_net.parameters(), self.C_LR)
        if not os.path.exists('param'):
            os.makedirs('param/net_param')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_params(self):
        torch.save(self.actor_net.state_dict(), 'param/net_param/' + self.env.case_name + 'actor_net.model')
        torch.save(self.critic_net.state_dict(), 'param/net_param/' + self.env.case_name + 'critic_net.model')

    def load_params(self, model_name):
        self.critic_net.load_state_dict(torch.load('param/net_param/' + model_name + 'critic_net.model'))
        self.actor_net.load_state_dict(torch.load('param/net_param/' + model_name + 'actor_net.model'))

    def update(self, bs, ba, br, bp):
        # get old actor log prob
        old_action_log_prob = torch.tensor(bp, dtype=torch.float).view(-1, 1)
        state = torch.tensor(np.array(bs), dtype=torch.float)
        action = torch.tensor(ba, dtype=torch.long).view(-1, 1)
        d_reward = torch.tensor(br, dtype=torch.float)

        for i in range(self.A_UPDATE_STEPS):
            for index in BatchSampler(SubsetRandomSampler(range(len(ba))), self.batch_size, True):
                #  compute the advantage
                d_reward_index = d_reward[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = d_reward_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                ratio = (action_prob / old_action_log_prob[index])
                surrogate = ratio * advantage
                clip_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                action_loss = -torch.min(surrogate, clip_loss).mean()

                # update actor network
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(d_reward_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

    def test(self, model_name):
        self.load_params(model_name)
        t0 = time.time()
        converged_value = []
        for i in range(30):
            state = self.env.reset()
            while True:
                action, _ = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                if done:
                    break
            print(self.env.current_time)
            converged_value.append(self.env.current_time)
        return min(converged_value), 30, time.time() - t0, 0

    def train(self, model_name, is_reschedule=False):
        if is_reschedule:
            self.load_params(model_name)
        column = ["episode", "make_span", "reward", "no-op"]
        results = pd.DataFrame(columns=column, dtype=float)
        index = 0
        converged = 0
        converged_value = []
        t0 = time.time()
        for i_epoch in range(4000):
            if time.time()-t0 >= 3600:
                break
            bs, ba, br, bp = [], [], [], []
            for m in range(self.memory_size):  # memory size is the number of complete episode
                buffer_s, buffer_a, buffer_r, buffer_p = [], [], [], []
                state = self.env.reset()
                episode_reward = 0
                while True:
                    action, action_prob = self.select_action(state)
                    next_state, reward, done = self.env.step(action)
                    buffer_s.append(state)
                    buffer_a.append(action)
                    buffer_r.append(reward)
                    buffer_p.append(action_prob)

                    state = next_state
                    episode_reward += reward
                    if done:
                        v_s_ = 0
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + self.gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs[len(bs):len(bs)] = buffer_s
                        ba[len(ba):len(ba)] = buffer_a
                        br[len(br):len(br)] = discounted_r
                        bp[len(bp):len(bp)] = buffer_p
                        # Episode: make_span: Episode reward
                        print('{}    {}    {:.2f}  {}'.format(i_epoch, self.env.current_time, episode_reward,
                                                              self.env.no_op_cnt))
                        index = i_epoch * self.memory_size + m
                        results.loc[index] = [i_epoch, self.env.current_time, episode_reward, self.env.no_op_cnt]
                        converged_value.append(self.env.current_time)
                        if len(converged_value) >= 31:
                            converged_value.pop(0)
                        break
            self.update(bs, ba, br, bp)
            converged = index
            if min(converged_value) == max(converged_value) and len(converged_value) >= 30:
                converged = index
                break
        if not os.path.exists('results'):
            os.makedirs('results')
        results.to_csv("results/" + str(self.env.case_name) + "_" + model_name + ".csv")
        self.save_params()
        return min(converged_value), converged, time.time()-t0, self.env.no_op_cnt


if __name__ == '__main__':
    data_set_name = "16-solution-9202"
    path = "../data_set_sizes/"
    parameters = data_set_name
    param = [parameters, "converge_cnt", "total_time", "no-op"]

    simple_results = pd.DataFrame(columns=param, dtype=int)
    for file_name in os.listdir(path):
        print(file_name + "========================")
        title = file_name.split('.')[0]
        name = file_name.split('_')[0]
        env = JobEnv(title, path, no_op=False)
        scale = env.job_num * env.machine_num
        model = PPO(env, unit_num=env.state_num, memory_size=9, batch_size=2 * scale, clip_ep=0.2)
        simple_results.loc[title] = model.train(name, is_reschedule=False)
    simple_results.to_csv(parameters + ".csv")
    # test cases
    # for file_name in os.listdir(path):
    #     print(file_name + "========================")
    #     title = file_name.split('.')[0]
    #     name = file_name.split('_')[0]
    #     env = JobEnv(title, path, no_op=False)
    #     scale = env.job_num * env.machine_num
    #     model = PPO(env, memory_size=7, batch_size=2 * scale, clip_ep=0.25)
    #     simple_results.loc[title] = model.test(name)
    # simple_results.to_csv(parameters + "_result.csv")
