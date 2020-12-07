import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

import numpy as np

EPOCH=0

def reset_callback(world):
    anneal_over = 10_000
    if EPOCH <= 10_000:
        r = 0.1
    if EPOCH <= 15_000:
        r = 0.5
    else:
        r = 1.0

    r = 1.0
    # random properties for agents
    for i, agent in enumerate(world.agents):
        agent.color = np.array([0.35, 0.35, 0.85])
    # random properties for landmarks
    for i, landmark in enumerate(world.landmarks):
        landmark.color = np.array([0.25, 0.25, 0.25])
    # set random initial states
    for agent in world.agents:
        agent.state.p_pos = np.random.uniform(-r, +r,
                                              world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

    for i, landmark in enumerate(world.landmarks):
        # landmark.state.p_pos = world.agents[i].state.p_pos + np.random.uniform(-r, +r,
        #                                                            world.dim_p)
        landmark.state.p_pos = np.random.uniform(-r, +r, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.fig = None

    def run(self, num):
        global EPOCH
        train_steps = 0
        # print('Run {} start'.format(num))
        self.env.reset_callback = reset_callback #TODO
        for epoch in range(self.args.n_epoch):
            EPOCH = epoch
            # print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0:
                # print('Run {}, train epoch {}, evaluating'.format(num, epoch))
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(self.rolloutWorker.epsilon)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        if self.fig is None:
            self.fig = plt.figure()

        fig = self.fig
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('epsilon')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')
        plt.tight_layout()

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.clf()
        # plt.close(fig)









