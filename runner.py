import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import time


class Runner:
    def __init__(self, env, args, timestamp):
        # TODO: refactor references to args to be to self.env.args, that way we can
        #  change just env.args and the changes will propagate nicely.
        self.timestamp = timestamp
        self.env = env
        self.env.runner = self

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(env, args, self.timestamp)
            self.rolloutWorker = RolloutWorker(env, self.agents, args) # TODO: when change number of agents, must create new agents.
        # if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
        #     self.buffer = ReplayBuffer(args)

        self.args = args
        self.win_rates = []
        self.eval_rewards = []
        self.train_rewards = []
        self.train_win_rates = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        # print('Run {} start'.format(num))
        cumulative_train_reward = 0
        train_win_number = 0
        for epoch in range(self.args.n_epoch):
            if epoch % self.args.evaluate_cycle == 0:
                self.train_rewards.append(cumulative_train_reward / self.args.evaluate_cycle)
                self.train_win_rates.append(train_win_number / self.args.evaluate_cycle)
                cumulative_train_reward = 0
                train_win_number = 0

                print('{} Run {:4} eval epoch  {:12}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), num, epoch))

                win_rate, eval_episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.eval_rewards.append(eval_episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            if epoch % self.args.evaluate_cycle == 0:
                print('{} Run {:4} train epoch {:12}'.format( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), num, epoch))

            for episode_idx in range(self.args.n_episodes):
                episode, ep_reward, win_tag = self.rolloutWorker.generate_episode(episode_idx)
                cumulative_train_reward += ep_reward
                train_win_number += int(win_tag)
                episodes.append(episode)


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
                # self.buffer.store_episode(episode_batch)
                self.env.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    # mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    mini_batch = self.env.buffer.sample(min(self.env.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        self.plt(num)

    def evaluate(self, render=False):
        win_number = 0
        episode_rewards = 0
        self.env.eval()
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True, render=render)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        self.env.train()
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch


    def plt(self, num):
        # num = 'test'
        fig = plt.figure()
        fig.set_size_inches(15, 10)
        plt.cla()

        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.subplot(4, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.ylabel('eval win_rate')

        plt.subplot(4, 1, 2)
        plt.plot(range(len(self.train_win_rates)), self.train_win_rates)
        plt.ylabel('train win_rate')

        plt.subplot(4, 1, 3)
        plt.plot(range(len(self.eval_rewards)), self.eval_rewards)
        plt.ylabel('eval cumul. R')


        plt.subplot(4, 1, 4)
        plt.plot(range(len(self.train_rewards)), self.train_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('train cumul. R')

        plt.tight_layout()

        if not os.path.isdir(f'{self.save_path}/{self.timestamp}'):
            os.mkdir(f'{self.save_path}/{self.timestamp}')

        plt.savefig(self.save_path + '/{}/{}.png'.format(self.timestamp, num), format='png')
        np.save(self.save_path + '/{}/win_rates_{}'.format(self.timestamp, num), self.win_rates)
        np.save(self.save_path + '/{}/eval_rewards_{}'.format(self.timestamp, num), self.eval_rewards)
        np.save(self.save_path + '/{}/train_rewards_{}'.format(self.timestamp, num), self.train_rewards)
        plt.close(fig)









