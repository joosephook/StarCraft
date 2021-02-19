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

        self.metrics = {
            'eval': dict(win_tag=[], ep_reward=[]),
            'train': dict(win_tag=[], ep_reward=[]),
        }

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0

        for epoch in range(1, self.args.n_epoch+1):
            episodes = []

            for episode_idx in range(self.args.n_episodes):
                episode, ep_reward, win_tag = self.rolloutWorker.generate_episode(episode_idx)
                self.metrics['train']['ep_reward'].append(ep_reward)
                self.metrics['train']['win_tag'].append(win_tag)
                episodes.append(episode)

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.env.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.env.buffer.sample(min(self.env.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

            # evaluate
            if epoch % self.args.evaluate_cycle == 0:
                # print('{} Run {:4} eval epoch  {:12}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), num, epoch))
                self.evaluate()
                self.plt(num)

        self.plt(num)

    def evaluate(self, render=False):
        self.env.eval()
        for epoch in range(self.args.evaluate_epoch):
            _, ep_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True, render=render)
            self.metrics['eval']['ep_reward'].append(ep_reward)
            self.metrics['eval']['win_tag'].append(win_tag)
        self.env.train()


    def plt(self, num):
        # num = 'test'
        labels = []
        datas = []
        n_plots = 0


        for phase, metrics in self.metrics.items():
            for metric, data in metrics.items():
                labels.append(f'{phase} {metric}')
                datas.append(np.array(data))
                n_plots += 1

        fig, axes = plt.subplots(n_plots, 1)
        fig.set_size_inches(15, 10)

        # self.args.evaluate_cycle
        for ax, data, label in zip(axes, datas, labels):
            d = np.sort(data.reshape(self.args.evaluate_cycle, -1), axis=0)
            img = ax.imshow(d)
            plt.colorbar(img, ax=ax)
            ax.set_ylabel(label)

        plt.tight_layout()

        if not os.path.isdir(f'{self.save_path}/{self.timestamp}'):
            os.mkdir(f'{self.save_path}/{self.timestamp}')

        plt.savefig(self.save_path + '/{}/plot.png'.format(self.timestamp), format='png')

        for phase, metrics in self.metrics.items():
            for metric, data in metrics.items():
                np.save(f'{self.save_path}/{self.timestamp}/{phase}_{metric}.npy', np.array(data))
                n_plots += 1

        plt.close(fig)
        plt.cla()









