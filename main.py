import time
from argparse import Namespace
from itertools import chain
from typing import List

from ma_gym.envs.multiagent import MultiAgentBase, TranslatorMixin

import shutil
import os
import torch
import gym

from runner import Runner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import numpy as np

from common.replay_buffer import ReplayBuffer

class Curriculum:
    def __init__(self, train_envs: List[MultiAgentBase], eval_env: MultiAgentBase, target_env: MultiAgentBase, train_env_duration=None, args=None):
        assert(len(train_envs)), "Nowhere to train"
        assert args.n_epoch % args.evaluate_cycle == 0, "Make sure the number of episodes trained is divisible by evaluate cycle"
        assert target_env.n_agents == max(env.n_agents for env in chain(train_envs, [eval_env, target_env]))
        assert len(train_envs) == len(train_env_duration), "Each training env needs a corresponding train duration"

        args.n_agents_max = target_env.n_agents
        args.n_epoch = sum(train_env_duration)

        for env in chain(train_envs, [eval_env]):
            translator = TranslatorMixin(env.observations[0].sections,
                                         target_env.observations[0].sections,
                                         env.state.sections,
                                         target_env.state.sections)
            env.translator = translator

            arg = Namespace(**dict(**vars(args)))
            env.episode_limit = env._max_steps
            env_info = env.get_env_info()
            arg.n_actions = env_info["n_actions"]
            arg.n_agents = env_info["n_agents"]
            arg.state_shape = env_info["state_shape"]
            arg.obs_shape = env_info["obs_shape"]
            arg.episode_limit = env._max_steps
            env.args = arg

        for env in train_envs:
            env.args.n_actions = eval_env.get_env_info()['n_actions']

        env_info = eval_env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env._max_steps

        for env in train_envs:
            env.buffer = ReplayBuffer(env.args)

        self.train_envs = list(zip(train_envs, train_env_duration))
        self.eval_env = eval_env
        self.runner = None
        self._train = True

        self.episode_limit = None
        self.env = None
        self.next_env()
        self.train()

    def __getattr__(self, item):
        if item == 'reset':
            if self._train and self.episode_limit is not None and self.env.episode_count >= self.episode_limit:
                print(f'Old env @ {self.env.episode_count} episodes w/ {self.env.args.n_agents} agents')
                print(self.runner.rolloutWorker.epsilon)
                self.env = None
                del self.train_env

                self.next_env()
                self.env = self.train_env
                # self.runner.rolloutWorker.epsilon = 1.0
                print(f'Now have {self.env.args.n_agents} agents')
                # TODO: optimizer reset?
                # print(f'Resetting optimiser...')
                # self.runner.agents.policy.reset_optimiser()

        return getattr(self.env, item)
        # return getattr(getattr(self, 'env'), item)

    def next_env(self):
        self.train_env, self.episode_limit = self.train_envs.pop(0)
        self.train_env.buffer.create(self.train_env.buffer.args)

    def train(self):
        self.env = self.train_env
        self._train = True

    def eval(self):
        self.env = self.eval_env
        self._train = False


if __name__ == '__main__':


    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # 12x12 10A5P
    # 100x100 40A20P
    for i in range(5):
        seed = 2 ** 32-i-1

        np.random.seed(seed)
        torch.random.manual_seed(seed)

        eval_seed = 100+i

        timestamp = f'{int(time.time())}_combat_{i}_noreset_epsilon_seed_{seed}_eval_seed_{eval_seed}'

        train_env_duration = [
            20_000,
        ]

        eval_env =   gym.make('Combat-v0', seed=eval_seed)
        target_env = gym.make('Combat-v0')

        env = Curriculum(
            [
                gym.make('Combat-v0', n_agents=5, n_opponents=5)
            ]
        , eval_env, target_env, args=args, train_env_duration=train_env_duration)

        runner = Runner(env, args, timestamp)

        shutil.copy(os.path.basename(__file__), runner.save_path+f'/{os.path.basename(__file__)}')
        shutil.copy('common/arguments.py', runner.save_path+'/arguments.py')


        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
