import time
from argparse import Namespace
from itertools import chain
from ma_gym.envs.multiagent import MultiAgentBase, TranslatorMixin


from runner import Runner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import numpy as np


def calculate_observation_structure(scenario_name, environment):
    # p_pos + p_vel
    dim_agent = 4
    dim_p_pos = 2
    dim_p_vel = 2
    dim_landmark = 2
    dim_comm = 2
    dim_color = 3

    if scenario_name == 'simple_spread':
        # p_pos
        num_landmarks = len(
            [x for x in environment.world.entities if isinstance(x, Landmark)])
        dim_landmarks = dim_landmark * num_landmarks
        # excluding itself
        num_other_agents = len(environment.agents) - 1
        dim_other_agents = dim_p_pos * num_other_agents
        dim_other_comms = dim_comm * num_other_agents

        observation_structure = [0, dim_agent, dim_landmarks, dim_other_agents,
                                 dim_other_comms]
        observation_ranges = np.cumsum(observation_structure)

        return {False: observation_ranges}
    elif scenario_name == 'simple_tag':

        dim_entity_pos = sum(
            dim_p_pos for entity in environment.world.landmarks if
            not entity.boundary)
        num_agents = len(environment.agents)
        num_adv = sum(1 for agent in environment.agents if agent.adversary)
        num_non_adv = num_agents - num_adv

        non_adv_structure = [0, dim_agent, dim_entity_pos,
                             (num_agents - 1) * dim_p_pos,
                             (num_non_adv - 1) * dim_p_vel]
        non_adv_ranges = np.cumsum(non_adv_structure)
        adv_structure = [0, dim_agent, dim_entity_pos,
                         (num_agents - 1) * dim_p_pos, num_non_adv * dim_p_vel]
        adv_ranges = np.cumsum(adv_structure)

        return {False: non_adv_ranges, True: adv_ranges}
    elif scenario_name == 'simple_adversary':
        entity_pos = len(environment.world.landmarks) * dim_landmark
        other_pos = (len(environment.agents) - 1) * dim_p_pos

        non_adv_structure = [0, dim_p_pos, entity_pos, other_pos]
        adv_structure = [0, entity_pos, other_pos]

        non_adv_ranges = np.cumsum(non_adv_structure)
        adv_ranges = np.cumsum(adv_structure)

        return {False: non_adv_ranges, True: adv_ranges}
        pass
    elif scenario_name == 'simple_push':
        entity_pos = len(environment.world.landmarks) * dim_landmark
        other_pos = (len(environment.agents) - 1) * dim_p_pos
        entity_color = len(environment.world.landmarks) * dim_color

        non_adv_structure = [0, dim_p_vel, dim_p_pos, dim_color, entity_pos,
                             entity_color, other_pos]
        adv_structure = [0, dim_p_vel, entity_pos, other_pos]

        non_adv_ranges = np.cumsum(non_adv_structure)
        adv_ranges = np.cumsum(adv_structure)

        return {False: non_adv_ranges, True: adv_ranges}
    elif scenario_name == 'simple_spread_independent_reward':
        # p_pos
        num_landmarks = len(
            [x for x in environment.world.entities if
             isinstance(x, Landmark)])
        dim_landmarks = dim_landmark * num_landmarks
        # excluding itself
        num_other_agents = len(environment.agents) - 1
        dim_other_agents = dim_p_pos * num_other_agents
        dim_other_comms = dim_comm * num_other_agents

        observation_structure = [0, dim_agent, dim_landmarks,
                                 dim_other_agents,
                                 dim_other_comms]
        observation_ranges = np.cumsum(observation_structure)

        return {False: observation_ranges}
    else:
        raise NotImplementedError(
            f'This scenario is not supported: {scenario_name}')

def create_env(scenario_name, scenario_parameters={}):
    """
    Creates a multi-agent enviroment based on the scenario name and parameters
    :param scenario_name: string (e.g. 'simple_tag')
    :param scenario_parameters:  dictionary (e.g. {'num_agents': 2, 'num_adversaries': 1}
    :return: a MultiAgentEnv representing the scenario created with the specified parameters
    """
    # scenario = scenarios.load(scenario_name + ".py").Scenario()
    # world = scenario.make_world(**scenario_parameters)
    # env = MultiAgentEnv(world,
    #                     reset_callback=scenario.reset_world,
    #                     reward_callback=scenario.reward,
    #                     observation_callback=scenario.observation,
    #                     discrete_action_input=True,
    #                     discrete_action_space=True,
    #                     episode_limit=25)
    # env.observation_structures = calculate_observation_structure(scenario_name, env)
    # env.shared_reward = True
    #
    # for a in env.agents:
    #     a.adversary = getattr(a, 'adversary', False)
    #
    # return env

from common.replay_buffer import ReplayBuffer

class Curriculum:
    def __init__(self, train_envs, eval_env, target_env, train_env_duration=None, args=None):
        assert(len(train_envs)), "Nowhere to train"

        args.n_agents_max = target_env.n_agents

        # target_obs_structure = target_env.get_agent_sections(0)
        # target_state_structure = target_env.get_state(structure=True)

        for env in chain(train_envs, [eval_env]):
            translator = TranslatorMixin(from_env=env, to_env=target_env)
            env.translator = translator
            env.episode_limit = 25
            # env.translate_observation = target_obs_structure
            # env.translate_state = target_state_structure
            # make copy of namespace
            arg = Namespace(**dict(**vars(args)))
            env_info = env.get_env_info()
            arg.n_actions = env_info["n_actions"]
            arg.n_agents = env_info["n_agents"]
            arg.state_shape = env_info["state_shape"]
            arg.obs_shape = env_info["obs_shape"]
            arg.episode_limit = env._max_steps
            env.args = arg


        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env._max_steps

        for env in train_envs:
            env.buffer = ReplayBuffer(env.args)

        assert len(train_envs) == len(train_env_duration)
        self.train_envs = list(zip(train_envs, train_env_duration))
        self.eval_env = eval_env
        self.runner = None
        self._train = True
        self.next_env()

    def __getattr__(self, item):
        if item == 'reset':
            if self._train and self.episode_limit and self.env.num_episodes >= self.episode_limit:
                print(f'Old env @ {self.env.num_episodes} episodes w/ {self.env.args.n_agents} agents')
                self.next_env()
                self.env = self.train_env
                print(f'Now have {self.env.args.n_agents} agents')

        return getattr(getattr(self, 'env'), item)

    def next_env(self):
        self.train_env, self.episode_limit = self.train_envs.pop(0)

    def train(self):
        self.env = self.train_env
        self._train = True

    def eval(self):
        self.env = self.eval_env
        self._train = False


if __name__ == '__main__':
    import torch
    import gym
    import ma_gym

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
    timestamp = f'{int(time.time())}_test'

    for i in [0]:
        seed = 2**32-0-1
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        env = gym.make('PredatorPrey5x5-v0')


        train_envs = [
            env
        ]
        eval_env = gym.make('PredatorPrey5x5-v0')


        target_env = gym.make('PredatorPrey5x5-v0')

        train_env_duration = [None]

        env = Curriculum(train_envs, eval_env, target_env, args=args, train_env_duration=train_env_duration)

        runner = Runner(env, args, timestamp)
        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
