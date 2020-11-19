from multiagent.core import Landmark
import numpy as np

from runner import Runner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args

from multiagent import scenarios
from multiagent.environment import MultiAgentEnv


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
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(**scenario_parameters)
    world.scenario_name = scenario_name

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, discrete_action_input=True,
                        discrete_action_space=True)

    observation_structures = calculate_observation_structure(scenario_name, env)
    env.observation_structures = observation_structures
    env.episode_limit = 25

    for a in env.agents:
        a.adversary = getattr(a, 'adversary', False)

    return env

if __name__ == '__main__':
    for i in range(1):
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
        # env = MyEnv(map_name=args.map,
        env = create_env('simple_spread')
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        print(env.get_env_info())
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
