#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-01-21 00:52
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.action_space import MultiAgentActionSpace

# from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text


class CastawayAndShark(gym.Env):
    """
    Observation:
        Type: MultiAgentObservationSpace Box(4)
        For the Castaway (observation_space[0])
        Num     Observation                      Min                    Max
        0       Castaway Position r_C            0                      1
        1       Castaway Position theta_C        0 rad                  2*np.pi
        2       Castaway Velocity Direction \phi 0 rad     2*np.pi rad (180 deg)

        For the Shark (observation_space[1])
        Num     Observation                      Min                     Max
        0       Shark Position r_S               1                       1
        1       Shark Position theta_S           0 rad                   2*np.pi
        2       Shark Angular Velocity \omega_S    -4s rad/s or 4s rad/s
    Hparam:
        act_dim: how many discrete points the \phi space is devided into.
        s: m/s, speed unit the Castaway (s) and the Shark (4s) default 2.0.
        coefficient: default 4.0, meaning the shark's speed is 4 times as large as the castaway.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, act_dim, s=2.0, coefficient=4.0, tau=0.02):
        self.act_dim = act_dim
        self.s = s
        self.coefficient = coefficient
        self.tau = tau  # seconds between state updates

        self.n_agents = 2
        self.action_space_castaway = spaces.Discrete(
            act_dim)  # 0, 1, 2..., N-1
        self.action_space_shark = spaces.Discrete(2)  # 0, 1
        self.action_space = MultiAgentActionSpace(
            [self.action_space_castaway, self.action_space_shark])

        self.box_castaway = spaces.Box(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2*np.pi, 0.5*np.pi]))
        self.box_shark = spaces.Box(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 2*np.pi, 0.5*np.pi]))
        self.observation_space = MultiAgentObservationSpace(
            [self.box_castaway, self.box_shark])

        # Initial rewards of the Castaway and the Shark
        self.agent_reward = {0: {'lemon': -10, 'apple': 10},
                             1: {'lemon': -1, 'apple': 1}}

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, agents_action):
        """step function based on dynamics
        agents_action: actions all the agents in the multi-agent system
        """
        for i, action in enumerate(agents_action):
            err_msg = "%r (%s) invalid" % (action, type(action))
            assert self.action_space[i].contains(action), err_msg

        (r_C, theta_C, phi, r_S, theta_S, omega_S) = self.state
        # print(r_C, theta_C, phi, r_S, theta_S, omega_S)

        # phi and omega_S are both state variables and action variables
        action_castaway = agents_action[0]
        action_shark = agents_action[1]
        phi = 0.5 * np.pi / (self.act_dim-1) * action_castaway
        if action_shark == 0:  # 0
            omega_S = self.coefficient*self.s  # e.g. 4s rad/s
        else:  # 1
            omega_S = -self.coefficient*self.s

        # Dynamics
        r_C_0 = r_C
        r_C += self.s * np.cos(phi) * self.tau
        theta_C += self.s * np.cos(phi) / r_C_0 * self.tau
        r_S = 1
        theta_S += omega_S * self.tau

        self.state = (r_C, theta_C, phi, r_S, theta_S, omega_S)

        # TODO
        done = False
        done = bool(done)

        # TODO reward
        if not done:
            reward = -1.0
        else:
            reward = 1.0

        return self.state, reward, done, {}

    def reset(self):
        # The Castaway
        r_C = self.np_random.uniform(low=0.0, high=2*np.pi)
        theta_C = self.np_random.uniform(low=0.0, high=2*np.pi)
        phi = self.np_random.uniform(low=0.0, high=2*np.pi)

        # The Shark
        r_S = 1.0
        theta_S = self.np_random.uniform(low=0.0, high=2*np.pi)
        omega_S = self.np_random.uniform(low=0.0, high=1)
        if omega_S < 0.5:  # 0.5 the probability
            omega_S = self.coefficient*self.s  # e.g. 4s rad/s
        else:
            omega_S = -self.coefficient*self.s

        self.state = (r_C, theta_C, phi, r_S, theta_S, omega_S)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        return None

    def close(self):
        return None


def main():
    env = CastawayAndShark(act_dim=50)
    env.reset()
    env.step(env.action_space.sample())
    print(env.state)
    env.step(env.action_space.sample())
    print(env.state)


if __name__ == "__main__":
    main()
