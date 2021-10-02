#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-01-21 00:52
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import copy
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text


def norm_angle(phases):
    """norm_angle
    """
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


class Castaway(gym.Env):
    """
    Observation:
        Type: MultiAgentObservationSpace Box(4)
        For the Castaway (observation_space[0])
        Num     Observation                      Min                    Max
        0       Castaway Position r_C            0                      1
        1       Castaway Position theta_C        0 rad                  2*np.pi
        2       Castaway Velocity Direction \phi 0 rad     2*np.pi rad (180 deg)

        Num     Observation                      Min                     Max
        0       Shark Position r_S               1                       1
        1       \Delta\theta           -np.pi                   np.pi
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

    def __init__(self, act_dim, s=2.0, speed_ratio=4.0, tau=0.02):
        self.act_dim = act_dim
        self.s = s
        self.speed_ratio = speed_ratio  # the shart and the castaway's speed ratio
        self.tau = tau  # seconds between state updates

        self.action_space = spaces.Discrete(
            act_dim)  # 0, 1, 2..., N-1

        self.box_castaway = spaces.Box(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2*np.pi, 0.5*np.pi]))
        self.box_shark = spaces.Box(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 2*np.pi, 0.5*np.pi]))
        self.observation_space = self.box_castaway

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

    def step(self, action):
        """step function based on dynamics
        action: actions all the agents in the multi-agent system
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        (r_C, theta_C, phi, r_S, theta_S, omega_S) = self.state
        # print(r_C, theta_C, phi, r_S, theta_S, omega_S)

        # phi and omega_S are both state variables and action variables
        phi = 0.5 * np.pi / (self.act_dim-1) * action

        # The shark's strategy
        omega_S_abs = self.speed_ratio*self.s  # 4s rad/s
        delta_theta = theta_C - theta_S
        omega_S = np.sign(delta_theta) * omega_S_abs  # +4s rad/s or -4s rad/s

        # Dynamics
        r_C_0 = r_C
        r_C += self.s * np.cos(phi) * self.tau
        theta_C += self.s * np.cos(phi) / r_C_0 * self.tau
        r_S = 1
        theta_S += omega_S * self.tau

        self.state = (r_C, theta_C, phi, r_S, theta_S, omega_S)

        done = False
        # Escaped
        if r_C >= 1.0 and delta_theta != 0:
            done = True
        done = bool(done)

        # linear reward
        reward = (1-r_C) + np.abs(delta_theta)

        return self.state, reward, done, {}

    def _draw(self, args):
        from PIL import Image, ImageDraw
        width, height = 200, 100
        self._base_img = Image.new(
            mode='RGB', size=(width, height), color='black')
        self._base_img = draw_circle(self._base_img,
                                     (50, 50), cell_size=50, fill='white')

    def reset(self):
        # The Castaway
        r_C = self.np_random.uniform(low=0.0, high=1.0)
        theta_C = self.np_random.uniform(low=-np.pi, high=np.pi)
        phi = self.np_random.uniform(low=0.0, high=2*np.pi)

        # The Shark
        r_S = 1.0
        theta_S = self.np_random.uniform(low=-np.pi, high=np.pi)
        omega_S = self.np_random.uniform(low=0.0, high=1)
        if omega_S < 0.5:  # 0.5 the probability
            omega_S = self.speed_ratio*self.s  # e.g. 4s rad/s
        else:
            omega_S = -self.speed_ratio*self.s

        self.state = (r_C, theta_C, phi, r_S, theta_S, omega_S)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def main():
    pass


if __name__ == "__main__":
    main()
