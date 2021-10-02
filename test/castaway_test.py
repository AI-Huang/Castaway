#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-01-21 05:37
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : http://example.org

import numpy as np
from Castaway.envs.castaway import Castaway


def main():
    env = Castaway(act_dim=50)
    print(env.action_space.n)
    print(
        f"env.observation_space.shape: {env.observation_space.shape}")

    env.reset()

    action = env.action_space.sample()
    phi = 0.5 * np.pi / (50-1) * action
    print(f"phi: {phi}")
    state, reward, done, _ = env.step(action)
    r_C, theta_C, phi, r_S, theta_S, omega_S = state
    print(f"r_C: {r_C}")
    print(reward)

    # env.render()  # TODO
    action = env.action_space.sample()
    phi = 0.5 * np.pi / (50-1) * action
    print(f"phi: {phi}")
    state, reward, done, _ = env.step(action)
    r_C, theta_C, phi, r_S, theta_S, omega_S = state
    print(f"r_C: {r_C}")
    print(reward)


if __name__ == "__main__":
    main()
