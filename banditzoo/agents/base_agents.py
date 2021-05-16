#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class of agents.

usage:
a = Agent()
...
a.observe(context)
actions = a.act()
...
a.update(reward,cost)
"""

import numpy as np


class Agent(object):
    """
    Base reinforcement learning agent class
    """

    def __init__(self, name=None, seed=0, **kwargs):
        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.i_t = None  # current action
        self.c_t = None  # current context
        self.t_t = 0  # current iteration

        self.reward = []  # keep track of rewards
        self.build(**kwargs)

    def build(self, **kwargs):
        pass

    def observe(self, c):
        self.c_t = c  # update context

    def update(self, rewards=None):
        self.update_agent(rewards)
        self.update_metrics(rewards)

    def act(self):
        raise NotImplementedError

    def update_agent(self, rewards=None):
        raise NotImplementedError

    def update_metrics(self, rewards=None):
        raise NotImplementedError

    def combine_rewards(self, rewards=None):
        return np.sum(rewards)
