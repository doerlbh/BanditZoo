#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class of worlds.

usage:
w = World()
...
w.add_agent(agent)
w.run_experiments(T=1000)
...
results = w.get_results()
"""

import numpy as np

from .utils import print_progress


class World(object):
    """
    Base class of world object
    """

    def __init__(self, name=None, seed=0):
        """Initialize the base world object.

        Args:
            name (str, optional): [world name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
        """

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.agents = []
        self.history = {}
        self.metrics = {}

    def add_agent(self, agent):
        """Enter agent into the world.

        Args:
            agent (banditzoo.agents class object): [an agent instance].
        """
        self.history[len(self.agents)] = []
        self.metrics[len(self.agents)] = self.init_metrics()
        agent.build(**self.get_env_config())
        self.agents.append(agent)

    def add_agent_pool(self, agents):
        """Enter a pool of agents into the world.

        Args:
            agents (list of banditzoo.agents objects): [a list of agent instances]
        """
        for a in agents:
            self.add_agent(a)

    def filter_agent(self, agent_name, get_index=False):
        """Filter agent given an agent name.

        Args:
            agent_name (str): [the agent name].
            get_index (bool, optional): [whether to return the agent indices or instances]. Defaults to False.

        Returns:
            [list of banditzoo.agents objects]: [a list of agent instances or indices that matches the agent name].
        """
        if get_index:
            filtered = [a.name == agent_name for a in self.agents]
            return np.arange(len(self.agents))[filtered]
        else:
            return [a for a in self.agents if a.name == agent_name]

    def filter_history(self, agent_name):
        return [self.history[a] for a in self.filter_agent(agent_name, get_index=True)]

    def filter_metrics(self, agent_name):
        return [self.metrics[a] for a in self.filter_agent(agent_name, get_index=True)]

    def run_experiments(self, T, progress=False):
        for t in range(T):
            context = self.provide_context(t)
            for i in range(len(self.agents)):
                self.agents[i].observe(context)
                a = self.agents[i].act()
                self.history[i].append(a)
                r = self.assign_reward(a)
                self.agents[i].update(r)
                self.metrics[i] = self.update_metrics(
                    self.metrics[i], r, self.agents[i]
                )
            if progress:
                print_progress(t, T)

    def get_results(self):
        return self.agents, self.history, self.metrics

    def provide_context(self, t):
        raise NotImplementedError

    def assign_reward(self, action):
        raise NotImplementedError

    def init_metrics(self):
        raise NotImplementedError

    def update_metrics(self, metrics, reward, agent):
        raise NotImplementedError

    def get_env_config(self):
        raise NotImplementedError
