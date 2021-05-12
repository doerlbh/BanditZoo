#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes of Games, i.e. experiments to run agents in worlds

usage:
g = Game(N=5,M=10)
...
g.add_world_class(worlds.BernoulliMultiArmedBandits, M=3)
g.add_agent_class(agents.TS, M=3)
g.run_experiments(T=1000)
...
results = g.get_metrics()
"""

import numpy as np


class Game(object):
    """
    Base class of Game object
    """

    def __init__(self, name=None, seed=0, N=None, M=None, T=None):

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.N = N  # number of world instances per world class
        self.M = M  # number of agent instances per agent class
        self.T = T  # number of time steps to run

        self.world_names = []
        self.agent_names = []
        self.world_pools = {}
        self.agent_pools = {}
        self.history_pools = {}
        self.metrics_pools = {}

        self.run_lock = False  # make sure run_experiments only be run once

    def add_world_class(self, world_class, **kwargs):
        world_name = kwargs.get("name", None)
        self.world_names.append(world_name)
        world_instances = []
        self.agent_pools[world_name] = {}
        self.history_pools[world_name] = {}
        self.metrics_pools[world_name] = {}
        for i in range(self.N):
            world_instances.append(world_class(**kwargs, seed=i))
            self.agent_pools[world_name][i] = {}
            self.history_pools[world_name][i] = {}
            self.metrics_pools[world_name][i] = {}
        self.world_pools[world_name] = world_instances

    def add_agent_class(self, agent_class, **kwargs):
        if len(self.world_pools) == 0:
            raise Exception("Please initiate all the worlds before adding agents.")
        agent_name = kwargs.get("name", None)
        self.agent_names.append(agent_name)
        agent_instances = []
        for i in range(self.M):
            agent_instances.append(agent_class(**kwargs, seed=i))
        for k in self.world_names:
            for i in range(self.N):
                self.agent_pools[k][i][agent_name] = agent_instances
                self.history_pools[k][i][agent_name] = {}
                self.metrics_pools[k][i][agent_name] = {}

    def sync_agent(self, w, a):
        raise NotImplementedError

    def sync_history(self, w, a):
        raise NotImplementedError

    def sync_metrics(self, w, a):
        raise NotImplementedError

    def aggregate_world_metrics(self):
        raise NotImplementedError

    def aggregate_agent_metrics(self):
        raise NotImplementedError

    def run_experiments(self, progress=False):
        if self.run_lock:
            raise Exception(
                "Please reinitiate the game session before running the experiment."
            )
        for k in self.world_names:
            for i, w in enumerate(self.world_pools[k]):
                if progress:
                    print("==============================================")
                    print(
                        "Now running the world " + k + " " + str(i) + "/" + str(self.N)
                    )
                for a in self.agent_names:
                    w.add_agent_pool(self.agent_pools[k][i][a])
                w.run_experiments(T=self.T, progress=progress)
                for a in self.agent_names:
                    self.agent_pools[k][i][a] = self.sync_agent(w, a)
                    self.history_pools[k][i][a] = self.sync_history(w, a)
                    self.metrics_pools[k][i][a] = self.sync_metrics(w, a)
        self.run_lock = True

    def get_full_data(self):
        return (
            self.world_pools,
            self.agent_pools,
            self.history_pools,
            self.metrics_pools,
        )

    def get_metrics(self):
        return self.aggregate_world_metrics(), self.aggregate_agent_metrics()
