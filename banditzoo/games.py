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
from collections import defaultdict

class Game(object):
    """
    Base class of Game object
    """

    def __init__(self, name=None, seed=0, N=None, M=None):
        """Generate a Game object.

        Args:
            name (str, optional): [name of the game]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            N (int, optional): [number of world instances per world class]. Defaults to None.
            M (int, optional): [number of agent instances per agent class]. Defaults to None.
        """

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.N = N 
        self.M = M 

        self.world_names = []
        self.agent_names = []
        self.world_pools = {}
        self.agent_pools = {}
        self.history_pools = {}
        self.metrics_pools = {}

        self.agent_add_lock = False  
        self.world_add_lock = False  

    def add_world_class(self, world_class, **kwargs):
        """Add world class into the game.

        Args:
            world_class ([banditzoo.worlds object]): [the world class].
            **kwargs ([any], optional): [the args to initialize the worlds].

        Raises:
            Exception: [if the game has started, no new world can enter].
            Exception: [if the agents are in, no new world can enter].
        """
        
        if self.world_add_lock and self.agent_add_lock:
            raise Exception("No worlds can enter anymore because the game is run.")
        if self.world_add_lock and not self.agent_add_lock:
            raise Exception("No worlds can enter anymore because the agents are in.")
        
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
        """Add world class into the game.

        Args:
            world_class (banditzoo.worlds object): [the world class].
            **kwargs (any, optional): [the args to initialize the worlds].

        Raises:
            Exception: [if the game has started, no new agent can enter].
            Exception: [if the game has no worlds, no agent can enter].
        """
        
        if self.agent_add_lock:
            raise Exception("No agents can enter anymore because the game is run.")
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
                self.history_pools[k][i][agent_name] = []
                self.metrics_pools[k][i][agent_name] = []

    def aggregate_world_metrics(self):
        """Aggregate the metrics in the M dimension (the agent instances).
        """
        agg_metrics = {}
        for k in self.world_names:
            agg_metrics[k] = {}
            metrics_keys = list(self.world_pools[k][0].metrics.keys())
            for i in range(self.N):
                agg_metrics[k][i] = {}
                for a in self.agent_names:
                    metrics = self.metrics_pools[k][i][a]
                    agg_metrics[k][i][a] = defaultdict(lambda : [])
                    for m in metrics:
                        for m_key, m_val in m.items():
                            agg_metrics[k][i][a][m_key].append(m_val)
                    for mk in metrics_keys:
                        agg_metrics[k][i][a][mk+'_avg'] = np.mean(agg_metrics[k][i][a][mk], axis=0)
                        agg_metrics[k][i][a][mk+'_std'] = np.std(agg_metrics[k][i][a][mk], axis=0)
                        agg_metrics[k][i][a][mk+'_sem'] = np.std(agg_metrics[k][i][a][mk], axis=0, ddof=1) / np.sqrt(self.M)
        return agg_metrics            
                    
    def aggregate_agent_metrics(self):
        """Aggregate the metrics in both M and N dimensions (world and agent instances).
        """
        world_agg_metrics = self.aggregate_world_metrics()
        agg_metrics = {}
        for k in self.world_names:
            agg_metrics[k] = defaultdict(lambda : {})
            metrics_keys = list(self.world_pools[k][0].metrics.keys())
            for i in range(self.N):
                for a in self.agent_names:
                    agg_metrics[k][a] = defaultdict(lambda : [])
                    agg_metrics[k][a]['name'] = a
                    for mk in metrics_keys:
                        for m in world_agg_metrics[k][i][a][mk]:
                            agg_metrics[k][a][mk].append(m)
            for a in self.agent_names:
                for mk in metrics_keys:
                    agg_metrics[k][a][mk+'_avg'] = np.mean(agg_metrics[k][a][mk], axis=0)
                    agg_metrics[k][a][mk+'_std'] = np.std(agg_metrics[k][a][mk], axis=0)
                    agg_metrics[k][a][mk+'_sem'] = np.std(agg_metrics[k][a][mk], axis=0, ddof=1) / np.sqrt(self.M*self.N)                         
        return agg_metrics            

    def run_experiments(self, T, progress=False):
        """Run the game with certain iterations.

        Args:
            T (int): [number of time steps for the game in this run].
            progress (bool, optional): [whether to print progress]. Defaults to False.
        """
        for k in self.world_names:
            for i, w in enumerate(self.world_pools[k]):
                if progress:
                    print("==============================================")
                    print(
                        "Now running the world " + k + " " + str(i) + "/" + str(self.N)
                    )
                for a in self.agent_names:
                    w.add_agent_pool(self.agent_pools[k][i][a])
                w.run_experiments(T=T, progress=progress)
                for a in self.agent_names:
                    self.agent_pools[k][i][a] = w.filter_agent(a)
                    self.history_pools[k][i][a] = w.filter_history(a)
                    self.metrics_pools[k][i][a] = w.filter_metrics(a)
        self.agent_add_lock = True
        self.world_add_lock = True

    def get_full_data(self):
        """Output the full historical data of the game.

        Returns:
            [tuple]: [a tuple with the world instances, agent instances, histories and metrics].
        """
        return (
            self.world_pools,
            self.agent_pools,
            self.history_pools,
            self.metrics_pools,
        )

    def get_metrics(self, group_by='agent'):
        """Output the metrics of the agents in the worlds.

        Args:
            group_by (str, optional): [output format of the metrics].
                If 'agent', the metrics are aggregated by both N and M dimension.
                If 'world', the metrics are aggregated only in the M dimension (the 
                agent instances) and not the world instances. Defaults to 'agent'.

        Returns:
            [dict]: [the aggregated metrics of the agents].
            
        Raises:
            ValueError: [if the game has started, no new agent can enter].
        """
        if group_by == 'agent':
            return self.aggregate_agent_metrics()
        elif group_by == 'world':
            return self.aggregate_world_metrics()
        else:
            raise ValueError("Please select a supported grouping tag ('agent', 'world')")
