#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes of Worlds, i.e. synthetic real-world bandit or RL environments

usage:
w = World()
...
w.add_agent(agent)
w.run_experiments(T=1000)
...
results = w.get_results()
"""

import numpy as np


class World(object):
    """
    Base class of world object
    """

    def __init__(self, name=None, seed=0):

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.agents = []
        self.history = {}
        self.metrics = {}

    def add_agent(self, agent):
        self.history[len(self.agents)] = []
        self.metrics[len(self.agents)] = self.init_metrics()
        self.agents.append(agent)
    
    def provide_context(self, t):
        raise NotImplementedError

    def assign_reward(self, action):
        raise NotImplementedError

    def init_metrics(self):
        raise NotImplementedError
    
    def update_metrics(self, metrics, reward, agent):
        raise NotImplementedError

    def run_experiments(self, T=1000):
        for t in range(T):
            context = self.provide_context(t)
            for i in range(len(self.agents)):
                self.agents[i].observe(context)
                a = self.agents[i].act()
                self.history[i].append(a)
                r = self.assign_reward(a)
                self.agents[i].update(r)
                self.metrics[i] = self.update_metrics(self.metrics[i], r, self.agents[i])

    def get_results(self):
        return self.agents, self.history, self.metrics


class BernoulliMultiArmedBandits(World):
    """
    Multi-Armed Bandits Problem with Bernoulli rewards
    """

    def __init__(
        self,
        M=5,
        name=None,
        seed=0,
        reward_means=None,
        cost_means=None,
        use_cost=False,
        reward_scale=1,
    ):
        super().__init__(name=name, seed=seed)

        self.M = M  # number of arms
        self.use_cost = use_cost
        self.reward_functions = None
        self.cost_functions = None
        if reward_means is None:
            self.reward_means = np.random.uniform(0, reward_scale, self.M)
        else:
            self.reward_means = reward_means
        if self.use_cost:
            if cost_means is None:
                self.cost_means = np.random.uniform(0, reward_scale, self.M)
            else:
                self.cost_means = cost_means
       
    def provide_context(self, t):
        pass
    
    def assign_reward(self, action):
        self.reward_functions = np.random.binomial(1, self.reward_means)
        reward = self.reward_functions[action]
        if self.use_cost:
            cost = np.random.multivariate_normal(
                self.cost_means, np.eye(self.M)
            )
            cost = self.reward_functions[action]
            return [reward, cost]
        else:
            return reward

    def init_metrics(self):
        if self.use_cost:
            return { 'reward': [0], 'regret': [0], 'cost': [0]}        
        else:
            return { 'reward': [0], 'regret': [0]}
    
    def update_metrics(self, metrics, reward, agent):
        metrics['reward'].append(metrics['reward'][-1]+reward)
        metrics['regret'].append(agent.regret[-1])
        return metrics


class EpidemicControl(World):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    # for example:
    # four possible actions: school closure, diet, vaccine, travel control
    # they each have different levels of interventions: 0-3, 0-3, 0-4, and 0-2
    """

    def __init__(
        self,
        K=5,
        N=[3, 3, 3, 3, 3],
        C=15,
        name=None,
        seed=0,
        reward_means=None,
        cost_means=None,
        combinatorial_cost=False,
        reward_scale=100,
    ):
        super().__init__(name=name, seed=seed)

        self.K = K  # number of possible intervention action dimensions, e.g. 4
        self.N = N  # number of possible values in each action, e.g. [4,4,5,3]
        self.C = C  # dimension of the context, e.g. 100

        self.combinatorial_cost = combinatorial_cost
        self.num_comb_actions = np.prod(N)
        self.num_action_values = np.sum(N)
        if self.combinatorial_cost:
            self.cost_dimension = self.num_comb_actions
        else:
            self.cost_dimension = self.num_action_values
        self.comb_index = self._fill_comb_index(self.N, 0)
        self.action_index = self._fill_action_index(self.N)

        # If we let num_comb_actions = prod(N), i.e. all possible actions sets,
        # if we let num_action_values = sum(N), i.e. all possible action values,
        # by default, among the context C, the first num_action_values features
        # are the cost values for each of all the possible actions. Or in the
        # case, where combinatorial cost are used, meaning all the action costs are
        # interacting with one another, num_comb_actions determines the context dimension.

        # For instance, if there are two action dimensions, school closure and traffic
        # control, thus K = 2, and they each have three levels, thus N = [2, 3], then
        # num_comb_actions = prod(N) = 6, and the first 6 values of the context will be the
        # costs of the 6 possible action values. E.g, if controlling traffic at levels 1, 2, 3
        # when school is open cost the government 12M, 20M and 24M, and controlling traffic at
        # at levels 1, 2, 3  when school is closed cost the government 22M, 30M and 50M.
        # Then the context can be (12, 20, 24, 22, 30, 50, ...).
        # Say, if temperature is another useful measurement for disease spread, we can
        # add it as our 7th feature in the context.

        self.reward_functions = None
        self.cost_functions = None
        if reward_means is None:
            self.reward_means = np.random.uniform(
                0, reward_scale, self.num_comb_actions
            )
        else:
            self.reward_means = reward_means
        if cost_means is None:
            self.cost_means = np.random.uniform(0, reward_scale, self.cost_dimension)
        else:
            self.cost_means = cost_means
 
    def init_metrics(self):
        return { 'reward': [0], 'cost': [0]}      
    
    def update_metrics(self, metrics, reward, agent):
        metrics['reward'].append(metrics['reward'][-1]+reward[0])
        metrics['cost'].append(metrics['cost'][-1]+reward[1])
        return metrics
    
    def _fill_comb_index(self, N, count):
        if len(N) == 1:
            action_dict = {}
            for i in range(N[0]):
                action_dict[i] = count
                count += 1
            return action_dict
        else:
            action_dict = {}
            for i in range(N[0]):
                action_dict[i] = self._fill_comb_index(N[1:], count + i * N[1])
            return action_dict

    def _fill_action_index(self, N):
        action_dict = {}
        count = 0
        for i, n in enumerate(N):
            action_dict[i] = np.arange(count, count + n)
            count += n
        return action_dict

    def _get_action_comb_index(self, action):
        action_index = self.comb_index
        for a in action:
            action_index = action_index[a]
        return action_index

    def _get_action_one_hot(self, action):
        action_one_hot = np.zeros(self.cost_dimension)
        for i, a in enumerate(action):
            action_one_hot[self.action_index[i][a]] = 1
        return action_one_hot


class EpidemicControl_v1(EpidemicControl):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    v1: the cost weight metric is stochastic but stationary (say, for a certain city
    in a given season)
    """

    def provide_context(self, t):
        context = np.random.random(self.C)
        self.cost_functions = np.random.multivariate_normal(
            self.cost_means, np.eye(self.cost_dimension)
        )
        context[: self.cost_dimension] = self.cost_functions
        return context

    def assign_reward(self, action):
        self.reward_functions = np.random.multivariate_normal(
            self.reward_means, np.eye(self.num_comb_actions)
        )
        reward = self.reward_functions[self._get_action_comb_index(action)]
        if self.combinatorial_cost:
            cost = (self.cost_functions[self._get_action_comb_index(action)],)
        else:
            cost = self.cost_functions @ self._get_action_one_hot(action)
        return [reward, cost]


class EpidemicControl_v2(EpidemicControl_v1):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    v2: the cost weight metric is stochastic but nonstationary (say, for a certain city
    which changes its stringency weight every month)
    """

    def __init__(
        self,
        K=5,
        N=[3, 3, 3, 3, 3],
        C=15,
        name=None,
        seed=0,
        reward_means=None,
        cost_means=None,
        combinatorial_cost=False,
        reward_scale=100,
        change_every=30,
    ):
        super().__init__(
            K=K,
            N=N,
            C=C,
            name=name,
            seed=seed,
            reward_means=reward_means,
            cost_means=cost_means,
            combinatorial_cost=combinatorial_cost,
            reward_scale=reward_scale,
        )
        self.change_every = change_every

    def provide_context(self, t):
        context = np.random.random(self.C)
        if t // self.change_every == 0:
            np.random.shuffle(self.cost_means)
        self.cost_functions = np.random.multivariate_normal(
            self.cost_means, np.eye(self.cost_dimension)
        )
        context[: self.cost_dimension] = self.cost_functions
        return context
