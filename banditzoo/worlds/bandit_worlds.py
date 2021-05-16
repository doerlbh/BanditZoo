#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes of real-world bandit environments.
"""

import numpy as np

from .base_worlds import World


class BernoulliMultiArmedBandits(World):
    """
    Multi-Armed Bandits Problem with Bernoulli rewards
    """

    def __init__(
        self,
        name="BernoulliMultiArmedBandits",
        seed=0,
        **kwargs,
    ):
        M = kwargs.get("M", 5)  # number of arms
        use_cost = kwargs.get("use_cost", False)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        reward_scale = kwargs.get("reward_scale", 1)
        cost_scale = kwargs.get("cost_scale", 1)
        World.__init__(self, name=name, seed=seed)

        self.M = M
        self.use_cost = use_cost
        self.reward_functions = None
        self.cost_functions = None
        if reward_means is None:
            self.reward_means = np.random.uniform(0, reward_scale, self.M)
        else:
            self.reward_means = reward_means
        if self.use_cost:
            if cost_means is None:
                self.cost_means = np.random.uniform(0, cost_scale, self.M)
            else:
                self.cost_means = cost_means

    def get_env_config(self):
        return {"M": self.M}

    def provide_context(self, t):
        pass

    def assign_reward(self, action):
        self.reward_functions = np.random.binomial(1, self.reward_means)
        reward = self.reward_functions[action]
        if self.use_cost:
            cost = np.random.multivariate_normal(self.cost_means, np.eye(self.M))
            cost = self.reward_functions[action]
            return [reward, cost]
        else:
            return reward

    def init_metrics(self):
        if self.use_cost:
            return {"reward": [0], "regret": [0], "cost": [0]}
        else:
            return {"reward": [0], "regret": [0]}

    def update_metrics(self, metrics, reward, agent):
        metrics["reward"].append(metrics["reward"][-1] + reward)
        metrics["regret"].append(agent.regret[-1])
        return metrics


class ContextualCombinatorialBandits(World):
    """
    Contextual Combinatorial Bandit Problem
    """

    def __init__(
        self,
        name="ContextualCombinatorialBandits",
        seed=0,
        **kwargs,
    ):
        K = kwargs.get("K", 5)
        N = kwargs.get("N", [3, 3, 3, 3, 3])
        C = kwargs.get("C", 15)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        use_cost = kwargs.get("use_cost", False)
        combinatorial_cost = kwargs.get("combinatorial_cost", False)
        reward_scale = kwargs.get("reward_scale", 1)
        cost_scale = kwargs.get("cost_scale", 1)
        World.__init__(self, name=name, seed=seed)

        self.K = K  # number of possible intervention action dimensions, e.g. 4
        self.N = N  # number of possible values in each action, e.g. [4,4,5,3]
        self.C = C  # dimension of the context, e.g. 100

        self.num_comb_actions = np.prod(N)
        self.num_action_values = np.sum(N)
        self.comb_index = self._fill_comb_index(self.N, 0)
        self.action_index = self._fill_action_index(self.N)
        self.reward_functions = None
        if reward_means is None:
            self.reward_means = np.random.uniform(
                0, reward_scale, self.num_comb_actions
            )
        else:
            self.reward_means = reward_means

        self.use_cost = use_cost
        if self.use_cost:
            self.combinatorial_cost = combinatorial_cost
            if self.combinatorial_cost:
                self.cost_dimension = self.num_comb_actions
            else:
                self.cost_dimension = self.num_action_values
            self.cost_functions = None
            if cost_means is None:
                self.cost_means = np.random.uniform(0, cost_scale, self.cost_dimension)
            else:
                self.cost_means = cost_means

    def get_env_config(self):
        return {"K": self.K, "N": self.N, "C": self.C}

    def init_metrics(self):
        if self.use_cost:
            return {"reward": [0], "cost": [0]}
        else:
            return {"reward": [0]}

    def update_metrics(self, metrics, reward, agent):
        metrics["reward"].append(metrics["reward"][-1] + reward[0])
        if self.use_cost:
            metrics["cost"].append(metrics["cost"][-1] + reward[1])
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
        for k, v in action.items():
            action_index = action_index[v]
        return action_index

    def _get_action_one_hot(self, action):
        action_one_hot = np.zeros(self.cost_dimension)
        for i, a in enumerate(action):
            action_one_hot[self.action_index[i][a]] = 1
        return action_one_hot


class EpidemicControl(ContextualCombinatorialBandits):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    # for example:
    # four possible actions: school closure, diet, vaccine, travel control
    # they each have different levels of interventions: 0-3, 0-3, 0-4, and 0-2
    """

    def __init__(
        self,
        name="EpidemicControl",
        seed=0,
        **kwargs,
    ):
        K = kwargs.get("K", 5)
        N = kwargs.get("N", [3, 3, 3, 3, 3])
        C = kwargs.get("C", 15)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        use_cost = kwargs.get("use_cost", True)
        combinatorial_cost = kwargs.get("combinatorial_cost", False)
        reward_scale = kwargs.get("reward_scale", 1)
        cost_scale = kwargs.get("cost_scale", 1)
        ContextualCombinatorialBandits.__init__(
            self,
            K=K,
            N=N,
            C=C,
            reward_means=reward_means,
            cost_means=cost_means,
            use_cost=use_cost,
            combinatorial_cost=combinatorial_cost,
            reward_scale=reward_scale,
            cost_scale=cost_scale,
            name=name,
            seed=seed,
        )

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
        if self.use_cost:
            if self.combinatorial_cost:
                cost = (self.cost_functions[self._get_action_comb_index(action)],)
            else:
                cost = self.cost_functions @ self._get_action_one_hot(action)
            return [reward, cost]
        else:
            return [reward]


class EpidemicControl_v1(EpidemicControl):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    v1: the cost weight metric is stochastic but stationary (say, for a certain city
    in a given season)
    """

    def __init__(
        self,
        name="EpidemicControl_v1",
        seed=0,
        **kwargs,
    ):
        EpidemicControl.__init__(self, name=name, seed=seed, **kwargs)


class EpidemicControl_v2(EpidemicControl):
    """
    Contextual Combinatorial Bandit Problem of Epidemic Control

    v2: the cost weight metric is stochastic but nonstationary (say, for a certain city
    which changes its stringency weight every month)
    """

    def __init__(
        self,
        name="EpidemicControl_v2",
        seed=0,
        **kwargs,
    ):
        K = kwargs.get("K", 5)
        N = kwargs.get("N", [3, 3, 3, 3, 3])
        C = kwargs.get("C", 15)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        combinatorial_cost = kwargs.get("combinatorial_cost", False)
        reward_scale = kwargs.get("reward_scale", 1)
        change_every = kwargs.get("change_every", 10)
        EpidemicControl.__init__(
            self,
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
