# Copyright 2021 Baihan Lin
#
# Licensed under the GNU General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        return {"M": self.M, "oracle": self.reward_means}

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
        return {"K": self.K, "N": self.N, "C": self.C, "oracle": self.reward_means}

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
        for k, v in action.items():
            action_one_hot[self.action_index[k][v]] = 1
        return action_one_hot
