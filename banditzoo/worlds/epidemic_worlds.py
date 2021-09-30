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
Classes of real-world bandit environments on epidemic control.
"""

import numpy as np

from .base_worlds import World
from .bandit_worlds import ContextualCombinatorialBandits


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
        action_dimension = kwargs.get("action_dimension", 5)
        action_options = kwargs.get("action_options", [3, 3, 3, 3, 3])
        context_dimension = kwargs.get("context_dimension", 15)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        combinatorial_cost = kwargs.get("combinatorial_cost", False)
        reward_scale = kwargs.get("reward_scale", 1)
        cost_scale = kwargs.get("cost_scale", 1)
        ContextualCombinatorialBandits.__init__(
            self,
            action_dimension=action_dimension,
            action_options=action_options,
            context_dimension=context_dimension,
            reward_means=reward_means,
            cost_means=cost_means,
            combinatorial_cost=combinatorial_cost,
            reward_scale=reward_scale,
            cost_scale=cost_scale,
            name=name,
            seed=seed,
        )

        # If we let num_comb_actions = prod(action_options), i.e. all possible actions sets,
        # if we let num_action_values = sum(action_options), i.e. all possible action values,
        # by default, among the context context_dimension, the first num_action_values features
        # are the cost values for each of all the possible actions. Or in the
        # case, where combinatorial cost are used, meaning all the action costs are
        # interacting with one another, num_comb_actions determines the context dimension.

        # For instance, if there are two action dimensions, school closure and traffic
        # control, thus action_dimension = 2, and they each have three levels, thus action_options = [2, 3], then
        # num_comb_actions = prod(action_options) = 6, and the first 6 values of the context will be the
        # costs of the 6 possible action values. E.g, if controlling traffic at levels 1, 2, 3
        # when school is open cost the government 12M, 20M and 24M, and controlling traffic at
        # at levels 1, 2, 3  when school is closed cost the government 22M, 30M and 50M.
        # Then the context can be (12, 20, 24, 22, 30, 50, ...).
        # Say, if temperature is another useful measurement for disease spread, we can
        # add it as our 7th feature in the context.

    def _provide_contexts(self, t):
        context = np.random.random(self.context_dimension)
        self.cost_functions = np.random.multivariate_normal(
            self.cost_means, np.eye(self.cost_dimension)
        )
        context[: self.cost_dimension] = self.cost_functions
        return context

    def _assign_feedbacks(self, action):
        self.reward_functions = np.random.multivariate_normal(
            self.reward_means, np.eye(self.num_comb_actions)
        )
        reward = self.reward_functions[self._get_action_comb_index(action)]
        if self.combinatorial_cost:
            cost = self.cost_functions[self._get_action_comb_index(action)]
        else:
            cost = self.cost_functions @ self._get_action_one_hot(action)
        rewards = [reward]
        costs = [cost]
        return {"rewards": rewards, "costs": costs}


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
        action_dimension = kwargs.get("action_dimension", 5)
        action_options = kwargs.get("action_options", [3, 3, 3, 3, 3])
        context_dimension = kwargs.get("context_dimension", 15)
        reward_means = kwargs.get("reward_means", None)
        cost_means = kwargs.get("cost_means", None)
        combinatorial_cost = kwargs.get("combinatorial_cost", False)
        reward_scale = kwargs.get("reward_scale", 1)
        change_every = kwargs.get("change_every", 10)
        EpidemicControl.__init__(
            self,
            action_dimension=action_dimension,
            action_options=action_options,
            context_dimension=context_dimension,
            name=name,
            seed=seed,
            reward_means=reward_means,
            cost_means=cost_means,
            combinatorial_cost=combinatorial_cost,
            reward_scale=reward_scale,
        )
        self.change_every = change_every

    def _provide_contexts(self, t):
        context = np.random.random(self.context_dimension)
        if t // self.change_every == 0:
            np.random.shuffle(self.cost_means)
        self.cost_functions = np.random.multivariate_normal(
            self.cost_means, np.eye(self.cost_dimension)
        )
        context[: self.cost_dimension] = self.cost_functions
        return context
