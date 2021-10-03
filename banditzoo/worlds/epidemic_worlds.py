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
        cost_scale = kwargs.get("cost_scale", 1)
        ContextualCombinatorialBandits.__init__(
            self,
            name=name,
            seed=seed,
            cost_scale=cost_scale,
            **kwargs,
        )

        # If we let n_arms = prod(action_options), i.e. all possible actions sets,
        # if we let n_arms_sum = sum(action_options), i.e. all possible action values,
        # by default, among the context context_dimension, the first n_arms_sum features
        # are the cost values for each of all the possible actions. Or in the
        # case, where combinatorial cost are used, meaning all the action costs are
        # interacting with one another, n_arms determines the context dimension.

        # For instance, if there are two action dimensions, school closure and traffic
        # control, thus action_dimension = 2, and they each have three levels, thus action_options = [2, 3], then
        # n_arms = prod(action_options) = 6, and the first 6 values of the context will be the
        # costs of the 6 possible action values. E.g, if controlling traffic at levels 1, 2, 3
        # when school is open cost the government 12M, 20M and 24M, and controlling traffic at
        # at levels 1, 2, 3  when school is closed cost the government 22M, 30M and 50M.
        # Then the context can be (12, 20, 24, 22, 30, 50, ...).
        # Say, if temperature is another useful measurement for disease spread, we can
        # add it as our 7th feature in the context.

    def _provide_contexts(self, t):
        context = np.random.random(self.context_dimension)
        cost_parameters = np.array(self.cost_function.draw_function()).squeeze()
        context[: len(cost_parameters)] = cost_parameters
        return context

    def _assign_feedbacks(self, action):
        rewards = self.reward_function.get(self._get_action_comb_index(action))
        if self.combinatorial_cost:
            costs = self.cost_function.get(self._get_action_comb_index(action))
        else:
            costs = self.cost_function.get(
                self._get_action_one_hot(action), one_hot=True
            )
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
        change_every = kwargs.get("change_every", 10)
        EpidemicControl.__init__(
            self,
            name=name,
            seed=seed,
            **kwargs,
        )
        self.change_every = change_every

    def _provide_contexts(self, t):
        context = np.random.random(self.context_dimension)
        if t // self.change_every == 0:
            np.random.shuffle(self.cost_means)
            self._reset_cost_function()
        cost_parameters = np.array(self.cost_function.draw_function()).squeeze()
        context[: len(cost_parameters)] = cost_parameters
        return context
