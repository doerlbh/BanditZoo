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
from .base_feedbacks import GaussianFeedback
from .base_feedbacks import BernoulliFeedback
from .utils import check_and_correct_dimensions


class MultiArmedBandits(World):
    """
    Multi-Armed Bandits Problem with Gaussian Rewards
    """

    def __init__(
        self,
        name="MultiArmedBandits",
        seed=0,
        **kwargs,
    ):
        self.n_arms = kwargs.get("n_arms")  # number of arms
        self.reward_means = kwargs.get("reward_means", None)
        self.reward_stds = kwargs.get("reward_stds", None)
        self.reward_scale = kwargs.get("reward_scale", 1)
        self.reward_dimension = kwargs.get("reward_dimension", 1)
        self.reward_function_class = kwargs.get(
            "reward_function_class", GaussianFeedback
        )
        self.reward_reveal_frequency = kwargs.get("reward_reveal_frequency", 1)
        self.reward_reveal_function = kwargs.get("reward_reveal_function", lambda: 1)
        self.cost_means = kwargs.get("cost_means", None)
        self.cost_stds = kwargs.get("cost_stds", None)
        self.cost_scale = kwargs.get("cost_scale", 0)
        self.cost_dimension = kwargs.get("cost_dimension", 1)
        self.cost_function_class = kwargs.get("cost_function_class", GaussianFeedback)
        self.cost_reveal_frequency = kwargs.get("cost_reveal_frequency", 1)
        self.cost_reveal_function = kwargs.get("cost_reveal_function", lambda: 1)
        self.cost_arms = kwargs.get("cost_arms", self.n_arms)
        World.__init__(self, name=name, seed=seed)

        self.build()

    def build(self):
        self._assign_reward_cost_variables()
        self._check_predefined_variables()
        self._reset_reward_function()
        self._reset_cost_function()

    def _reset_reward_function(self):
        self.reward_function = self.reward_function_class(
            means=self.reward_means,
            stds=self.reward_stds,
            scale=self.reward_scale,
            dimension=self.reward_dimension,
            reveal_frequency=self.reward_reveal_frequency,
            reveal_function=self.reward_reveal_function,
            seed=self.seed,
            name="reward_function",
        )

    def _reset_cost_function(self):
        self.cost_function = self.cost_function_class(
            means=self.cost_means,
            stds=self.cost_stds,
            scale=self.cost_scale,
            dimension=self.cost_dimension,
            reveal_frequency=self.cost_reveal_frequency,
            reveal_function=self.cost_reveal_function,
            seed=self.seed,
            name="cost_function",
        )

    def _assign_reward_cost_variables(self):
        def check_none_and_assign(a, b):
            return b if a is None else np.array(a)

        self.reward_means = check_none_and_assign(
            self.reward_means,
            np.random.uniform(
                0, self.reward_scale, (self.n_arms, self.reward_dimension)
            ),
        )
        self.reward_stds = check_none_and_assign(
            self.reward_stds,
            np.random.uniform(
                0, self.reward_scale, (self.n_arms, self.reward_dimension)
            ),
        )
        self.cost_means = check_none_and_assign(
            self.cost_means,
            np.random.uniform(
                0, self.cost_scale, (self.cost_arms, self.cost_dimension)
            ),
        )
        self.cost_stds = check_none_and_assign(
            self.cost_stds,
            np.random.uniform(
                0, self.cost_scale, (self.cost_arms, self.cost_dimension)
            ),
        )

    def _check_predefined_variables(self):
        (
            self.reward_means,
            self.reward_stds,
            self.reward_dimension,
        ) = check_and_correct_dimensions(
            "reward", self.reward_means, self.reward_stds, self.reward_dimension, 2
        )
        (
            self.cost_means,
            self.cost_stds,
            self.cost_dimension,
        ) = check_and_correct_dimensions(
            "cost", self.cost_means, self.cost_stds, self.cost_dimension, 2
        )

    def get_env_config(self):
        return {
            "n_arms": self.n_arms,
            "reward_dimension": self.reward_dimension,
            "reward_means": self.reward_means,
            "reward_stds": self.reward_stds,
        }

    def _provide_contexts(self, t):
        pass

    def _assign_feedbacks(self, action):
        rewards = self.reward_function.get(action)
        costs = self.cost_function.get(action)
        return {"rewards": rewards, "costs": costs}

    def _init_metrics(self):
        metrics_dict = {"reward": [0], "regret": [0], "cost": [0]}
        for i in range(self.reward_dimension):
            metrics_dict["reward_" + str(i)] = [0]
        for i in range(self.cost_dimension):
            metrics_dict["cost_" + str(i)] = [0]
        return metrics_dict

    def _update_metrics(self, metrics, feedbacks, agent):
        metrics["reward"].append(metrics["reward"][-1] + np.mean(feedbacks["rewards"]))
        metrics["cost"].append(metrics["cost"][-1] + np.mean(feedbacks["costs"]))
        for i in range(self.reward_dimension):
            metrics["reward_" + str(i)].append(
                metrics["reward_" + str(i)][-1] + feedbacks["rewards"][i]
            )
        for i in range(self.cost_dimension):
            metrics["cost_" + str(i)].append(
                metrics["cost_" + str(i)][-1] + feedbacks["costs"][i]
            )
        metrics["regret"].append(agent.regret[-1])
        return metrics


class BernoulliMultiArmedBandits(MultiArmedBandits):
    """
    Multi-Armed Bandits Problem with Bernoulli Rewards
    """

    def __init__(
        self,
        name="BernoulliMultiArmedBandits",
        seed=0,
        reward_function_class=BernoulliFeedback,
        **kwargs,
    ):
        MultiArmedBandits.__init__(
            self,
            name=name,
            seed=seed,
            reward_function_class=reward_function_class,
            **kwargs,
        )


class ContextualCombinatorialBandits(MultiArmedBandits):
    """
    Contextual Combinatorial Bandit Problem
    """

    def __init__(
        self,
        name="ContextualCombinatorialBandits",
        seed=0,
        **kwargs,
    ):
        self.action_dimension = kwargs.get(
            "action_dimension", 5
        )  # number of possible intervention action dimensions, e.g. 4
        self.action_options = kwargs.get(
            "action_options", [3, 3, 3, 3, 3]
        )  # number of possible values in each action, e.g. [4,4,5,3]
        self.context_dimension = kwargs.get(
            "context_dimension", 15
        )  # the dimension of the context, e.g. 100
        combinatorial_cost = kwargs.get("combinatorial_cost", False)

        self.n_arms = np.prod(self.action_options)  # the combinatorial count of actions
        self.n_arms_sum = np.sum(
            self.action_options
        )  # the sum of possible individual actions
        self.comb_index = self._fill_comb_index(self.action_options, 0)
        self.action_index = self._fill_action_index(self.action_options)
        self.combinatorial_cost = combinatorial_cost
        self.cost_arms = self.n_arms if self.combinatorial_cost else self.n_arms_sum
        kwargs["n_arms"] = self.n_arms
        kwargs["cost_arms"] = self.cost_arms
        MultiArmedBandits.__init__(self, name=name, seed=seed, **kwargs)

    def get_env_config(self):
        return {
            "action_dimension": self.action_dimension,
            "action_options": self.action_options,
            "context_dimension": self.context_dimension,
            "oracle": self.reward_means,
        }

    def _fill_comb_index(self, action_options, count):
        if len(action_options) == 1:
            action_dict = {}
            for i in range(action_options[0]):
                action_dict[i] = count
                count += 1
            return action_dict
        else:
            action_dict = {}
            for i in range(action_options[0]):
                action_dict[i] = self._fill_comb_index(
                    action_options[1:], count + i * action_options[1]
                )
            return action_dict

    def _fill_action_index(self, action_options):
        action_dict = {}
        count = 0
        for i, n in enumerate(action_options):
            action_dict[i] = np.arange(count, count + n)
            count += n
        return action_dict

    def _get_action_comb_index(self, action):
        action_index = self.comb_index
        for k, v in action.items():
            action_index = action_index[v]
        return action_index

    def _get_action_one_hot(self, action):
        action_one_hot = np.zeros(self.cost_arms)
        for k, v in action.items():
            action_one_hot[self.action_index[k][v]] = 1
        return action_one_hot
