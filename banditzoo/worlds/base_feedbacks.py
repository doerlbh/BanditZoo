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
Base class of feedbacks.

usage:
f = Feedback()
...
feedback = f.get(action)
"""

from abc import abstractmethod
import numpy as np


class Feedback(object):
    """
    Base class of feedback object
    """

    def __init__(
        self, name=None, seed=0, reveal_frequency=1, reveal_function=lambda: 1
    ):
        """Initialize the base feedback object.

        Args:
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (float, optional): [frequency to reveal feedback], Defaults to 1.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """

        self.name = name
        self.seed = seed
        self.reveal_frequency = reveal_frequency
        self.reveal_function = reveal_function
        np.random.seed(seed)

        self.feedback_function = None

    def get(self, action, one_hot=False):
        reveal = self.reveal_function() * np.random.binomial(1, self.reveal_frequency)
        self.feedback_function = self.draw_function()
        if one_hot:
            actual_feedbacks = [action @ np.array(self.feedback_function).squeeze()]
        else:
            actual_feedbacks = [r[action] for r in self.feedback_function]
        if reveal:
            feedbacks = {
                "revealed_feedback": actual_feedbacks,
                "hidden_feedback": actual_feedbacks,
            }
        else:
            feedbacks = {"revealed_feedback": None, "hidden_feedback": actual_feedbacks}
        return feedbacks

    @abstractmethod
    def draw_function(self, action):
        raise NotImplementedError


class GaussianFeedback(Feedback):
    """
    class of Gaussian feedback object
    """

    def __init__(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        scale: float,
        dimension: int,
        name: str = None,
        seed: int = 0,
        reveal_frequency: int = 1,
        reveal_function=lambda: 1,
        **kwargs
    ):
        """Initialize the Gaussian feedback object.

        Args:
            means (np.ndarray): [the average values of the feedback].
            stds (np.ndarray): [the standard deviations of the feedback].
            scale (float): [the scale of the feedback].
            dimension (int): [the number of stream or dimension of the feedback parameters].
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (float, optional): [frequency to reveal feedback], Defaults to 1.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """
        self.means = means
        self.stds = stds
        self.scale = scale
        self.dimension = dimension
        Feedback.__init__(
            self,
            name=name,
            seed=seed,
            reveal_frequency=reveal_frequency,
            reveal_function=reveal_function,
        )

    def draw_function(self):
        return [
            np.random.multivariate_normal(self.means[:, i], np.diag(self.stds[:, i]))
            for i in range(self.dimension)
        ]


class BernoulliFeedback(Feedback):
    """
    class of Bernoulli feedback object
    """

    def __init__(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        scale: float,
        dimension: int,
        name: str = None,
        seed: int = 0,
        reveal_frequency: int = 1,
        reveal_function=lambda: 1,
        **kwargs
    ):
        """Initialize the Bernoulli feedback object.

        Args:
            means (np.ndarray): [the average values of the feedback].
            stds (np.ndarray): [the standard deviations of the feedback].
            scale (float): [the scale of the feedback].
            dimension (int): [the number of stream or dimension of the feedback parameters].
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (float, optional): [frequency to reveal feedback], Defaults to 1.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """
        self.means = means
        self.stds = stds
        self.scale = scale
        self.dimension = dimension
        Feedback.__init__(
            self,
            name=name,
            seed=seed,
            reveal_frequency=reveal_frequency,
            reveal_function=reveal_function,
        )

    def draw_function(self):
        return [np.random.binomial(1, self.means[:, i]) for i in range(self.dimension)]
