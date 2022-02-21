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

from typing import (
    Any,
    List,
    Dict,
    Optional,
    Callable,
)


class Feedback(object):
    """
    Base class of feedback object
    """

    def __init__(
        self,
        dimension: int,
        name: str = None,
        seed: int = 0,
        reveal_frequency: List[float] = [1],
        reveal_function: Callable = lambda: 1,
    ):
        """Initialize the base feedback object.

        Args:
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (List[float], optional): [frequency to reveal feedback], Defaults to 1s.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """

        self.name = name
        self.seed = seed
        self.reveal_frequency = reveal_frequency
        self.reveal_function = reveal_function
        self.dimension = dimension
        np.random.seed(seed)

        if self.dimension != len(self.reveal_frequency):
            raise ValueError(
                "Please specify the same shape for feedback dimension and reveal_frequency, now "
                + str(self.dimension)
                + " and "
                + str(len(self.reveal_frequency))
            )

        self.feedback_function = None

    def get(self, action, one_hot=False):
        reveal = [
            self.reveal_function() * np.random.binomial(1, rf)
            for rf in self.reveal_frequency
        ]
        self.feedback_function = self.draw_function()
        if one_hot:
            actual_feedbacks = [action @ np.array(self.feedback_function).squeeze()]
        else:
            actual_feedbacks = [r[action] for r in self.feedback_function]
        feedbacks = {
            "revealed_feedback": actual_feedbacks,
            "hidden_feedback": actual_feedbacks,
        }
        for i in range(self.dimension):
            if not reveal[i]:
                feedbacks["revealed_feedback"][i] = None
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
        min: float,
        max: float,
        dimension: int,
        name: str = None,
        seed: int = 0,
        reveal_frequency: List[float] = [1],
        reveal_function: Callable = lambda: 1,
        **kwargs
    ):
        """Initialize the Gaussian feedback object.

        Args:
            means (np.ndarray): [the average values of the feedback].
            stds (np.ndarray): [the standard deviations of the feedback].
            scale (float): [the scale of the feedback].
            min (float): [the clipping min of the feedback].
            max (float): [the clipping max of the feedback].
            dimension (int): [the number of stream or dimension of the feedback parameters].
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (List[float], optional): [frequency to reveal feedback], Defaults to 1s.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """
        self.means = means
        self.stds = stds
        self.scale = scale
        self.min = min
        self.max = max
        Feedback.__init__(
            self,
            dimension=dimension,
            name=name,
            seed=seed,
            reveal_frequency=reveal_frequency,
            reveal_function=reveal_function,
        )

    def draw_function(self):
        return list(
            np.clip(
                [
                    np.random.multivariate_normal(
                        self.means[:, i], np.diag(self.stds[:, i])
                    )
                    for i in range(self.dimension)
                ],
                self.min,
                self.max,
            )
        )


class BernoulliFeedback(Feedback):
    """
    class of Bernoulli feedback object
    """

    def __init__(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        scale: float,
        min: float,
        max: float,
        dimension: int,
        name: str = None,
        seed: int = 0,
        reveal_frequency: List[float] = [1],
        reveal_function: Callable = lambda: 1,
        **kwargs
    ):
        """Initialize the Bernoulli feedback object.

        Args:
            means (np.ndarray): [the average values of the feedback].
            stds (np.ndarray): [the standard deviations of the feedback].
            scale (float): [the scale of the feedback].
            min (float): [the clipping min of the feedback].
            max (float): [the clipping max of the feedback].
            dimension (int): [the number of stream or dimension of the feedback parameters].
            name (str, optional): [feedback name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            reveal_frequency (List[float], optional): [frequency to reveal feedback], Defaults to 1s.
            reveal_function (Callable, optional): [function to reveal feedback], Defaults to lambda:1.
        """
        self.means = means
        self.stds = stds
        self.scale = scale
        self.min = min
        self.max = max
        if self.scale != 1:
            raise ValueError(
                "The reward_scale in Bernoulli bandits can only be 1, now "
                + str(self.scale)
            )
        Feedback.__init__(
            self,
            dimension=dimension,
            name=name,
            seed=seed,
            reveal_frequency=reveal_frequency,
            reveal_function=reveal_function,
        )

    def draw_function(self):
        return list(
            np.clip(
                [
                    np.random.binomial(1, self.means[:, i])
                    for i in range(self.dimension)
                ],
                self.min,
                self.max,
            )
        )
