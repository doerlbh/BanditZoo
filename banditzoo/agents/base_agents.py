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
Base class of agents.

usage:
a = Agent()
...
a.observe(context)
actions = a.act()
...
a.update(feedbacks)
"""

from abc import abstractmethod
import numpy as np
from .utils import default_obj


from typing import (
    Any,
    List,
    Dict,
    Optional,
    Callable,
)


class IOAgent(object):
    """
    Base agent with IO functions
    """

    def __init__(self, load: bool = False, fpath: Optional[str] = None):
        if load:
            self.load(fpath)

    def load(self, fpath: Optional[str] = None):
        """[summary]

        Args:
            fpath (Optional[str], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        # TODO
        raise NotImplementedError

    def save(self, fpath: Optional[str] = None):
        """[summary]

        Args:
            fpath (Optional[str], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        # TODO
        raise NotImplementedError


class OnlineAgent(object):
    """
    Base online reinforcement learning agent class
    """

    def __init__(self):

        self.i_t = None  # current action
        self.c_t = None  # current context
        self.t_t = 0  # current iteration

        self.reward = []  # keep track of rewards

    def observe(self, c: Optional[np.ndarray] = None):
        """[summary]

        Args:
            c (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        self.c_t = c  # update context

    def update(self, feedbacks: Optional[List[Any]] = None):
        """[summary]

        Args:
            feedbacks (Optional[List[Any]], optional): [description]. Defaults to None.
        """
        self._update_agent(feedbacks)
        self._update_metrics(feedbacks)

    @abstractmethod
    def act(self):
        raise NotImplementedError

    @abstractmethod
    def _update_agent(self):
        raise NotImplementedError

    @abstractmethod
    def _update_metrics(self):
        raise NotImplementedError


class OfflineAgent(object):
    """
    Base online reinforcement learning agent class
    """

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class MultiObjectiveAgent(object):
    """
    Base agent that learns from multiple feedback signals.
    """

    def __init__(
        self,
        obj_func: Callable = default_obj,
        obj_params: Dict[str, Any] = {},
        **kwargs,
    ):
        self.obj_func = obj_func  # the combined objective function
        self.obj_params = obj_params  # the params to compute objective function

    def combine_feedbacks(self, feedbacks: Dict[str, Any]) -> float:
        return self.obj_func(feedbacks, self.obj_params)


class Agent(IOAgent, OnlineAgent, OfflineAgent, MultiObjectiveAgent):
    """
    Base reinforcement learning agent class
    """

    def __init__(
        self,
        seed: int = 0,
        load: bool = False,
        obj_func: Callable = default_obj,
        obj_params: Dict[str, Any] = {},
        name: Optional[str] = None,
        fpath: Optional[str] = None,
        **kwargs: Optional[Dict[str, Any]],
    ):
        IOAgent.__init__(self, load=load, fpath=fpath)
        OnlineAgent.__init__(self)
        MultiObjectiveAgent.__init__(self, obj_func=obj_func, obj_params=obj_params)

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.build(**kwargs)

    @abstractmethod
    def build(self):
        pass
