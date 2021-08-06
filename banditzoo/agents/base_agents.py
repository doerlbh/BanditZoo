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
a.update(reward,cost)
"""

from abc import abstractmethod
import numpy as np

from typing import (
    Any,
    List,
    Dict,
    Optional,
)


class IOAgent(object):
    """
    Base agent with IO functions
    """

    def __init__(self, load: bool=False, fpath: Optional[str]=None):
        if load:
            self.load(fpath)
    
    def load(self, fpath: Optional[str]=None):
        """[summary]

        Args:
            fpath (Optional[str], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        #TODO
        raise NotImplementedError

    def save(self, fpath: Optional[str]=None):
        """[summary]

        Args:
            fpath (Optional[str], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        #TODO
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

    def observe(self, c: Optional[np.ndarray]=None):
        """[summary]

        Args:
            c (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        self.c_t = c  # update context

    def update(self, rewards:  Optional[List[Any]] =None):
        """[summary]

        Args:
            rewards (Optional[List[Any]], optional): [description]. Defaults to None.
        """
        self.update_agent(rewards)
        self.update_metrics(rewards)

    @abstractmethod
    def act(self):
        raise NotImplementedError

    @abstractmethod
    def update_agent(self):
        raise NotImplementedError

    @abstractmethod
    def update_metrics(self):
        raise NotImplementedError
 
 
class OfflineAgent(object):
    """
    Base online reinforcement learning agent class
    """
    
    @abstractmethod
    def fit(self):    
        raise NotImplementedError   
    
class Agent(IOAgent, OnlineAgent, OfflineAgent):
    """
    Base reinforcement learning agent class
    """

    def __init__(self, seed: int =0, load: bool =False, name: Optional[str]=None, fpath:Optional[str]=None, **kwargs: Optional[Dict[str, Any]]):
        IOAgent.__init__(self, load=load, fpath=fpath)
        OnlineAgent.__init__(self)
        
        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.build(**kwargs)

    def combine_rewards(self, rewards: Optional[List[Any]] =None) -> float:
        """[summary]

        Args:
            rewards (Optional[List[Any]], optional): [description]. Defaults to None.

        Returns:
            float: [description]
        """
        return np.sum(rewards)
    
    @abstractmethod
    def build(self):
        pass

