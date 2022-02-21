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
Base class of worlds.

usage:
w = World()
...
w.add_agent(agent)
w.run_experiments(T=1000)
...
results = w.get_results()
"""

from abc import abstractmethod
import numpy as np

from .utils import print_progress


class World(object):
    """
    Base class of world object
    """

    def __init__(self, name=None, seed=0):
        """Initialize the base world object.

        Args:
            name (str, optional): [world name]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
        """

        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.agents = []
        self.history = {}
        self.metrics = {}

    def add_agent(self, agent):
        """Enter agent into the world.

        Args:
            agent (banditzoo.agents class object): [an agent instance].
        """
        self.history[len(self.agents)] = []
        self.metrics[len(self.agents)] = self._init_metrics()
        agent.build(**self.get_env_config())
        self.agents.append(agent)

    def add_agent_pool(self, agents):
        """Enter a pool of agents into the world.

        Args:
            agents (array of banditzoo.agents objects): [a list of agent instances or indices]
        """
        for a in agents:
            self.add_agent(a)

    def filter_agent(self, agent_name, get_index=False):
        """Filter agent given an agent name.

        Args:
            agent_name (str): [the agent name].
            get_index (bool, optional): [whether to return the agent indices or instances]. Defaults to False.

        Returns:
            [list of banditzoo.agents objects]: [a list of agent instances or indices that matches the agent name].
        """
        filtered = np.array([a.name == agent_name for a in self.agents])
        if get_index:
            return np.arange(len(self.agents))[filtered]
        else:
            return np.array(self.agents)[filtered]

    def filter_history(self, agent_name):
        return [self.history[a] for a in self.filter_agent(agent_name, get_index=True)]

    def filter_metrics(self, agent_name):
        return [self.metrics[a] for a in self.filter_agent(agent_name, get_index=True)]

    def run_experiments(self, T, progress=False):
        for t in range(T):
            context = self._provide_contexts(t)
            for i in range(len(self.agents)):
                self.agents[i].observe(context)
                a = self.agents[i].act()
                self.history[i].append(a)
                feedback = self._assign_feedbacks(a)
                self.agents[i].update(feedback)
                self.metrics[i] = self._update_metrics(
                    self.metrics[i], feedback, self.agents[i]
                )
            if progress:
                print_progress(t, T)

    def get_results(self):
        return self.agents, self.history, self.metrics

    @abstractmethod
    def _provide_contexts(self):
        raise NotImplementedError

    @abstractmethod
    def _assign_feedbacks(self):
        raise NotImplementedError

    @abstractmethod
    def _init_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def _update_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def get_env_config(self):
        raise NotImplementedError
