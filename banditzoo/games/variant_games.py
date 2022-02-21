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
Classes of variant agents.
"""

import copy
import json
from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
from collections import defaultdict
from .base_games import Game
from .utils import get_comb_parameters
from .utils import quantize_metrics


class MultiObjectiveGame(Game):
    """
    Class of Multi-Objective Game object
    """

    def __init__(
        self, name=None, seed=0, n_world_instances=None, n_agent_instances=None
    ):
        """Generate a Game object.

        Args:
            name (str, optional): [name of the game]. Defaults to None.
            seed (int, optional): [random seed]. Defaults to 0.
            n_world_instances (int, optional): [number of world instances per world class]. Defaults to None.
            n_agent_instances (int, optional): [number of agent instances per agent class]. Defaults to None.
        """
        Game.__init__(
            self,
            name=name,
            seed=seed,
            n_world_instances=n_world_instances,
            n_agent_instances=n_agent_instances,
        )

        self.games = {}
        self.params_search = None
        self.params_list = None
        self.params_lock = False

    def set_params_sweep(self, **kwargs):
        """Set the parameters to search for multi-objective pareto-optimal search.

        Args:
            **kwargs ([any], optional): [the params to search in the objective function].
        """

        if self.params_lock:
            print("[WARNING] the previous params sweep is now overwritten.")
        self.params_search = kwargs
        self.params_list = get_comb_parameters(self.params_search)

        self.games = {}
        for p in self.params_list:
            game = copy.deepcopy(self)
            game.set_agent_params(**p)
            self.games[json.dumps(p)] = game
        self.params_lock = True

    def set_agent_params(self, **kwargs):
        """Set the objective function parameters for all agents.

        Args:
            **kwargs (any, optional): [the parameters to set the objective function of the agents].

        Raises:
            Exception: [if the game has no agent, no objective params can be set].
        """

        if len(self.agent_names) == 0:
            raise Exception(
                "Please initiate all the agents before setting objective params."
            )
        if len(kwargs) == 0:
            raise Exception(
                "Please have at least one objective function parameters to set."
            )

        for k in self.world_names:
            for i in range(self.n_world_instances):
                for a_name in self.agent_names:
                    for j in range(self.n_agent_instances):
                        for params_name, params_val in kwargs.items():
                            if (
                                hasattr(self.agent_pools[k][i][a_name][j], "obj_params")
                                and params_name
                                in self.agent_pools[k][i][a_name][j].obj_params.keys()
                            ):
                                self.agent_pools[k][i][a_name][j].obj_params[
                                    params_name
                                ] = params_val

    def get_tabular_metrics(self):
        """Extract the metrics in a tabular format in a pandas dataframe."""
        if self.params_lock:
            return {
                g_params: g.get_tabular_metrics() for g_params, g in self.games.items()
            }
        else:
            return Game.get_tabular_metrics(self)

    def _aggregate_world_metrics(self):
        """Aggregate the metrics in the n_agent_instances dimension (the agent instances)."""
        if self.params_lock:
            return {
                g_params: g._aggregate_world_metrics()
                for g_params, g in self.games.items()
            }
        else:
            return Game._aggregate_world_metrics(self)

    def _aggregate_agent_metrics(self):
        """Aggregate the metrics in both n_agent_instances and n_world_instances dimensions (world and agent instances)."""
        if self.params_lock:
            return {
                g_params: g._aggregate_agent_metrics()
                for g_params, g in self.games.items()
            }
        else:
            return Game._aggregate_agent_metrics(self)

    def run_experiments(self, T, progress=False):
        """Run the game with certain iterations.

        Args:
            T (int): [number of time steps for the game in this run].
            progress (bool, optional): [whether to print progress]. Defaults to False.
        """
        if self.params_lock:
            for g_params, g in self.games.items():
                print("\n\nRunning experiment for objective parameters: ", g_params)
                g.run_experiments(T=T, progress=progress)
        else:
            Game.run_experiments(self, T=T, progress=progress)

    def get_full_data(self):
        """Output the full historical data of the game.

        Returns:
            [dict]: [a dict with params name and tuple with the world instances, agent instances,
                histories and metrics]. If no params have been specified, the returns just a tuple
                for the default game instance.
        """
        if self.params_lock:
            return {
                g_params: (
                    g.world_pools,
                    g.agent_pools,
                    g.history_pools,
                    g.metrics_pools,
                )
                for g_params, g in self.games.items()
            }
        else:
            return Game.get_full_data(self)

    def get_metrics(self, form="tabular"):
        """Output the metrics of the agents in the worlds.

        Args:
            group_by (str, optional): [output format of the metrics].
                If 'tabular', the metrics are stored in a pandas dataframe.
                If 'agent', the metrics are aggregated by both n_world_instances and n_agent_instances dimension.
                If 'world', the metrics are aggregated only in the n_agent_instances dimension (the
                agent instances) and not the world instances. Defaults to 'tabular'.

        Returns:
            [dict, pd.DataFrame]: [the aggregated metrics of the agents in different
                objective params]. If no params have been specified, the returns the
                metrics for the default game instance.

        Raises:
            ValueError: [if the game has started, no new agent can enter].
        """
        if self.params_lock:
            return {
                g_params: g.get_metrics(form=form) for g_params, g in self.games.items()
            }
        else:
            return Game.get_metrics(self, form=form)

    def get_pareto_metrics(self, quantile_bin=False, n_bins=20):
        """Output the metrics of the agents in the worlds.

        Args:
            quantile_bin (bool, optional): [whether to use quantile in quantization]
            n_bins (int, optional): [number of bins in the quantization].

        Returns:
            [pd.DataFrame]: [the aggregated metrics of the agents in different objective params].

        Raises:
            Exception: [if no objective params were given].
        """

        if self.params_search is None:
            raise Exception(
                "There is no objective function parameters to construct the pareto frontier."
            )
        metrics = self.get_metrics(form="tabular")
        m_dfs = []
        for g_params, m_df in metrics.items():
            for p_key, p_val in json.loads(g_params).items():
                m_df[p_key] = [p_val] * m_df.shape[0]
            m_dfs.append(m_df)
        return quantize_metrics(
            pd.concat(m_dfs, ignore_index=True), quantile_bin, n_bins
        )
