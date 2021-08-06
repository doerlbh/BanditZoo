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
utils functions and classes related to agents
"""

import numpy as np


def default_obj(rewards, obj_params):
    """the combined reward functions in multi-objective scenario

    Args:
        rewards (Any): the reward feedback signals
        obj_params (Dict[str, Any]): the parameters to balance among different rewards

    Returns:
        [Any]: a combined reward functions modulated by the weight
    """
    return np.sum(rewards)


def budget_obj_v1(rewards, obj_params):
    """the combined reward functions that balances a reward and a cost.

    Args:
        rewards (Any): the reward feedback signals
        obj_params (Dict[str, Any]): the parameters to balance among different rewards

    Returns:
        [Any]: a combined reward functions modulated by the weight
    """
    reward, cost = rewards
    w = obj_params.get("w", 0.5)
    return w * reward + (1 - w) / cost


def budget_obj_v2(rewards, obj_params):
    """the combined reward functions that balances a reward and a cost.

    Args:
        rewards (Any): the reward feedback signals
        obj_params (Dict[str, Any]): the parameters to balance among different rewards

    Returns:
        [Any]: a combined reward functions modulated by the weight
    """
    reward, cost = rewards
    w = obj_params.get("w", 0.5)
    return w * reward - (1 - w) * cost
