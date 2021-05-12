#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions and classes
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


def budget_obj(rewards, obj_params):
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
