#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions and classes
"""


def default_obj(rewards, w):
    """the combined reward functions in multi-objective scenario

    Args:
        rewards (Any): the reward feedback signals
        w (Any): the weight to balance among different rewards

    Returns:
        [Any]: a combined reward functions modulated by the weight
    """
    r, s = rewards
    return w * r + (1 - w) / s
