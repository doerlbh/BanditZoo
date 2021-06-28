#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions and classes related to games
"""

import itertools
import pandas as pd


def get_comb_parameters(params_dict):
    """Get the combinatorial set of parameters.

    Args:
        params_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    keys, values = zip(*params_dict.items())
    comb_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return comb_dicts


def quantize_metrics(df, quantile_bin=False, nbins=20):
    """Quantize the metrics for plotting.

    Args:
        df ([type]): [description]
        quantile_bin (bool, optional): [description]. Defaults to False.
        nbins (int, optional): [description]. Defaults to 20.

    Returns:
        [type]: [description]
    """
    df["reward_bins"] = [None] * len(df)
    df["cost_bins"] = [None] * len(df)
    for a in df["agent"].unique():
        selected = df["agent"]==a
        if quantile_bin:
            df.loc[selected,"reward_bins"] = pd.cut(df[selected]["reward"], bins=nbins)
            df.loc[selected,"cost_bins"] = pd.cut(df[selected]["cost"], bins=nbins)
        else:
            df.loc[selected,"reward_bins"] = pd.qcut(df[selected]["reward"], q=nbins)
            df.loc[selected,"cost_bins"] = pd.qcut(df[selected]["cost"], q=nbins)        
    df['reward_'] =df.groupby('reward_bins')['reward'].transform("mean")
    df['cost_'] =df.groupby('cost_bins')['cost'].transform("mean")
    return df