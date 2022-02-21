#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from banditzoo import agents
from banditzoo import worlds
from banditzoo import games


def plot_results(metrics, plot=True):

    metric_names = ["reward", "regret"]
    print("---- mean ----")
    print(
        metrics[metrics["time"] == np.max(metrics["time"])][
            ["world", "agent"] + metric_names
        ]
        .groupby(["world", "agent"])
        .mean()
    )
    print("---- standard err ----")
    print(
        metrics[metrics["time"] == np.max(metrics["time"])][
            ["world", "agent"] + metric_names
        ]
        .groupby(["world", "agent"])
        .sem()
    )

    if plot:
        import seaborn as sns

        for m in metric_names:
            sns_plot = sns.relplot(
                data=metrics,
                x="time",
                y=m,
                hue="agent",
                col="world",
                kind="line",
                ci=68,
            )
            sns_plot.savefig("mab_" + m + "_test.png")


def mab_env(n_world_instances=10, n_agent_instances=10, T=100, plot=True):
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB5")
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=3, name="MAB3")
    g.add_agent_class(agents.TS, name="Thompson Sampling")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.EGreedy, epsilon=0.05, name="Epsilon Greedy (e=0.05)")
    g.add_agent_class(agents.EGreedy, epsilon=0.1, name="Epsilon Greedy (e=0.1)")
    g.add_agent_class(agents.EGreedy, epsilon=0.2, name="Epsilon Greedy (e=0.2)")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    metrics = g.get_metrics(form="tabular")
    plot_results(metrics, plot)


def main():
    mab_env(n_world_instances=10, n_agent_instances=10, T=100, plot=True)


def test():
    mab_env(n_world_instances=2, n_agent_instances=1, T=2, plot=False)


if __name__ == "__main__":
    main()
