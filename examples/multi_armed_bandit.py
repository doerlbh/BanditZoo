#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from banditzoo import agents
from banditzoo import worlds
from banditzoo import games


def plot_results(metrics):
    import seaborn as sns

    metric_names = ["reward", "regret"]
    for m in metric_names:
        sns_plot = sns.relplot(
            data=metrics, x="time", y=m, hue="agent", col="world", kind="line", ci=68
        )
        sns_plot.savefig("mab_" + m + "_test.png")


def main():
    g = games.Game(n_world_instances=3, n_agent_instances=10)
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB5")
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=3, name="MAB3")
    g.add_agent_class(agents.TS, name="Thompson Sampling")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.EGreedy, epsilon=0.05, name="Epsilon Greedy (e=0.05)")
    g.add_agent_class(agents.EGreedy, epsilon=0.1, name="Epsilon Greedy (e=0.1)")
    g.add_agent_class(agents.EGreedy, epsilon=0.2, name="Epsilon Greedy (e=0.2)")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=100, progress=True)
    metrics = g.get_metrics(form="tabular")
    plot_results(metrics)


def test():
    g = games.Game(n_world_instances=2, n_agent_instances=2)
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB5")
    g.add_agent_class(agents.EGreedy, epsilon=0.05, name="Epsilon Greedy (e=0.05)")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=2, progress=True)
    metrics = g.get_metrics(form="tabular")


if __name__ == "__main__":
    main()
