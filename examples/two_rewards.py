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
        sns_plot.savefig("two_rewards_" + m + "_test.png")


def main():
    sparse_probability = 0.05
    g = games.Game(n_world_instances=10, n_agent_instances=10)
    g.add_world_class(
        worlds.MultiArmedBandits,
        n_arms=3,
        reward_stds=[[[5, 1]] * 3][0],
        reward_dimension=2,
        reward_reveal_frequency=[1, sparse_probability],
        name="MAB 3 arms (p=" + str(sparse_probability) + ")",
    )
    g.add_world_class(
        worlds.MultiArmedBandits,
        n_arms=10,
        reward_stds=[[[5, 1]] * 10][0],
        reward_dimension=2,
        reward_reveal_frequency=[1, sparse_probability],
        name="MAB 10 arms (p=" + str(sparse_probability) + ")",
    )
    g.add_agent_class(agents.IUCB, sparse_probability=sparse_probability, name="IUCB")
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_filter=False,
        name="ID-UCB",
    )
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_noisy=False,
        name="D-UCB",
    )
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_sparse=False,
        name="N-UCB",
    )
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=100, progress=True)
    metrics = g.get_metrics(form="tabular")
    plot_results(metrics)


def test():
    sparse_probability = 0.1
    g = games.Game(n_world_instances=2, n_agent_instances=2)
    g.add_world_class(
        worlds.MultiArmedBandits,
        n_arms=3,
        reward_stds=[[5] * 3, [1] * 3],
        reward_dimension=2,
        reward_reveal_frequency=[1, sparse_probability],
        name="MAB3",
    )
    g.add_agent_class(agents.IUCB, sparse_probability=sparse_probability, name="IUCB")
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_filter=False,
        name="ID-UCB",
    )
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_noisy=False,
        name="D-UCB",
    )
    g.add_agent_class(
        agents.IUCB,
        sparse_probability=sparse_probability,
        use_sparse=False,
        name="N-UCB",
    )
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=3, progress=True)
    metrics = g.get_metrics(form="tabular")


if __name__ == "__main__":
    main()
