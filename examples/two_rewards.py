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
            sns_plot.savefig("two_rewards_" + m + "_test.png")


def two_reward_env(n_world_instances=5, n_agent_instances=5, T=100, plot=True):
    sparse_probability = 0.1
    std_ratio = 10
    std_base = 0.1
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    for sparse_probability in [0.01, 0.1, 0.5, 1]:
        g.add_world_class(
            worlds.MultiArmedBandits,
            n_arms=5,
            reward_stds=[[[std_ratio * std_base, std_base]] * 5][0],
            reward_dimension=2,
            reward_reveal_frequency=[1, sparse_probability],
            name="MAB 5 arms (p="
            + str(sparse_probability)
            + ", r="
            + str(std_ratio)
            + ")",
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
    g.run_experiments(T=T, progress=False)
    metrics = g.get_metrics(form="tabular")
    plot_results(metrics, plot=plot)


def main():
    two_reward_env(n_world_instances=50, n_agent_instances=1, T=1000, plot=True)


def test():
    two_reward_env(n_world_instances=2, n_agent_instances=1, T=2, plot=False)


if __name__ == "__main__":
    main()
