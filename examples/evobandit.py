#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from banditzoo import agents
from banditzoo import worlds
from banditzoo import games


def plot_results(metrics, filename="evobandit", plot=True):

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
            sns_plot.savefig(filename + "_" + m + "_test.png")


def eval_ablation(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ ablation ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB of 5 arms")
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=10, name="MAB of 10 arms"
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=50, name="MAB of 50 arms"
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C+, M+)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=False,
        do_mutation=True,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C-, M+)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=True,
        do_mutation=False,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C+, M-)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=False,
        do_mutation=False,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C-, M-)",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_ablation", plot=plot)


def eval_pop(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ pop ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB of 5 arms")
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=10, name="MAB of 10 arms"
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=50, name="MAB of 50 arms"
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p100",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=25,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p25",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=10,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p10",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_pop", plot=plot)


def eval_mutation(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ mutation ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(worlds.BernoulliMultiArmedBandits, n_arms=5, name="MAB of 5 arms")
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=10, name="MAB of 10 arms"
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits, n_arms=50, name="MAB of 50 arms"
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-m10",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=50,
        mutation_max_val=1,
        name="GTS-m50",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=100,
        mutation_max_val=1,
        name="GTS-m100",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_mutation", plot=plot)


def eval_ns_ablation(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ ns ablation ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=5,
        name="Nonstationary MAB of 5 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=10,
        name="Nonstationary MAB of 10 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=50,
        name="Nonstationary MAB of 50 arms",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C+, M+)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=False,
        do_mutation=True,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C-, M+)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=True,
        do_mutation=False,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C+, M-)",
    )
    g.add_agent_class(
        agents.GTS,
        do_crossover=False,
        do_mutation=False,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS (C-, M-)",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_ns_ablation", plot=plot)


def eval_ns_pop(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ ns pop ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=5,
        name="Nonstationary MAB of 5 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=10,
        name="Nonstationary MAB of 10 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=50,
        name="Nonstationary MAB of 50 arms",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p100",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=25,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p25",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=10,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-p10",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_ns_pop", plot=plot)


def eval_ns_mutation(T=100, n_world_instances=10, n_agent_instances=10, plot=True):
    print("==============================================")
    print("============ ns mutation ================")
    g = games.Game(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=5,
        name="Nonstationary MAB of 5 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=10,
        name="Nonstationary MAB of 10 arms",
    )
    g.add_world_class(
        worlds.BernoulliMultiArmedBandits,
        change_reward_every=10,
        n_arms=50,
        name="Nonstationary MAB of 50 arms",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=10,
        mutation_max_val=1,
        name="GTS-m10",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=50,
        mutation_max_val=1,
        name="GTS-m50",
    )
    g.add_agent_class(
        agents.GTS,
        n_population=100,
        mutation_times=100,
        mutation_max_val=1,
        name="GTS-m100",
    )
    g.add_agent_class(agents.TS, name="TS")
    g.add_agent_class(agents.UCB1, name="UCB1")
    g.add_agent_class(agents.Random, name="Random")
    g.run_experiments(T=T, progress=False)
    plot_results(g.get_metrics(form="tabular"), "evobandit_ns_mutation", plot=plot)


def main():
    eval_ns_ablation(T=100, n_world_instances=10, n_agent_instances=10)
    eval_ablation(T=100, n_world_instances=10, n_agent_instances=10)
    eval_ns_pop(T=100, n_world_instances=10, n_agent_instances=10)
    eval_pop(T=100, n_world_instances=10, n_agent_instances=10)
    eval_ns_mutation(T=100, n_world_instances=10, n_agent_instances=10)
    eval_mutation(T=100, n_world_instances=10, n_agent_instances=10)


def test():
    eval_ns_ablation(T=2, n_world_instances=2, n_agent_instances=1, plot=False)
    eval_ablation(T=2, n_world_instances=2, n_agent_instances=1, plot=False)
    eval_ns_pop(T=2, n_world_instances=2, n_agent_instances=1, plot=False)
    eval_pop(T=2, n_world_instances=2, n_agent_instances=1, plot=False)
    eval_ns_mutation(T=2, n_world_instances=2, n_agent_instances=1, plot=False)
    eval_mutation(T=2, n_world_instances=2, n_agent_instances=1, plot=False)


if __name__ == "__main__":
    main()
