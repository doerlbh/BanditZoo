#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
from banditzoo import agents
from banditzoo import worlds
from banditzoo import games


def plot_results(metrics, condition):
    metric_names = ["reward", "cost", "budget", "cases"]
    for m in metric_names:
        sns_plot = sns.relplot(
            data=metrics, x="time", y=m, hue="agent", col="world", kind="line", ci=68
        )
        sns_plot.savefig("epidemic_" + str(condition) + "_" + m + "_test.png")


def plot_pareto(metrics):
    sns_plot = sns.relplot(
        data=metrics,
        x="budget",
        y="cases",
        hue="agent",
        col="world",
        kind="line",
        ci=68,
    )
    sns_plot.savefig("epidemic_pareto_test.png")


def epidemic_setup(N, M, w):
    g = games.MultiObjectiveGame(N=N, M=M)
    g.add_world_class(
        worlds.EpidemicControl_v1,
        K=2,
        N=[2, 3],
        C=5,
        reward_scale=1,
        name="Epidemic Simulation (constant)",
    )
    g.add_world_class(
        worlds.EpidemicControl_v2,
        K=2,
        N=[2, 3],
        C=5,
        reward_scale=1,
        name="Epidemic Simulation (change every 10 days)",
    )
    g.add_world_class(
        worlds.EpidemicControl_v2,
        K=2,
        N=[2, 3],
        C=5,
        change_every=1,
        name="Epidemic Simulation (change every 1 day)",
    )
    g.add_agent_class(
        agents.CCTSB,
        alpha=0.01,
        nabla=1,
        obj_func=agents.utils.budget_obj_v2,
        obj_params={"w": w},
        name="CCTSB",
    )
    g.add_agent_class(
        agents.CCMABB,
        agent_base=agents.UCB1,
        obj_func=agents.utils.budget_obj_v2,
        obj_params={"w": w},
        name="IndComb-UCB1",
    )
    g.add_agent_class(agents.CombRandom, name="Random")
    g.add_agent_class(agents.CombRandomFixed, name="RandomFixed")
    return g


def epidemic_extreme(N, M, T, condition):
    g = epidemic_setup(N=N, M=M, w=condition)
    g.set_params_sweep(w=[condition])
    g.run_experiments(T=T, progress=True)
    metrics = g.get_pareto_metrics()
    metrics["budget"] = metrics["cost_"]
    metrics["cases"] = np.exp(-metrics["reward_"] / metrics["reward_"].mean())
    plot_results(metrics, condition)


def epidemic_pareto(N, M, T):
    g = epidemic_setup(N=N, M=M, w=0.5)
    g.set_params_sweep(w=[0, 0.25, 0.5, 0.75, 1])
    g.run_experiments(T=T, progress=True)
    metrics = g.get_pareto_metrics()
    metrics["budget"] = metrics["cost_"]
    metrics["cases"] = np.exp(-metrics["reward_"] / metrics["reward_"].mean())
    plot_pareto(metrics)


def main():
    epidemic_extreme(N=2, M=100, T=1000, condition=1)
    epidemic_extreme(N=2, M=100, T=1000, condition=0)
    epidemic_pareto(N=1, M=100, T=1000)


if __name__ == "__main__":
    main()
