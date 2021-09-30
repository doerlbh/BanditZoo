#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from banditzoo import agents
from banditzoo import worlds
from banditzoo import games


def plot_results(metrics, condition):
    import seaborn as sns

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


def epidemic_setup(n_world_instances, n_agent_instances, w):
    g = games.MultiObjectiveGame(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances
    )
    g.add_world_class(
        worlds.EpidemicControl_v1,
        action_dimension=2,
        action_options=[2, 3],
        context_dimension=5,
        reward_scale=1,
        name="Epidemic (constant context)",
    )
    g.add_world_class(
        worlds.EpidemicControl_v2,
        action_dimension=2,
        action_options=[2, 3],
        context_dimension=5,
        reward_scale=1,
        name="Epidemic (context changes every 10 days)",
    )
    g.add_world_class(
        worlds.EpidemicControl_v2,
        action_dimension=2,
        action_options=[2, 3],
        context_dimension=5,
        change_every=1,
        name="Epidemic (context changes every 1 day)",
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


def epidemic_extreme(n_world_instances, n_agent_instances, T, condition, plot=True):
    g = epidemic_setup(
        n_world_instances=n_world_instances,
        n_agent_instances=n_agent_instances,
        w=condition,
    )
    g.set_params_sweep(w=[condition])
    g.run_experiments(T=T, progress=True)
    metrics = g.get_pareto_metrics()
    metrics["budget"] = metrics["cost_"]
    metrics["cases"] = np.exp(-metrics["reward_"] / metrics["reward_"].mean())
    if plot:
        plot_results(metrics, condition)


def epidemic_pareto(n_world_instances, n_agent_instances, T, plot=True):
    g = epidemic_setup(
        n_world_instances=n_world_instances, n_agent_instances=n_agent_instances, w=0.5
    )
    g.set_params_sweep(w=[0, 0.25, 0.5, 0.75, 1])
    g.run_experiments(T=T, progress=True)
    metrics = g.get_pareto_metrics()
    metrics["budget"] = metrics["cost_"]
    metrics["cases"] = np.exp(-metrics["reward_"] / metrics["reward_"].mean())
    if plot:
        plot_pareto(metrics)


def main():
    epidemic_extreme(n_world_instances=2, n_agent_instances=100, T=1000, condition=1)
    epidemic_extreme(n_world_instances=2, n_agent_instances=100, T=1000, condition=0)
    epidemic_pareto(n_world_instances=1, n_agent_instances=100, T=1000)


def test():
    epidemic_extreme(
        n_world_instances=2, n_agent_instances=2, T=2, condition=1, plot=False
    )
    epidemic_extreme(
        n_world_instances=2, n_agent_instances=2, T=2, condition=0, plot=False
    )
    epidemic_pareto(n_world_instances=1, n_agent_instances=2, T=2, plot=False)


if __name__ == "__main__":
    main()
