#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from banditzoo.agents import CCTSB, CombRandom, CombRandomFixed
from banditzoo.worlds import EpidemicControl_v1, EpidemicControl_v2


def plot_results(results):
    plt.ion()  # turn on interactive mode
    agents, actions, metrics = results
    for i in range(len(metrics)):
        rewards = np.array([m[0] for m in metrics[i]])
        plt.plot(np.arange(len(rewards)), rewards, label=agents[i].name)
    plt.legend()
    plt.savefig("test.png")


def epidemic_control_v1():
    K = 2
    N = [2, 2]
    C = 5
    reward_means = [1000, 0, 0, 1000]
    cost_means = None
    w = EpidemicControl_v1(
        K=K, N=N, C=C, reward_means=reward_means, cost_means=cost_means, name="epidemic"
    )
    cctsb = CCTSB(K=K, N=N, C=C, alpha=0.2, nabla=0.2, w=1, name="CCTSB")
    crandom = CombRandom(K=K, N=N, C=C, name="Random")
    crandomfixed = CombRandomFixed(K=K, N=N, C=C, name="Random_Fixed")
    for a in [cctsb, crandom, crandomfixed]:
        w.add_agent(a)
    w.run_experiments(T=100)
    results = w.get_results()
    plot_results(results)
    print("metrics:", results[2][0][-1], results[2][1][-1], results[2][2][-1])


def main():
    epidemic_control_v1()


if __name__ == "__main__":
    main()
