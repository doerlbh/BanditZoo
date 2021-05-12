#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from banditzoo.agents import UCB1, EGreedy, CCTSB, CCMAB, CombRandom, CombRandomFixed
from banditzoo.worlds import EpidemicControl_v1, EpidemicControl_v2


def plot_results(results, N):
    plt.ion()  # turn on interactive mode
    agents, actions, metrics = results
    for i in range(len(metrics)):
        rewards = metrics[i]["reward"]
        plt.plot(np.arange(len(rewards)), rewards, label=agents[i].name)
    plt.legend()
    plt.savefig("epi_reward_test.png")
    plt.close()
    for i in range(len(metrics)):
        rewards = metrics[i]["cost"]
        plt.plot(np.arange(len(rewards)), rewards, label=agents[i].name)
    plt.legend()
    plt.savefig("epi_cost_test.png")
    for d, n in enumerate(N):
        plt.close()
        for i in range(len(metrics)):
            a = [np.int(a[d]) for a in actions[i]]
            acts = np.eye(n)[a]
            j = 0
            plt.plot(
                np.arange(len(acts[:, j])), np.cumsum(acts[:, j]), label=agents[i].name
            )
        plt.legend()
        plt.savefig("epi_action_" + str(d) + "test.png")
    # for d, n in enumerate(N):
    #     for i in range(len(metrics)):
    #         plt.close()
    #         a = [np.int(a[d]) for a in actions[i]]
    #         a = np.eye(n)[a]
    #         plt.imshow(a)
    #         plt.savefig("epi_"+str(d)+"_"+agents[i].name+"_test.png")


def epidemic_control_v1():
    K = 1
    N = [5]
    C = 9
    reward_means = [1000, 10, 0, 10, 20]
    # cost_means = [0,0,0,0,0]
    # reward_means = None
    cost_means = None
    w = EpidemicControl_v1(
        K=K, N=N, C=C, reward_means=reward_means, cost_means=cost_means, name="epidemic"
    )
    weight = 1
    agents = []
    agents.append(
        CCTSB(K=K, N=N, C=C, alpha=0.2, nabla=0.2, w=weight, seed=0, name="CCTSB")
    )
    agents.append(
        CCTSB(K=K, N=N, C=C, alpha=0.2, nabla=0.2, w=weight, seed=1, name="CCTSB")
    )
    agents.append(
        CCTSB(K=K, N=N, C=C, alpha=0.2, nabla=0.2, w=weight, seed=2, name="CCTSB")
    )
    agents.append(
        CCTSB(K=K, N=N, C=C, alpha=0.2, nabla=0.2, w=weight, seed=3, name="CCTSB")
    )
    agents.append(CCMAB(K=K, N=N, C=C, agent_base=UCB1, w=weight, name="CCMAB_UCB1"))
    agents.append(
        CCMAB(K=K, N=N, C=C, agent_base=EGreedy, w=weight, name="CCMAB_EGreedy")
    )
    agents.append(CombRandom(K=K, N=N, C=C, name="Random"))
    agents.append(CombRandomFixed(K=K, N=N, C=C, name="Random_Fixed"))
    for a in agents:
        w.add_agent(a)
    w.run_experiments(T=1000, progress=True)
    results = w.get_results()
    plot_results(results, N)


def main():
    epidemic_control_v1()


if __name__ == "__main__":
    main()
