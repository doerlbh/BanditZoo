#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from banditzoo.agents import Random, TS, OGreedy, EGreedy, UCB1
from banditzoo.worlds import BernoulliMultiArmedBandits


def plot_results(results):
    plt.ion()  # turn on interactive mode
    agents, actions, metrics = results
    for i in range(len(metrics)):
        rewards = metrics[i]['reward'] 
        plt.plot(np.arange(len(rewards)), rewards, label=agents[i].name)
    plt.legend()
    plt.savefig("mab_reward_test.png")
    plt.close()
    for i in range(len(metrics)):
        rewards = metrics[i]['regret'] 
        plt.plot(np.arange(len(rewards)), rewards, label=agents[i].name)
    plt.legend()
    plt.savefig("mab_regret_test.png")


def mab():
    M = 30
    reward_means = None
    cost_means = None
    w = BernoulliMultiArmedBandits(
        M=M, reward_means=reward_means, name="MAB"
    )
    rd = Random(M=M, name="Random")
    ts = TS(M=M, name="Thompson Sampling")
    og = OGreedy(M=M, name="Optimistic Greedy")
    eg = EGreedy(M=M, name="Epsilon Greedy")
    ucb1 = UCB1(M=M, name="UCB1")
    for a in [rd, ts, og, eg, ucb1]:
        w.add_agent(a)
    w.run_experiments(T=10000)
    results = w.get_results()
    plot_results(results)


def main():
    mab()


if __name__ == "__main__":
    main()
