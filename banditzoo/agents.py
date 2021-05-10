#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes of Bandit/RL Agents

usage:
a = Agent()
...
a.observe(context)
actions = a.act()
...
a.update(reward,cost)
"""

import numpy as np
from .utils import default_obj


class Agent(object):
    """
    Base reinforcement learning agent class
    """

    def __init__(self, name=None, seed=0):
        self.name = name
        self.seed = seed
        np.random.seed(seed)

    def observe(self, c):
        self.c_t = c

    def act(self):
        raise NotImplementedError

    def update(self, rewards=None):
        raise NotImplementedError


class MultiArmedAgent(Agent):
    """
    Base agent that performs the multi-armed bandit problem
    """

    def __init__(self, M=None, name=None, seed=0):
        Agent.__init__(self, name=name, seed=seed)
        self.M = M  # number of possible action arms, e.g. 4
        self.i_t = None  # current actions


class ContextualAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem
    """

    def __init__(self, C=None, name=None, seed=0):
        Agent.__init__(self, name=name, seed=seed)

        self.C = C  # dimension of the context, e.g. 100
        self.c_t = None  # current context
        self.i_t = None  # current actions


class CombinatorialAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem
    """

    def __init__(self, K=None, N=None, name=None, seed=0):
        Agent.__init__(self, name=name, seed=seed)

        self.K = K  # number of possible action dimensions, e.g. 4
        self.N = N  # number of possible values in each action, e.g. [4,4,5,3]


class ContextualCombinatorialAgent(CombinatorialAgent, ContextualAgent):
    """
    Base agent that performs contextual combinatorial bandit problem
    """

    def __init__(
        self,
        K=None,
        N=None,
        C=None,
        name=None,
        seed=0,
    ):
        CombinatorialAgent.__init__(self, K=K, N=N, name=name, seed=seed)
        ContextualAgent.__init__(self, C=C, name=name, seed=seed)


class TS(MultiArmedAgent):
    """
    Thompson Sampling algorithm

    Reference: Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
    another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.
    """

    def __init__(self, M=None, name=None, seed=0):
        MultiArmedAgent.__init__(self, M=M, name=name, seed=seed)

        if self.M is not None:
            self.S = [1] * self.M  # success
            self.F = [1] * self.M  # failure

    def act(self):
        theta = []
        for i in range(self.M):
            theta.append(np.random.beta(self.S[i], self.F[i]))
        self.i_t = np.argmax(theta)
        return self.i_t

    def update(self, rewards=None):
        self.S[self.i_t] += rewards
        self.F[self.i_t] += 1 - rewards


class OGreedy(MultiArmedAgent):
    """
    Optimistic Greedy algorithm
    """

    def __init__(self, M=None, q_start=100, name=None, seed=0):
        MultiArmedAgent.__init__(self, M=M, name=name, seed=seed)

        self.q_start = q_start
        if self.M is not None:
            self.H = [0] * self.M  # the historical time certain arm is pulled
            self.Q = [q_start] * self.M  # the estimated action Q value

    def act(self):
        self.i_t = np.argmax(self.Q)
        return self.i_t

    def update(self, rewards=None):
        self.Q[self.i_t] = (self.Q[self.i_t] * self.H[self.i_t] + rewards) / (
            self.H[self.i_t] + 1
        )
        self.H[self.i_t] += 1


class EGreedy(OGreedy):
    """
    Epsilon Greedy algorithm
    """

    def __init__(self, M=None, epsilon=0.1, q_start=100, name=None, seed=0):
        OGreedy.__init__(self, M=M, q_start=q_start, name=name, seed=seed)

        self.epsilon = epsilon

    def act(self):
        if np.random.uniform() < self.epsilon:
            self.i_t = np.random.choice(self.M)
        else:
            self.i_t = np.argmax(self.Q)
        return self.i_t


class CCTSB(ContextualCombinatorialAgent):
    """
    Contextual Combinatorial Thompson Sampling with Budget

    Reference: Lin, B., & Bouneffouf, D. (2021). Contextual Combinatorial Bandit with Budget as
    Context for Pareto Optimal Epidemic Intervention. arXiv preprint arXiv:.

    usage:
    bandit = CCTSB(K=5, N=[4,3,3,4,5], C=100, alpha=0.5, nabla=0.5, w=0.5, obj_func=default_obj, seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(reward,cost)
    """

    def __init__(
        self,
        K=None,
        N=None,
        C=None,
        alpha=0.1,
        nabla=0.1,
        w=0.5,
        obj_func=default_obj,
        name=None,
        seed=0,
    ):
        ContextualCombinatorialAgent.__init__(self, K=K, N=N, C=C, name=name, seed=seed)
        self.alpha = alpha
        self.nabla = nabla
        self.w = w
        self.obj_func = obj_func

        if self.N is not None:
            self.B_i_k = [n * [np.eye(C)] for n in self.N]
            self.z_i_k = [n * [np.zeros((C))] for n in self.N]
            self.theta_i_k = [n * [np.zeros((C))] for n in self.N]

    def act(self):
        sample_theta = [n * [0] for n in self.N]
        i_t = {}
        for k in range(self.K):

            for i in range(len(sample_theta[k])):
                sample_theta[k][i] = np.random.multivariate_normal(
                    self.theta_i_k[k][i],
                    self.alpha ** 2 * np.linalg.pinv(self.B_i_k[k][i]),
                )

            i_t[k] = np.argmax((self.c_t.T @ np.array(sample_theta[k]).T))

        self.i_t = i_t
        return self.i_t

    def update(self, rewards):
        r_star = self.obj_func(rewards, self.w)
        for k in range(self.K):
            i = self.i_t[k]
            self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + self.c_t @ self.c_t.T
            self.z_i_k[k][i] += self.c_t * r_star
            self.theta_i_k[k][i] = np.linalg.pinv(self.B_i_k[k][i]) @ self.z_i_k[k][i]


class CRandom(ContextualCombinatorialAgent):
    """
    Random agent that performs combinatorial actions randomly at each round
    """

    def act(self):
        i_t = {}
        for k in range(self.K):
            i_t[k] = np.random.choice(self.N[k])
        self.i_t = i_t
        return self.i_t

    def update(self, rewards=None):
        pass


class CRandomFixed(ContextualCombinatorialAgent):
    """
    Random agent that performs a set of fixed combinatorial actions
    """

    def act(self):
        if self.i_t is None:
            i_t = {}
            for k in range(self.K):
                i_t[k] = np.random.choice(self.N[k])
            self.i_t = i_t
        return self.i_t

    def update(self, rewards=None):
        pass
