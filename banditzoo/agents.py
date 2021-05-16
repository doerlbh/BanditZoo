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

    def __init__(self, name=None, seed=0, **kwargs):
        self.name = name
        self.seed = seed
        np.random.seed(seed)

        self.i_t = None  # current action
        self.c_t = None  # current context
        self.t_t = 0  # current iteration

        self.reward = []  # keep track of rewards
        self.build(**kwargs)

    def build(self, **kwargs):
        pass

    def observe(self, c):
        self.c_t = c  # update context

    def update(self, rewards=None):
        self.update_agent(rewards)
        self.update_metrics(rewards)

    def act(self):
        raise NotImplementedError

    def update_agent(self, rewards=None):
        raise NotImplementedError

    def update_metrics(self, rewards=None):
        raise NotImplementedError

    def combine_rewards(self, rewards=None):
        return np.sum(rewards)


class MultiArmedAgent(Agent):
    """
    Base agent that performs the multi-armed bandit problem
    """

    def __init__(
        self,
        name="MultiArmedAgent",
        seed=0,
        **kwargs,
    ):
        Agent.__init__(self, name=name, seed=seed)
        
        self.o_arms = []  # keep track of optimal arms
        self.regret = []  # keep track of regrets
        self.build(**kwargs)
        
    def build(self, **kwargs):
        M = kwargs.get("M", None)
        
        self.M = M  # number of possible action arms, e.g. 5
        if self.M is not None:
            self.H = [0] * self.M  # the historical time certain arm is pulled
            self.Q = [0] * self.M  # the estimated action Q value

    def update_metrics(self, rewards):
        rewards = self.combine_rewards(rewards)
        self.t_t += 1
        self.Q[self.i_t] = (self.Q[self.i_t] * self.H[self.i_t] + rewards) / (
            self.H[self.i_t] + 1
        )
        self.H[self.i_t] += 1
        self.o_arms.append(np.argmax(self.Q))
        self.reward.append(rewards)
        self.regret.append(np.max(self.Q) * self.t_t - np.sum(self.reward))


class ContextualAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem
    """

    def __init__(
        self,
        name="ContextualAgent",
        seed=0,
        **kwargs,
    ):
        Agent.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)
        
    def build(self, **kwargs):
        C = kwargs.get("C", None)

        self.C = C  # dimension of the context, e.g. 100


class CombinatorialAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem
    """

    def __init__(
        self,
        name="CombinatorialAgent",
        seed=0,
        **kwargs,
    ):
        Agent.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)

    def build(self, **kwargs):
        K = kwargs.get("K", None)
        N = kwargs.get("N", None)
        
        self.K = K  # number of possible action dimensions, e.g. 4
        self.N = N  # number of possible values in each action, e.g. [4,4,5,3]

class ContextualCombinatorialAgent(CombinatorialAgent, ContextualAgent):
    """
    Base agent that performs contextual combinatorial bandit problem
    """

    def __init__(
        self,
        name="ContextualCombinatorialAgent",
        seed=0,
        **kwargs,
    ):
        CombinatorialAgent.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)

    def build(self, **kwargs):
        K = kwargs.get("K", None)
        N = kwargs.get("N", None)
        C = kwargs.get("C", None)
        CombinatorialAgent.build(self, K=K, N=N)
        ContextualAgent.build(self, C=C)

    def update_metrics(self, rewards):
        rewards = self.combine_rewards(rewards)
        self.t_t += 1
        self.reward.append(rewards)


class MultiObjectiveAgent(Agent):
    """
    Base agent that learns from multiple reward signals
    """

    def __init__(
        self,
        name="MultiObjectiveAgent",
        seed=0,
        **kwargs,
    ):
        obj_func = kwargs.get("obj_func", default_obj)
        obj_params = kwargs.get("obj_params", {})
        Agent.__init__(self, name=name, seed=seed)

        self.obj_func = obj_func  # the combined objective function
        self.obj_params = obj_params  # the params to compute objective function

    def combine_rewards(self, rewards=None):
        return self.obj_func(rewards, self.obj_params)


class Random(MultiArmedAgent):
    """
    Random agent to draw multi-armed bandits
    """

    def __init__(
        self,
        name="Random",
        seed=0,
        **kwargs,
    ):
        MultiArmedAgent.__init__(self, name=name, seed=seed, **kwargs)
    
    def build(self, **kwargs):
        MultiArmedAgent.build(self, **kwargs)

    def act(self):
        self.i_t = np.random.choice(self.M)
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class TS(MultiArmedAgent):
    """
    Thompson Sampling algorithm

    Reference: Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
    another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.
    """

    def __init__(
        self,
        name="TS",
        seed=0,
        **kwargs,
    ):
        MultiArmedAgent.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)

    def build(self, **kwargs):
        M = kwargs.get("M", None)        
        MultiArmedAgent.build(self, **kwargs)
        
        if self.M is not None:
            self.S = [1] * self.M  # success
            self.F = [1] * self.M  # failure
        
    def act(self):
        theta = []
        for i in range(self.M):
            theta.append(np.random.beta(self.S[i], self.F[i]))
        self.i_t = np.argmax(theta)
        return self.i_t

    def update_agent(self, rewards=None):
        rewards = self.combine_rewards(rewards)
        self.S[self.i_t] += rewards
        self.F[self.i_t] += 1 - rewards


class OGreedy(MultiArmedAgent):
    """
    Optimistic Greedy algorithm
    """

    def __init__(
        self,
        name="OGreedy",
        seed=0,
        **kwargs,
    ):
        MultiArmedAgent.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)

    def build(self, **kwargs):
        q_start = kwargs.get("q_start", 100)
        MultiArmedAgent.build(self, **kwargs)
        
        self.q_start = q_start

            
    def act(self):
        self.i_t = np.argmax(self.Q)
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class EGreedy(OGreedy):
    """
    Epsilon Greedy algorithm
    """

    def __init__(
        self,
        name="EGreedy",
        seed=0,
        **kwargs,
    ):
        OGreedy.__init__(self, name=name, seed=seed)
        
        self.build(**kwargs)
        
    def build(self, **kwargs):
        epsilon = kwargs.get("epsilon", 0.1)
        OGreedy.build(self, **kwargs)
        
        self.epsilon = epsilon
        
    def act(self):
        if np.random.uniform() < self.epsilon:
            self.i_t = np.random.choice(self.M)
        else:
            self.i_t = np.argmax(self.Q)
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class UCB1(OGreedy):
    """
    Upper Confidence Bound 1 (UCB1) algorithm

    Reference: Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation
    rules. Advances in applied mathematics, 6(1), 4-22.
    """

    def __init__(
        self,
        name="UCB1",
        seed=0,
        **kwargs,
    ):
        OGreedy.__init__(self, name=name, seed=seed)
        self.build(**kwargs)
        
    def build(self, **kwargs):
        OGreedy.build(self, **kwargs)
                
    def act(self):
        if self.t_t < self.M:
            self.i_t = self.t_t
        else:
            self.i_t = np.argmax(
                self.Q + np.sqrt(2 * np.log(self.t_t) / np.array(self.H))
            )
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class CCTS(ContextualCombinatorialAgent):
    """
    Contextual Combinatorial Thompson Sampling

    Reference: Lin, B., & Bouneffouf, D. (2021). Contextual Combinatorial Bandit with Budget as
    Context for Pareto Optimal Epidemic Intervention. arXiv preprint arXiv:.

    usage:
    bandit = CCTS(K=5, N=[4,3,3,4,5], C=100, alpha=0.5, nabla=0.5, name='CCTS', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(reward,cost)
    """

    def __init__(
        self,
        name="CCTS",
        seed=0,
        **kwargs,
    ):
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed)

        self.build(**kwargs)
        
    def build(self, **kwargs):
        alpha = kwargs.get("alpha", 0.1)
        nabla = kwargs.get("nabla", 0.1)
        ContextualCombinatorialAgent.build(self, **kwargs)
        
        self.alpha = alpha
        self.nabla = nabla

        if self.N is not None:
            self.B_i_k = [n * [np.eye(self.C)] for n in self.N]
            self.z_i_k = [n * [np.zeros((self.C))] for n in self.N]
            self.theta_i_k = [n * [np.zeros((self.C))] for n in self.N]
                    
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

    def update_agent(self, rewards):
        rewards = self.combine_rewards(rewards)
        for k in range(self.K):
            i = self.i_t[k]
            self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + self.c_t @ self.c_t.T
            self.z_i_k[k][i] += self.c_t * rewards
            self.theta_i_k[k][i] = np.linalg.pinv(self.B_i_k[k][i]) @ self.z_i_k[k][i]


class CCTSB(CCTS, MultiObjectiveAgent):
    """
    Contextual Combinatorial Thompson Sampling with Budget

    Reference: Lin, B., & Bouneffouf, D. (2021). Contextual Combinatorial Bandit with Budget as
    Context for Pareto Optimal Epidemic Intervention. arXiv preprint arXiv:.

    usage:
    bandit = CCTSB(K=5, N=[4,3,3,4,5], C=100, alpha=0.5, nabla=0.5, obj_func=obj_func,
        obj_params=obj_params name='CCTSB', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(reward,cost)
    """

    def __init__(
        self,
        name="CCTSB",
        seed=0,
        **kwargs,
    ):
        obj_func = kwargs.get("obj_func", default_obj)
        obj_params = kwargs.get("obj_params", {})
        MultiObjectiveAgent.__init__(
            self, name=name, seed=seed, obj_func=obj_func, obj_params=obj_params
        )
        
        self.build(**kwargs)

    def build(self, **kwargs):
        CCTS.build(self, **kwargs)
        
        
class CCMAB(ContextualCombinatorialAgent):
    """
    Independent MAB or Contextual Bandit agents to solve the contextual combinatorial bandit problem

    usage:
    bandit = CCMAB(K=5, N=[4,3,3,4,5], C=100, agent_base=UCB1, name='CCMAB-UCB1', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(reward,cost)
    """

    def __init__(
        self,
        name="CCMAB",
        seed=0,
        **kwargs,
    ):
        agent_base = kwargs.get('agent_base', UCB1)
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed)

        self.agent_base = agent_base
        self.build(**kwargs)

    def build(self, **kwargs):
        ContextualCombinatorialAgent.build(self, **kwargs)

        if self.N is not None:
            self.agents = [self.agent_base(M=n) for n in self.N]

    def act(self):
        i_t = {}
        for k in range(self.K):
            i_t[k] = self.agents[k].act()
        self.i_t = i_t
        return self.i_t

    def update_agent(self, rewards):
        rewards = self.combine_rewards(rewards)
        for k in range(self.K):
            self.agents[k].update(rewards)


class CCMABB(CCMAB, MultiObjectiveAgent):
    """
    Independent MAB or Contextual Bandit agents to solve the contextual combinatorial bandit
    problem with multiple objectives in the rewards

    usage:
    bandit = CCMABB(K=5, N=[4,3,3,4,5], C=100, agent_base=UCB1, obj_func=obj_func,
        obj_params=obj_params, name='CCMABB-UCB1', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(reward,cost)
    """

    def __init__(
        self,
        name="CCMABB",
        seed=0,
        **kwargs,
    ):
        agent_base = kwargs.get("agent_base", UCB1)
        obj_func = kwargs.get("obj_func", default_obj)
        obj_params = kwargs.get("obj_params", {})
        CCMAB.__init__(self, agent_base=agent_base, name=name, seed=seed)
        MultiObjectiveAgent.__init__(self, obj_func=obj_func, obj_params=obj_params, name=name, seed=seed)
        
        self.build(**kwargs)

    def build(self, **kwargs):
        CCMAB.build(self, **kwargs)


class CombRandom(ContextualCombinatorialAgent):
    """
    Random agent that performs combinatorial actions randomly at each round
    """

    def __init__(
        self,
        name="CombRandom",
        seed=0,
        **kwargs,
    ):
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed, **kwargs)

    def act(self):
        i_t = {}
        for k in range(self.K):
            i_t[k] = np.random.choice(self.N[k])
        self.i_t = i_t
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class CombRandomFixed(ContextualCombinatorialAgent):
    """
    Random agent that performs a set of fixed combinatorial actions
    """

    def __init__(
        self,
        name="CombRandomFixed",
        seed=0,
        **kwargs,
    ):
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed, **kwargs)

    def act(self):
        if self.i_t is None:
            i_t = {}
            for k in range(self.K):
                i_t[k] = np.random.choice(self.N[k])
            self.i_t = i_t
        return self.i_t

    def update_agent(self, rewards=None):
        pass
