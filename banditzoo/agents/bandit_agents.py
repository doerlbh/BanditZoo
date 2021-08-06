# Copyright 2021 Baihan Lin
#
# Licensed under the GNU General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes of bandit agents.
"""

import numpy as np
from .base_agents import Agent
from .utils import default_obj


class MultiArmedAgent(Agent):
    """
    Base agent that performs the multi-armed bandit problem.
    """

    def __init__(
        self,
        name="MultiArmedAgent",
        seed=0,
        **kwargs,
    ):
        Agent.__init__(self, name=name, seed=seed)

        self.regret = []  # keep track of regrets
        self.build(**kwargs)

    def build(self, **kwargs):
        M = kwargs.get("M", None)
        oracle = kwargs.get("oracle", None)

        self.M = M  # number of possible action arms, e.g. 5
        if self.M is not None:
            self.H = [0] * self.M  # the historical time certain arm is pulled
            self.Q = [0] * self.M  # the estimated action Q value
            if oracle is not None:
                self.oracle = oracle  # the reward function, only to compute regret
            else:
                self.oracle = [0] * self.M

    def update_metrics(self, rewards):
        rewards = self.combine_rewards(rewards)
        self.t_t += 1
        self.Q[self.i_t] = (self.Q[self.i_t] * self.H[self.i_t] + rewards) / (
            self.H[self.i_t] + 1
        )
        self.H[self.i_t] += 1
        self.reward.append(rewards)
        self.regret.append(
            (np.max(self.oracle) * self.t_t - np.sum(self.reward)) / self.t_t
        )


class ContextualAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem.
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

    def update_metrics(self, rewards):
        rewards = self.combine_rewards(rewards)
        self.t_t += 1
        # self.Q[self.i_t] = (self.Q[self.i_t] * self.H[self.i_t] + rewards) / (
        #     self.H[self.i_t] + 1
        # )
        # self.H[self.i_t] += 1
        self.reward.append(rewards)
        # self.regret.append((np.max(self.oracle) * self.t_t - np.sum(self.reward)) / self.t_t )


class CombinatorialAgent(Agent):
    """
    Base agent that performs the combinatorial bandit problem.
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
    Base agent that performs contextual combinatorial bandit problem.
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
    Base agent that learns from multiple reward signals.
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
    Random agent to draw multi-armed bandits.
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
    Thompson Sampling algorithm.

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
    Optimistic Greedy algorithm.
    """

    def __init__(
        self,
        name="OGreedy",
        seed=0,
        **kwargs,
    ):
        q_start = kwargs.get("q_start", 100)
        MultiArmedAgent.__init__(self, name=name, seed=seed)

        self.q_start = q_start
        self.build(**kwargs)

    def build(self, **kwargs):
        MultiArmedAgent.build(self, **kwargs)

    def act(self):
        self.i_t = np.argmax(self.Q)
        return self.i_t

    def update_agent(self, rewards=None):
        pass


class EGreedy(OGreedy):
    """
    Epsilon Greedy algorithm.
    """

    def __init__(
        self,
        name="EGreedy",
        seed=0,
        **kwargs,
    ):
        epsilon = kwargs.get("epsilon", 0.1)
        OGreedy.__init__(self, name=name, seed=seed)

        self.epsilon = epsilon
        self.build(**kwargs)

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
    Upper Confidence Bound 1 (UCB1) algorithm.

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


class CTS(ContextualAgent):
    """
    Contextual Thompson Sampling algorithm.

    Reference: Agrawal, S., & Goyal, N. (2013, May). Thompson sampling for contextual bandits
    with linear payoffs. In International Conference on Machine Learning (pp. 127-135). PMLR.
    """

    def __init__(
        self,
        name="CTS",
        seed=0,
        **kwargs,
    ):
        alpha = kwargs.get("alpha", 0.1)
        nabla = kwargs.get("nabla", 1.0)
        ContextualAgent.__init__(self, name=name, seed=seed)

        self.alpha = alpha
        self.nabla = nabla
        self.build(**kwargs)

    def build(self, **kwargs):
        M = kwargs.get("M", None)
        ContextualAgent.build(self, **kwargs)

        self.M = M
        if self.M is not None and self.C is not None:
            self.B_i = self.M * [np.eye(self.C)]
            self.z_i = self.M * [np.zeros((self.C))]
            self.theta_i = self.M * [np.zeros((self.C))]

    def act(self):
        sample_theta = self.M * [0]
        for i in range(self.M):
            sample_theta[i] = np.random.multivariate_normal(
                self.theta_i[i],
                self.alpha ** 2 * np.linalg.pinv(self.B_i[i]),
            )
        self.i_t = np.argmax((self.c_t.T @ np.array(sample_theta).T))
        return self.i_t

    def update_agent(self, rewards):
        rewards = self.combine_rewards(rewards)
        i = self.i_t
        self.B_i[i] = self.nabla * self.B_i[i] + self.c_t @ self.c_t.T
        self.z_i[i] += self.c_t * rewards
        self.theta_i[i] = np.linalg.pinv(self.B_i[i]) @ self.z_i[i]


class LinUCB(ContextualAgent):
    """
    Linear Upper Confidence Bound (LinUCB) algorithm.

    Reference: Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011, June). Contextual bandits
    with linear payoff functions. In Proceedings of the Fourteenth International Conference
    on Artificial Intelligence and Statistics (pp. 208-214). JMLR Workshop and Conference
    Proceedings.
    """

    def __init__(
        self,
        name="LinUCB",
        seed=0,
        **kwargs,
    ):
        alpha = kwargs.get("alpha", 0.1)
        nabla = kwargs.get("nabla", 1.0)
        ContextualAgent.__init__(self, name=name, seed=seed)

        self.alpha = alpha
        self.nabla = nabla
        self.build(**kwargs)

    def build(self, **kwargs):
        M = kwargs.get("M", None)
        ContextualAgent.build(self, **kwargs)

        self.M = M
        if self.M is not None and self.C is not None:
            self.A_i = self.M * [np.eye(self.C)]
            self.b_i = self.M * [np.zeros((self.C))]

    def act(self):
        p_t = self.M * [0]
        for i in range(self.M):
            theta_i = np.linalg.pinv(self.A_i[i]) @ self.b_i[i]
            p_t[i] = theta_i.T @ self.c_t + self.alpha * np.sqrt(
                self.c_t.T @ np.linalg.pinv(self.A_i[i]) @ self.c_t
            )
        self.i_t = np.argmax(p_t)
        return self.i_t

    def update_agent(self, rewards):
        rewards = self.combine_rewards(rewards)
        i = self.i_t
        self.A_i[i] = self.nabla * self.A_i[i] + self.c_t @ self.c_t.T
        self.b_i[i] += self.c_t * rewards


# TODO class BerlinUCB(LinUCB):




class CCTS(ContextualCombinatorialAgent):
    """
    Contextual Combinatorial Thompson Sampling.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

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
        alpha = kwargs.get("alpha", 0.1)
        nabla = kwargs.get("nabla", 1.0)
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed)

        self.alpha = alpha
        self.nabla = nabla
        self.build(**kwargs)

    def build(self, **kwargs):
        ContextualCombinatorialAgent.build(self, **kwargs)

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
    Contextual Combinatorial Thompson Sampling with Budget.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

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
        alpha = kwargs.get("alpha", 0.1)
        nabla = kwargs.get("nabla", 1.0)
        obj_func = kwargs.get("obj_func", default_obj)
        obj_params = kwargs.get("obj_params", {})
        CCTS.__init__(self, name=name, seed=seed, alpha=alpha, nabla=nabla)
        MultiObjectiveAgent.__init__(
            self, name=name, seed=seed, obj_func=obj_func, obj_params=obj_params
        )

        self.build(**kwargs)

    def build(self, **kwargs):
        CCTS.build(self, **kwargs)


class CCMAB(ContextualCombinatorialAgent):
    """
    Independent MAB or Contextual Bandit agents to solve the contextual combinatorial bandit problem.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

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
        agent_base = kwargs.get("agent_base", UCB1)
        ContextualCombinatorialAgent.__init__(self, name=name, seed=seed)

        self.agent_base = agent_base
        self.build(**kwargs)

    def build(self, **kwargs):
        ContextualCombinatorialAgent.build(self, **kwargs)

        if self.N is not None:
            self.agents = [self.agent_base(M=n, C=self.C) for n in self.N]

    def observe(self, c):
        self.c_t = c  # update context
        for k in range(self.K):
            self.agents[k].observe(self.c_t)

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
    problem with multiple objectives in the rewards.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

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
        MultiObjectiveAgent.__init__(
            self, obj_func=obj_func, obj_params=obj_params, name=name, seed=seed
        )

        self.build(**kwargs)

    def build(self, **kwargs):
        CCMAB.build(self, **kwargs)


class CombRandom(ContextualCombinatorialAgent):
    """
    Random agent that performs combinatorial actions randomly at each round.
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
    Random agent that performs a set of fixed combinatorial actions.
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
