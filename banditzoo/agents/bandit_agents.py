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

        self.oracle_estimate = []
        self.recorded_estimate = []
        self.build(**kwargs)

    def build(self, **kwargs):
        n_arms = kwargs.get("n_arms", None)
        oracle = kwargs.get("reward_means", None)

        self.n_arms = n_arms  # number of possible action arms, e.g. 5
        if self.n_arms is not None:
            self.H = [0] * self.n_arms  # the historical time certain arm is pulled
            self.Q = [0] * self.n_arms  # the estimated action Q value
            if oracle is not None:
                self.oracle = oracle  # the reward function, only to compute regret
            else:
                self.oracle = [0] * self.n_arms

    def _update_metrics(self, feedbacks):
        reward = self._combine_feedbacks(feedbacks)
        self.t_t += 1
        self.Q[self.i_t] = (self.Q[self.i_t] * self.H[self.i_t] + reward) / (
            self.H[self.i_t] + 1
        )
        self.H[self.i_t] += 1
        self.reward.append(reward)
        self.oracle_estimate.append(np.max(self.Q))
        # self.oracle_estimate.append(self.Q[np.argmax(self.oracle[:,0])])
        self.recorded_estimate.append(self.Q[self.i_t])
        self.regret.append(
            np.sum(self.oracle_estimate) - np.sum(self.recorded_estimate)
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
        context_dimension = kwargs.get("context_dimension", None)

        self.context_dimension = context_dimension  # dimension of the context, e.g. 100

    def _update_metrics(self, feedbacks):
        feedbacks = self._combine_feedbacks(feedbacks)
        self.t_t += 1
        self.reward.append(feedbacks)
        self.regret.append(-1)  # placeholder


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
        action_dimension = kwargs.get("action_dimension", None)
        action_options = kwargs.get("action_options", None)

        self.action_dimension = (
            action_dimension  # number of possible action dimensions, e.g. 4
        )
        self.action_options = (
            action_options  # number of possible values in each action, e.g. [4,4,5,3]
        )


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
        action_dimension = kwargs.get("action_dimension", None)
        action_options = kwargs.get("action_options", None)
        context_dimension = kwargs.get("context_dimension", None)
        CombinatorialAgent.build(
            self, action_dimension=action_dimension, action_options=action_options
        )
        ContextualAgent.build(self, context_dimension=context_dimension)

    def _update_metrics(self, feedbacks):
        feedbacks = self._combine_feedbacks(feedbacks)
        self.t_t += 1
        self.reward.append(feedbacks)
        self.regret.append(-1)  # placeholder


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
        self.i_t = np.random.choice(self.n_arms)
        return self.i_t

    def _update_agent(self, feedbacks=None):
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
        n_arms = kwargs.get("n_arms", None)
        MultiArmedAgent.build(self, **kwargs)

        if self.n_arms is not None:
            self.S = [1] * self.n_arms  # success
            self.F = [1] * self.n_arms  # failure

    def act(self):
        theta = []
        for i in range(self.n_arms):
            theta.append(np.random.beta(self.S[i], self.F[i]))
        self.i_t = np.argmax(theta)
        return self.i_t

    def _update_agent(self, feedbacks=None):
        feedbacks = self._combine_feedbacks(feedbacks)
        self.S[self.i_t] += feedbacks
        self.F[self.i_t] += 1 - feedbacks


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

    def _update_agent(self, feedbacks=None):
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
            self.i_t = np.random.choice(self.n_arms)
        else:
            self.i_t = np.argmax(self.Q)
        return self.i_t

    def _update_agent(self, feedbacks=None):
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
        if self.t_t < self.n_arms:
            self.i_t = self.t_t
        else:
            self.i_t = np.argmax(self.Q + self._confidence())
        return self.i_t

    def _confidence(self):
        return np.sqrt(2 * np.log(self.t_t) / np.array(self.H))

    def _update_agent(self, feedbacks=None):
        pass


class IUCB(OGreedy):
    """
    Imputation Upper Confidence Bound (IUCB) algorithm.

    Reference: Bouneffouf, D. & Lin, B. (2021). Multi-Armed Bandit with Sparse and Noisy
    Feedback with application to Question Answering System. arXiv preprint arXiv.
    """

    def __init__(
        self,
        name="IUCB",
        seed=0,
        **kwargs,
    ):
        OGreedy.__init__(self, name=name, seed=seed)
        self.build(**kwargs)
        self.phi = kwargs.get("phi", 0)
        self.use_noisy = kwargs.get("use_noisy", True)
        self.use_sparse = kwargs.get("use_sparse", True)
        self.use_filter = kwargs.get("use_filter", True)
        self.sparse_probability = kwargs.get("sparse_probability", 0.1)
        self.noisy_reward = []
        self.sparse_reward = []

    def build(self, **kwargs):
        OGreedy.build(self, **kwargs)

        self.v_hat = 0
        self.v_hat_s = 0
        if self.n_arms is not None:  # initialize for sparse feedback
            self.H_s = [1] * self.n_arms
            self.Q_s = [0] * self.n_arms
            self.confidence = np.array([0] * self.n_arms)
            self.confidence_s = np.array([0] * self.n_arms)

    def _confidence(self):
        return self.confidence

    def _combine_feedbacks(self, feedbacks):
        rewards = feedbacks["rewards"]
        if len(rewards) < 2:
            rewards = rewards + [rewards[0]]
        return rewards[0], rewards[1]

    def _update_metrics(self, feedbacks):
        noisy_reward, sparse_reward = self._combine_feedbacks(feedbacks)
        noisy_reward = noisy_reward if self.use_noisy else 0
        sparse_reward = sparse_reward if self.use_sparse else 0
        self.noisy_reward.append(noisy_reward)
        self.sparse_reward.append(sparse_reward or 0)
        if self.use_noisy and self.use_filter:
            noisy_reward = np.max(
                [
                    self.Q_s[self.i_t] - self.phi - self.confidence_s[self.i_t],
                    0,
                    np.min(
                        [
                            noisy_reward,
                            self.Q_s[self.i_t] + self.phi + self.confidence_s[self.i_t],
                        ]
                    ),
                ]
            )
        reward = noisy_reward + (sparse_reward or 0)
        self.reward.append(reward)

        self.t_t += 1
        self.H[self.i_t] += 1
        self.Q[self.i_t] = (self.Q[self.i_t] * (self.H[self.i_t] - 1) + reward) / (
            self.H[self.i_t]
        )
        self.v_hat = (
            self.v_hat * (self.H[self.i_t] - 1) + (self.Q[self.i_t] - reward) ** 2
        ) / self.H[self.i_t]
        self.confidence = (
            np.sqrt(2 * self.v_hat * np.log(self.t_t) / self.H[self.i_t])
            + 3 * np.log(self.t_t) / self.H[self.i_t]
        )

        if sparse_reward is not None:
            self.Q_s[self.i_t] = (
                self.Q_s[self.i_t] * (self.H_s[self.i_t] - 1) + sparse_reward
            ) / self.H[self.i_t]
            self.v_hat_s = (
                self.v_hat_s * (self.H[self.i_t] - 1)
                + (self.Q_s[self.i_t] - sparse_reward) ** 2
            ) / self.H[self.i_t]
            self.confidence_s[self.i_t] = (
                np.sqrt(2 * self.v_hat_s * np.log(self.t_t) / self.H_s[self.i_t])
                + 3 * np.log(self.t_t) / self.H_s[self.i_t]
            )
            self.H_s[self.i_t] += 1

        self.oracle_estimate.append(np.max(self.Q))
        self.recorded_estimate.append(self.Q[self.i_t])
        self.regret.append(
            (1 + self.sparse_probability) * np.sum(self.oracle_estimate)
            - np.sum(self.recorded_estimate)
        )

    def _update_agent(self, feedbacks=None):
        pass


class CTS(ContextualAgent):
    """
    Contextual Thompson Sampling algorithm.

    Reference: Agrawal, S., & Goyal, action_options. (2013, May). Thompson sampling for contextual bandits
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
        n_arms = kwargs.get("n_arms", None)
        ContextualAgent.build(self, **kwargs)

        self.n_arms = n_arms
        if self.n_arms is not None and self.context_dimension is not None:
            self.B_i = self.n_arms * [np.eye(self.context_dimension)]
            self.z_i = self.n_arms * [np.zeros((self.context_dimension))]
            self.theta_i = self.n_arms * [np.zeros((self.context_dimension))]

    def act(self):
        sample_theta = self.n_arms * [0]
        for i in range(self.n_arms):
            sample_theta[i] = np.random.multivariate_normal(
                self.theta_i[i],
                self.alpha ** 2 * np.linalg.pinv(self.B_i[i]),
            )
        self.i_t = np.argmax((self.c_t.T @ np.array(sample_theta).T))
        return self.i_t

    def _update_agent(self, feedbacks):
        feedbacks = self._combine_feedbacks(feedbacks)
        i = self.i_t
        self.B_i[i] = self.nabla * self.B_i[i] + self.c_t @ self.c_t.T
        self.z_i[i] += self.c_t * feedbacks
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
        n_arms = kwargs.get("n_arms", None)
        ContextualAgent.build(self, **kwargs)

        self.n_arms = n_arms
        if self.n_arms is not None and self.context_dimension is not None:
            self.A_i = self.n_arms * [np.eye(self.context_dimension)]
            self.b_i = self.n_arms * [np.zeros((self.context_dimension))]

    def act(self):
        p_t = self.n_arms * [0]
        for i in range(self.n_arms):
            theta_i = np.linalg.pinv(self.A_i[i]) @ self.b_i[i]
            p_t[i] = theta_i.T @ self.c_t + self.alpha * np.sqrt(
                self.c_t.T @ np.linalg.pinv(self.A_i[i]) @ self.c_t
            )
        self.i_t = np.argmax(p_t)
        return self.i_t

    def _update_agent(self, feedbacks):
        feedbacks = self._combine_feedbacks(feedbacks)
        i = self.i_t
        self.A_i[i] = self.nabla * self.A_i[i] + self.c_t @ self.c_t.T
        self.b_i[i] += self.c_t * feedbacks


# TODO class BerlinUCB(LinUCB):


class CCTS(ContextualCombinatorialAgent):
    """
    Contextual Combinatorial Thompson Sampling.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

    usage:
    bandit = CCTS(action_dimension=5, action_options=[4,3,3,4,5], context_dimension=100, alpha=0.5, nabla=0.5, name='CCTS', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(feedbacks)
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

        if self.action_options is not None:
            self.B_i_k = [
                n * [np.eye(self.context_dimension)] for n in self.action_options
            ]
            self.z_i_k = [
                n * [np.zeros((self.context_dimension))] for n in self.action_options
            ]
            self.theta_i_k = [
                n * [np.zeros((self.context_dimension))] for n in self.action_options
            ]

    def act(self):
        sample_theta = [n * [0] for n in self.action_options]
        i_t = {}
        for k in range(self.action_dimension):

            for i in range(len(sample_theta[k])):
                sample_theta[k][i] = np.random.multivariate_normal(
                    self.theta_i_k[k][i],
                    self.alpha ** 2 * np.linalg.pinv(self.B_i_k[k][i]),
                )

            i_t[k] = np.argmax((self.c_t.T @ np.array(sample_theta[k]).T))

        self.i_t = i_t
        return self.i_t

    def _update_agent(self, feedbacks):
        feedbacks = self._combine_feedbacks(feedbacks)
        for k in range(self.action_dimension):
            i = self.i_t[k]
            self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + self.c_t @ self.c_t.T
            self.z_i_k[k][i] += self.c_t * feedbacks
            self.theta_i_k[k][i] = np.linalg.pinv(self.B_i_k[k][i]) @ self.z_i_k[k][i]


class CCTSB(CCTS):
    """
    Contextual Combinatorial Thompson Sampling with Budget.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

    usage:
    bandit = CCTSB(action_dimension=5, action_options=[4,3,3,4,5], context_dimension=100, alpha=0.5, nabla=0.5, obj_func=obj_func,
        obj_params=obj_params name='CCTSB', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(feedbacks)
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
        CCTS.__init__(
            self,
            name=name,
            seed=seed,
            alpha=alpha,
            nabla=nabla,
            obj_func=obj_func,
            obj_params=obj_params,
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
    bandit = CCMAB(action_dimension=5, action_options=[4,3,3,4,5], context_dimension=100, agent_base=UCB1, name='CCMAB-UCB1', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(feedbacks)
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

        if self.action_options is not None:
            self.agents = [
                self.agent_base(n_arms=n, context_dimension=self.context_dimension)
                for n in self.action_options
            ]

    def observe(self, c):
        self.c_t = c  # update context
        for k in range(self.action_dimension):
            self.agents[k].observe(self.c_t)

    def act(self):
        i_t = {}
        for k in range(self.action_dimension):
            i_t[k] = self.agents[k].act()
        self.i_t = i_t
        return self.i_t

    def _update_agent(self, feedbacks):
        for k in range(self.action_dimension):
            self.agents[k].update(feedbacks)


class CCMABB(CCMAB):
    """
    Independent MAB or Contextual Bandit agents to solve the contextual combinatorial bandit
    problem with multiple objectives in the feedbacks.

    Reference: Lin, B., & Bouneffouf, D. (2021). Optimal Epidemic Control as a Contextual
    Combinatorial Bandit with Budget. arXiv preprint arXiv:2106.15808.

    usage:
    bandit = CCMABB(action_dimension=5, action_options=[4,3,3,4,5], context_dimension=100, agent_base=UCB1, obj_func=obj_func,
        obj_params=obj_params, name='CCMABB-UCB1', seed=0)
    bandit.observe(context)
    actions = bandit.act()
    bandit.update(feedbacks)
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
        CCMAB.__init__(
            self,
            agent_base=agent_base,
            obj_func=obj_func,
            obj_params=obj_params,
            name=name,
            seed=seed,
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
        for k in range(self.action_dimension):
            i_t[k] = np.random.choice(self.action_options[k])
        self.i_t = i_t
        return self.i_t

    def _update_agent(self, feedbacks=None):
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
            for k in range(self.action_dimension):
                i_t[k] = np.random.choice(self.action_options[k])
            self.i_t = i_t
        return self.i_t

    def _update_agent(self, feedbacks=None):
        pass
