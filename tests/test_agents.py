"""Test for the agents"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import agents


@parameterized_class(
    [
        {"agent": agents.Agent},
        {"agent": agents.MultiArmedAgent},
        {"agent": agents.ContextualAgent},
        {"agent": agents.CombinatorialAgent},
        {"agent": agents.MultiObjectiveAgent},
        {"agent": agents.Random},
        {"agent": agents.TS},
        {"agent": agents.OGreedy},
        {"agent": agents.EGreedy},
        {"agent": agents.UCB1},
        {"agent": agents.IUCB},
        {"agent": agents.GTS},
        {"agent": agents.CTS},
        {"agent": agents.LinUCB},
        {"agent": agents.CCTS},
        {"agent": agents.CCTSB},
        {"agent": agents.CCMAB},
        {"agent": agents.CCMABB},
        {"agent": agents.CombRandom},
        {"agent": agents.CombRandomFixed},
    ]
)
class TestAllAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent()


@parameterized_class(
    [
        {"agent": agents.Random},
        {"agent": agents.TS},
        {"agent": agents.OGreedy},
        {"agent": agents.EGreedy},
        {"agent": agents.UCB1},
        {"agent": agents.IUCB},
        {"agent": agents.GTS},
    ]
)
class TestMultiArmedAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent(n_arms=5)

    def test_the_agent_can_act(self):
        a = self.agent(n_arms=5)
        action = a.act()

    def test_the_agent_can_update(self):
        a = self.agent(n_arms=5)
        action = a.act()
        feedbacks = {"rewards": [1]}
        a.update(feedbacks)


@parameterized_class(
    [
        {"agent": agents.CTS},
        {"agent": agents.LinUCB},
    ]
)
class TestContextualAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent(n_arms=3, context_dimension=10)

    def test_the_agent_can_act(self):
        a = self.agent(n_arms=3, context_dimension=10)
        context = np.arange(10)
        a.observe(context)
        action = a.act()

    def test_the_agent_can_update(self):
        a = self.agent(n_arms=3, context_dimension=10)
        context = np.arange(10)
        a.observe(context)
        action = a.act()
        feedbacks = {"rewards": [10]}
        a.update(feedbacks)


@parameterized_class(
    [
        {"agent": agents.CCTS},
        {"agent": agents.CCTSB},
        {"agent": agents.CCMAB},
        {"agent": agents.CCMABB},
        {"agent": agents.CombRandom},
        {"agent": agents.CombRandomFixed},
    ]
)
class TestContextualCombinatorialAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent(action_dimension=2, action_options=[3, 4], context_dimension=12)

    def test_the_agent_can_act(self):
        a = self.agent(action_dimension=2, action_options=[3, 4], context_dimension=12)
        context = np.arange(12)
        a.observe(context)
        action = a.act()

    def test_the_agent_can_update(self):
        a = self.agent(action_dimension=2, action_options=[3, 4], context_dimension=12)
        context = np.arange(12)
        a.observe(context)
        action = a.act()
        feedbacks = {"rewards": [10]}
        a.update(feedbacks)


class TestUtils(TestCase):
    def test_increment_std_works(self):
        x, mu_tm1, s_tm1, n = 10, 5, 5, 3
        new_std = agents.utils.increment_std(x, mu_tm1, s_tm1, n)
        expected_r = np.sqrt(s_tm1**2 * (n - 2) / (n - 1) + (x - mu_tm1) ** 2 / n)
        self.assertEqual(expected_r, expected_r)

    def test_the_default_obj_works(self):
        rewards = {"rewards": [1], "costs": [2]}
        obj_params = {}
        r = agents.utils.default_obj(rewards, obj_params)
        expected_r = 1.5
        self.assertEqual(expected_r, expected_r)

    def test_the_default_obj_v1_works(self):
        rewards = {"rewards": [1], "costs": [2]}
        w = 0.5
        obj_params = {"w": w}
        r = agents.utils.budget_obj_v1(rewards, obj_params)
        expected_r = w * np.mean(rewards["rewards"]) + (1 - w) / np.mean(
            rewards["costs"]
        )
        self.assertEqual(expected_r, expected_r)

    def test_the_default_obj_v2_works(self):
        rewards = {"rewards": [1], "costs": [2]}
        w = 0.5
        obj_params = {"w": w}
        r = agents.utils.budget_obj_v2(rewards, obj_params)
        expected_r = w * np.mean(rewards["rewards"]) - (1 - w) * np.mean(
            rewards["costs"]
        )
        self.assertEqual(expected_r, expected_r)
