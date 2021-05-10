"""Test for the agents"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import agents


@parameterized_class(
    [
        {"agent": agents.Agent},
        {"agent": agents.MultiArmedAgent},
        {"agent": agents.Random},
        {"agent": agents.TS},
        {"agent": agents.OGreedy},
        {"agent": agents.EGreedy},
        {"agent": agents.UCB1},
        {"agent": agents.ContextualAgent},
        {"agent": agents.CombinatorialAgent},
        {"agent": agents.CCTSB},
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
    ]
)
class TestMultiArmedAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent(M=5)

    def test_the_agent_can_act(self):
        a = self.agent(M=5)
        action = a.act()

    def test_the_agent_can_update(self):
        a = self.agent(M=5)
        action = a.act()
        reward = 1
        a.update(reward)


@parameterized_class(
    [
        {"agent": agents.CCTSB},
        {"agent": agents.CombRandom},
        {"agent": agents.CombRandomFixed},
    ]
)
class TestContextualCombinatorialAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent(K=2, N=[3, 4], C=12)

    def test_the_agent_can_act(self):
        a = self.agent(K=2, N=[3, 4], C=12)
        context = np.arange(12)
        a.observe(context)
        action = a.act()

    def test_the_agent_can_update(self):
        a = self.agent(K=2, N=[3, 4], C=12)
        context = np.arange(12)
        a.observe(context)
        action = a.act()
        reward = [10, 20]
        a.update(reward)
