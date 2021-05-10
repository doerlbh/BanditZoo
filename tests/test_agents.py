"""Test for the agents"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import agents


@parameterized_class(
    [
        {"agent": agents.Agent},
        {"agent": agents.MultiArmedAgent},
        {"agent": agents.TS},
        {"agent": agents.OGreedy},
        {"agent": agents.EGreedy},
        {"agent": agents.ContextualAgent},
        {"agent": agents.CombinatorialAgent},
        {"agent": agents.CCTSB},
        {"agent": agents.CRandom},
        {"agent": agents.CRandomFixed},
    ]
)
class TestAllAgents(TestCase):
    def test_the_agent_can_initialize(self):
        a = self.agent()


@parameterized_class(
    [
        {"agent": agents.TS},
        {"agent": agents.OGreedy},
        {"agent": agents.EGreedy},
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
        {"agent": agents.CRandom},
        {"agent": agents.CRandomFixed},
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
