"""Test for the worlds"""
from unittest import TestCase
from parameterized import parameterized_class

from banditzoo import worlds, agents



@parameterized_class(
    [
        {"world": worlds.BernoulliMultiArmedBandits},
    ]
)
class TestMultiArmedBanditWorlds(TestCase):
    def test_the_world_can_initialize(self):
        world = self.world
        w = world()

    def test_the_world_can_run_with_one_agent(self):
        world = self.world
        w = world(M=5)
        a = agents.Random(M=5)
        w.add_agent(a)
        w.run_experiments(T=10)

    def test_the_world_can_run_with_multiple_agents(self):
        world = self.world
        w = world(M=5)
        a1 = agents.Random(M=5)
        a2 = agents.TS(M=5)
        w.add_agent(a1)
        w.add_agent(a2)
        w.run_experiments(T=10)
        
@parameterized_class(
    [
        {"world": worlds.EpidemicControl_v1},
        {"world": worlds.EpidemicControl_v2},
    ]
)
class TestCombinatorialWorlds(TestCase):
    def test_the_world_can_initialize(self):
        world = self.world
        w = world()

    def test_the_world_can_run_with_one_agent(self):
        world = self.world
        w = world(K=2, N=[3, 4], C=12)
        a = agents.CCTSB(K=2, N=[3, 4], C=12)
        w.add_agent(a)
        w.run_experiments(T=10)

    def test_the_world_can_run_with_multiple_agents(self):
        world = self.world
        w = world(K=2, N=[3, 4], C=12)
        a1 = agents.CCTSB(K=2, N=[3, 4], C=12)
        a2 = agents.CombRandom(K=2, N=[3, 4], C=12)
        w.add_agent(a1)
        w.add_agent(a2)
        w.run_experiments(T=10)
