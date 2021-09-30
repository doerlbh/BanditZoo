"""Test for the worlds"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import worlds, agents


@parameterized_class(
    [
        {"world": worlds.World},
        {"world": worlds.MultiArmedBandits},
        {"world": worlds.BernoulliMultiArmedBandits},
        {"world": worlds.ContextualCombinatorialBandits},
        {"world": worlds.EpidemicControl},
        {"world": worlds.EpidemicControl_v1},
        {"world": worlds.EpidemicControl_v2},
    ]
)
class TestMultiArmedBanditWorlds(TestCase):
    def test_the_world_can_initialize(self):
        world = self.world
        w = world()

    def test_the_world_can_add_agent(self):
        world = self.world
        w = world()
        a = agents.Random(n_arms=5)
        w.add_agent(a)

    def test_the_world_can_add_agent_pools(self):
        world = self.world
        w = world()
        a = agents.Random(n_arms=5)
        w.add_agent_pool([a, a])

    def test_the_world_can_add_agent_and_build_them(self):
        world = self.world
        w = world(n_arms=5)
        a1 = agents.Random()
        a2 = agents.TS()
        w.add_agent_pool([a1, a2])

    def test_the_world_can_filter_agents(self):
        world = self.world
        w = world()
        a1 = agents.Random(n_arms=5, seed=0, name="Random")
        a2 = agents.Random(n_arms=5, seed=1, name="Random")
        a3 = agents.TS(n_arms=5, seed=0, name="TS")
        w.add_agent_pool([a1, a2, a3])
        self.assertEqual(len(w.filter_agent("Random")), 2)

    def test_the_world_can_filter_agents_index(self):
        world = self.world
        w = world()
        a1 = agents.Random(n_arms=5, seed=0, name="Random")
        a2 = agents.TS(n_arms=5, seed=0, name="TS")
        a3 = agents.Random(n_arms=5, seed=1, name="Random")
        w.add_agent_pool([a1, a2, a3])
        np.allclose(w.filter_agent("Random", get_index=True), [0, 2])


@parameterized_class(
    [
        {"world": worlds.MultiArmedBandits},
        {"world": worlds.BernoulliMultiArmedBandits},
    ]
)
class TestMultiArmedBanditWorlds(TestCase):
    def test_the_world_can_run_with_one_agent(self):
        world = self.world
        w = world(n_arms=5)
        a = agents.Random(n_arms=5)
        w.add_agent(a)
        w.run_experiments(T=10)

    def test_the_world_can_run_with_multiple_agents(self):
        world = self.world
        w = world(n_arms=5)
        a1 = agents.Random(n_arms=5)
        a2 = agents.UCB1(n_arms=5)
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
    def test_the_world_can_run_with_one_agent(self):
        world = self.world
        w = world(action_dimension=2, action_options=[3, 4], context_dimension=12)
        a = agents.CCTSB(
            action_dimension=2, action_options=[3, 4], context_dimension=12
        )
        w.add_agent(a)
        w.run_experiments(T=10)

    def test_the_world_can_run_with_multiple_agents(self):
        world = self.world
        w = world(action_dimension=2, action_options=[3, 4], context_dimension=12)
        a1 = agents.CCTSB(
            action_dimension=2, action_options=[3, 4], context_dimension=12
        )
        a2 = agents.CombRandom(
            action_dimension=2, action_options=[3, 4], context_dimension=12
        )
        w.add_agent(a1)
        w.add_agent(a2)
        w.run_experiments(T=10)

    def test_the_world_can_run_with_combinatorial_cost(self):
        world = self.world
        w = world(
            action_dimension=2,
            action_options=[3, 4],
            context_dimension=12,
            combinatorial_cost=True,
        )
        a1 = agents.CCTSB()
        a2 = agents.CombRandom()
        w.add_agent(a1)
        w.add_agent(a2)
        w.run_experiments(T=10)
