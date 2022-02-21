"""Test for the worlds"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import worlds, agents


@parameterized_class(
    [
        {"world": worlds.MultiArmedBandits},
        {"world": worlds.BernoulliMultiArmedBandits},
        {"world": worlds.ContextualCombinatorialBandits},
        {"world": worlds.EpidemicControl},
        {"world": worlds.EpidemicControl_v1},
        {"world": worlds.EpidemicControl_v2},
    ]
)
class TestBanditWorlds(TestCase):
    def test_the_world_can_initialize(self):
        world = self.world
        w = world(n_arms=5)

    def test_the_world_can_add_agent(self):
        world = self.world
        w = world(n_arms=5)
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_specified_reward_reveal_frequency(self):
        world = self.world
        w = world(n_arms=5, reward_reveal_frequency=[0.9])
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_specified_cost_reveal_frequency(self):
        world = self.world
        w = world(n_arms=5, cost_reveal_frequency=[0.9])
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_multiple_reward_dimensions(self):
        world = self.world
        w = world(n_arms=5, reward_dimension=3)
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_multiple_reward_dimensions_with_scale_and_base(
        self,
    ):
        world = self.world
        w = world(n_arms=5, reward_dimension=3, reward_scale=1, reward_base=3)
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_change_reward_every_n_rounds(
        self,
    ):
        world = self.world
        w = world(
            n_arms=5,
            change_reward_every=5,
            reward_dimension=3,
            reward_scale=1,
            reward_base=3,
        )
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_change_cost_every_n_rounds(
        self,
    ):
        world = self.world
        w = world(
            n_arms=5,
            change_cost_every=5,
            reward_dimension=3,
            reward_scale=1,
            reward_base=3,
        )
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_specified_multi_reward_reveal_frequency(self):
        world = self.world
        w = world(n_arms=5, reward_dimension=2, reward_reveal_frequency=[0.9, 0.7])
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_throws_error_if_feedback_reveal_frequency_mismatches_dimension(
        self,
    ):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(n_arms=5, reward_dimension=1, reward_reveal_frequency=[0.9, 0.7])
        self.assertTrue(
            "Please specify the same shape for feedback dimension and reveal_frequency, now 1 and 2"
            in str(context.exception)
        )

    def test_the_world_can_add_agent_with_specified_multi_cost_reveal_frequency(self):
        world = self.world
        w = world(n_arms=5, cost_dimension=2, cost_reveal_frequency=[0.9, 0.7])
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_predefined_reward(self):
        world = self.world
        w = world(
            n_arms=3, reward_means=[1, 2, 3], reward_stds=[0, 2, 1], reward_dimension=1
        )
        a = agents.Random(n_arms=3)
        w.add_agent(a)

    def test_the_world_can_add_agent_with_predefined_multi_dimensional_reward(self):
        world = self.world
        w = world(
            n_arms=3,
            reward_means=[[1, 2], [2, 3], [2, 1]],
            reward_stds=[[0, 2], [1, 2], [1, 0]],
            reward_dimension=1,
        )
        a = agents.Random(n_arms=3)
        w.add_agent(a)

    def test_the_world_can_add_agent_with_predefined_cost(self):
        world = self.world
        w = world(n_arms=3, cost_means=[1, 2, 3], cost_stds=[0, 2, 1])
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_can_add_agent_with_predefined_multi_dimensional_cost(self):
        world = self.world
        w = world(
            n_arms=3,
            cost_means=[[1, 2], [3, 2], [3, 1]],
            cost_stds=[[0, 2], [1, 2], [1, 0]],
        )
        a = agents.Random()
        w.add_agent(a)

    def test_the_world_throws_error_if_predefined_reward_dimension_mismatches_1(self):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(
                n_arms=3, reward_means=[1, 2, 3], reward_stds=[0, 2], reward_dimension=1
            )
        self.assertTrue(
            "Please specify the same shape for reward means and stds, now (3,) and (2,)"
            in str(context.exception)
        )

    def test_the_world_throws_error_if_predefined_reward_dimension_mismatches_2(self):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(
                n_arms=3,
                reward_means=[1, 2, 3],
                reward_stds=[0, 2, 1],
                reward_dimension=2,
            )
        self.assertTrue(
            "Please specify the same shape for reward dimension and the dimensions of its mean, now 2 and 1"
            in str(context.exception)
        )

    def test_the_world_throws_error_if_predefined_cost_dimension_mismatches_1(self):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(
                n_arms=3, cost_means=[1, 2, 3], cost_stds=[0, 2], cost_dimension=1
            )
        self.assertTrue(
            "Please specify the same shape for cost means and stds, now (3,) and (2,)"
            in str(context.exception)
        )

    def test_the_world_throws_error_if_predefined_cost_dimension_mismatches_2(self):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(
                n_arms=3, cost_means=[1, 2, 3], cost_stds=[0, 2, 1], cost_dimension=2
            )
        self.assertTrue(
            "Please specify the same shape for cost dimension and the dimensions of its mean, now 2 and 1"
            in str(context.exception)
        )

    def test_the_world_can_add_agent_pools(self):
        world = self.world
        w = world(n_arms=5)
        a = agents.Random()
        w.add_agent_pool([a, a])

    def test_the_world_can_add_agent_and_build_them(self):
        world = self.world
        w = world(n_arms=5)
        a1 = agents.Random()
        a2 = agents.TS()
        w.add_agent_pool([a1, a2])

    def test_the_world_can_filter_agents(self):
        world = self.world
        w = world(n_arms=5)
        a1 = agents.Random(seed=0, name="Random")
        a2 = agents.Random(seed=1, name="Random")
        a3 = agents.TS(seed=0, name="TS")
        w.add_agent_pool([a1, a2, a3])
        self.assertEqual(len(w.filter_agent("Random")), 2)

    def test_the_world_can_filter_agents_index(self):
        world = self.world
        w = world(n_arms=5)
        a1 = agents.Random(seed=0, name="Random")
        a2 = agents.TS(seed=0, name="TS")
        a3 = agents.Random(seed=1, name="Random")
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
        {"world": worlds.BernoulliMultiArmedBandits},
    ]
)
class TestBernoulliMultiArmedBanditWorlds(TestCase):
    def test_the_world_throws_error_if_reward_scale_is_not_1(self):
        world = self.world
        with self.assertRaises(Exception) as context:
            w = world(n_arms=3, reward_scale=2)
        self.assertTrue(
            "The reward_scale in Bernoulli bandits can only be 1, now 2"
            in str(context.exception)
        )


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


class TestUtils(TestCase):
    def test_print_progress_works(self):
        t, T, bar_length = 2, 10, 10
        worlds.utils.print_progress(t, T, bar_length)
