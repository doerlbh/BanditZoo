"""Test for the games"""
from unittest import TestCase
from parameterized import parameterized_class

from banditzoo import games, worlds, agents


@parameterized_class(
    [
        {"game": games.Game},
    ]
)
class TestGames(TestCase):
    def test_the_game_can_initialize(self):
        game = self.game
        g = game()

    def test_the_game_can_add_one_world(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")

    def test_the_game_can_add_multiple_worlds(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=4, name="MAB4")

    def test_the_game_can_add_one_agent_to_one_world(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS, M=5)

    def test_the_game_can_add_multiple_agents_to_one_world(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS, M=5)
        g.add_agent_class(agents.EGreedy, M=5, q_start=100, epsilon=0.2)

    def test_the_game_can_add_one_agent_to_multiple_worlds(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5A")
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, M=5, reward_scale=2, name="MAB5B"
        )
        g.add_agent_class(agents.TS, M=5)

    def test_the_game_can_add_multiple_agents_to_multiple_worlds(self):
        game = self.game
        g = game(M=6, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5A")
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5B")
        g.add_agent_class(agents.TS, M=5)
        g.add_agent_class(agents.EGreedy, M=5, q_start=100, epsilon=0.2)

    def test_the_game_raise_error_if_agents_are_assigned_before_worlds(self):
        game = self.game
        g = game(M=6, N=10)
        with self.assertRaises(Exception) as context:
            g.add_agent_class(agents.TS, M=5)
        self.assertTrue(
            "Please initiate all the worlds before adding agents."
            in str(context.exception)
        )
