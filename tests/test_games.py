"""Test for the games"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

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
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")

    def test_the_game_can_add_multiple_worlds(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=4, name="MAB4")

    def test_the_game_can_add_one_agent_to_one_world(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)

    def test_the_game_can_add_multiple_agents_to_one_world(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.add_agent_class(agents.EGreedy, q_start=100, epsilon=0.2)

    def test_the_game_can_add_one_agent_to_multiple_worlds(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5A")
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, reward_scale=2, name="MAB5B"
        )
        g.add_agent_class(agents.TS, M=5)

    def test_the_game_can_add_multiple_agents_to_multiple_worlds(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5A")
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5B")
        g.add_agent_class(agents.TS, M=5)
        g.add_agent_class(agents.EGreedy, M=5, q_start=100, epsilon=0.2)

    def test_the_game_can_run_multiple_agents_to_multiple_worlds_with_different_configs(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=3, name="MAB3")
        g.add_agent_class(agents.TS)
        g.add_agent_class(agents.EGreedy, q_start=100, epsilon=0.2)
        g.run_experiments(T=10)
        
    def test_the_game_raise_error_if_agents_are_assigned_before_worlds(self):
        game = self.game
        g = game(M=3, N=4)
        with self.assertRaises(Exception) as context:
            g.add_agent_class(agents.TS, M=5)
        self.assertTrue(
            "Please initiate all the worlds before adding agents."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_agents_are_assigned_if_game_is_run(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.run_experiments(T=10)
        with self.assertRaises(Exception) as context:
            g.add_agent_class(agents.TS)
        self.assertTrue(
            "No agents can enter anymore because the game is run."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_worlds_are_assigned_if_game_is_run(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.run_experiments(T=10)
        with self.assertRaises(Exception) as context:
            g.add_world_class(worlds.BernoulliMultiArmedBandits, M=3, name="MAB5")
        self.assertTrue(
            "No worlds can enter anymore because the game is run."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_worlds_are_assigned_after_agents_enter(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        with self.assertRaises(Exception) as context:
            g.add_world_class(worlds.BernoulliMultiArmedBandits, M=3, name="MAB5")
        self.assertTrue(
            "No worlds can enter anymore because the agents are in."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_get_metrics_gets_a_bad_tag(self):
        game = self.game
        g = game(M=3, N=4)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.run_experiments(T=10)
        with self.assertRaises(ValueError) as context:
            g.get_metrics(group_by="crazy")
        self.assertTrue(
            "Please select a supported grouping tag ('agent', 'world')."
            in str(context.exception)
        )
 
    def test_the_game_raise_error_if_get_metrics_works_when_group_by_agent(self):
        game = self.game
        g = game(M=10, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.run_experiments(T=2)
        expected_reward = [0, 0.3, 0.6]
        metrics = g.get_metrics(group_by="agent")
        avg_reward = metrics['MAB5']['TS']['reward_avg']
        np.allclose(avg_reward, expected_reward)
               
    def test_the_game_raise_error_if_get_metrics_works_when_group_by_world(self):
        game = self.game
        g = game(M=10, N=10)
        g.add_world_class(worlds.BernoulliMultiArmedBandits, M=5, name="MAB5")
        g.add_agent_class(agents.TS)
        g.run_experiments(T=2)
        expected_reward = [0, 0.3, 0.6]
        metrics = g.get_metrics(group_by="world")
        avg_reward = metrics['MAB5'][0]['TS']['reward_avg']
        np.allclose(avg_reward, expected_reward)