"""Test for the games"""
from unittest import TestCase
from parameterized import parameterized_class
import numpy as np

from banditzoo import games, worlds, agents


@parameterized_class(
    [
        {"game": games.Game},
        {"game": games.MultiObjectiveGame},
    ]
)
class TestGames(TestCase):
    def test_the_game_can_initialize(self):
        game = self.game
        g = game()

    def test_the_game_can_add_one_world(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )

    def test_the_game_can_add_multiple_worlds(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=4, name="MAB4"
        )

    def test_the_game_can_add_one_agent_to_one_world(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)

    def test_the_game_can_add_multiple_agents_to_one_world(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.add_agent_class(agents.EGreedy, q_start=100, epsilon=0.2)

    def test_the_game_can_add_one_agent_to_multiple_worlds(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5A"
        )
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, reward_scale=2, name="MAB5B"
        )
        g.add_agent_class(agents.TS, n_agent_instances=5)

    def test_the_game_can_add_multiple_agents_to_multiple_worlds(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5A"
        )
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5B"
        )
        g.add_agent_class(agents.TS, n_agent_instances=5)
        g.add_agent_class(agents.EGreedy, n_agent_instances=5, q_start=100, epsilon=0.2)

    def test_the_game_can_run_multiple_agents_to_multiple_worlds_with_different_configs(
        self,
    ):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=3, name="MAB3"
        )
        g.add_agent_class(agents.TS)
        g.add_agent_class(agents.EGreedy, q_start=100, epsilon=0.2)
        g.run_experiments(T=10)

    def test_the_game_raise_error_if_agents_are_assigned_before_worlds(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        with self.assertRaises(Exception) as context:
            g.add_agent_class(agents.TS, n_agent_instances=5)
        self.assertTrue(
            "Please initiate all the worlds before adding agents."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_agents_are_assigned_if_game_is_run(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
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
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.run_experiments(T=10)
        with self.assertRaises(Exception) as context:
            g.add_world_class(
                worlds.BernoulliMultiArmedBandits, n_agent_instances=3, name="MAB5"
            )
        self.assertTrue(
            "No worlds can enter anymore because the game is run."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_worlds_are_assigned_after_agents_enter(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        with self.assertRaises(Exception) as context:
            g.add_world_class(
                worlds.BernoulliMultiArmedBandits, n_agent_instances=3, name="MAB5"
            )
        self.assertTrue(
            "No worlds can enter anymore because the agents are in."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_get_metrics_gets_a_bad_tag(self):
        game = self.game
        g = game(n_agent_instances=3, n_world_instances=4)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.run_experiments(T=10)
        with self.assertRaises(ValueError) as context:
            g.get_metrics(form="crazy")
        self.assertTrue(
            "Please select a supported format ('tabular', 'agent', 'world')."
            in str(context.exception)
        )

    def test_if_get_metrics_works_in_tabular_form(self):
        game = self.game
        g = game(n_agent_instances=10, n_world_instances=10)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.run_experiments(T=2)
        expected_shape = [300, 5]
        metrics = g.get_metrics(form="tabular")
        np.allclose(metrics.shape, expected_shape)

    def test_if_get_metrics_works_when_group_by_agent(self):
        game = self.game
        g = game(n_agent_instances=10, n_world_instances=10)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.run_experiments(T=2)
        expected_reward = [0, 0.3, 0.6]
        metrics = g.get_metrics(form="agent")
        avg_reward = metrics["MAB5"]["TS"]["reward_avg"]
        np.allclose(avg_reward, expected_reward)

    def test_if_get_metrics_works_when_group_by_world(self):
        game = self.game
        g = game(n_agent_instances=10, n_world_instances=10)
        g.add_world_class(
            worlds.BernoulliMultiArmedBandits, n_agent_instances=5, name="MAB5"
        )
        g.add_agent_class(agents.TS)
        g.run_experiments(T=2)
        expected_reward = [0, 0.3, 0.6]
        metrics = g.get_metrics(form="world")
        avg_reward = metrics["MAB5"][0]["TS"]["reward_avg"]
        np.allclose(avg_reward, expected_reward)


@parameterized_class(
    [
        {"game": games.MultiObjectiveGame},
    ]
)
class TestMultiObjectiveGames(TestCase):
    def test_the_game_can_set_params_search_to_multiple_worlds(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            n_world_instances=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=2,
            n_world_instances=[2, 2],
            context_dimension=4,
            name="EpidemicB",
        )
        g.add_agent_class(agents.CCTSB)
        g.add_agent_class(agents.CCMABB, agent_base=agents.UCB1)
        g.set_params_sweep(w=[0, 0.5, 1])

    def test_the_game_can_set_multiple_params_search_to_multiple_worlds(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            n_world_instances=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=2,
            n_world_instances=[2, 2],
            context_dimension=4,
            name="EpidemicB",
        )
        g.add_agent_class(agents.CCTSB)
        g.add_agent_class(agents.CCMABB, agent_base=agents.UCB1)
        g.set_params_sweep(w=[0, 0.5, 1], dummy=[1, 2])

    def test_the_game_can_run_with_different_obj_params(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            action_options=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=2,
            action_options=[2, 2],
            context_dimension=4,
            name="EpidemicB",
        )
        g.add_agent_class(agents.CCTSB)
        g.add_agent_class(agents.CCMABB, agent_base=agents.UCB1)
        g.set_params_sweep(w=[0, 0.5, 1], dummy=[1, 2])
        g.run_experiments(T=10)

    def test_the_game_raise_error_if_agents_are_set_params_before_assigned(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            n_world_instances=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=2,
            n_world_instances=[2, 2],
            context_dimension=4,
            name="EpidemicB",
        )
        with self.assertRaises(Exception) as context:
            g.set_agent_params(w=0)
        self.assertTrue(
            "Please initiate all the agents before setting objective params."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_agents_are_set_params_without_params(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            n_world_instances=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=2,
            n_world_instances=[2, 2],
            context_dimension=4,
            name="EpidemicB",
        )
        g.add_agent_class(agents.CCTSB)
        with self.assertRaises(Exception) as context:
            g.set_agent_params()
        self.assertTrue(
            "Please have at least one objective function parameters to set."
            in str(context.exception)
        )

    def test_the_game_raise_error_if_pareto_are_computed_without_params_set(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            action_options=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_agent_class(agents.CCMABB, agent_base=agents.UCB1)
        g.run_experiments(T=10)
        with self.assertRaises(Exception) as context:
            g.get_pareto_metrics()
        self.assertTrue(
            "There is no objective function parameters to construct the pareto frontier."
            in str(context.exception)
        )

    def test_if_get_pareto_metrics_works_in_tabular_form(self):
        game = self.game
        g = game(n_agent_instances=2, n_world_instances=2)
        g.add_world_class(
            worlds.EpidemicControl_v1,
            action_dimension=1,
            action_options=[2],
            context_dimension=2,
            name="EpidemicA",
        )
        g.add_agent_class(agents.CCTSB)
        g.add_agent_class(agents.CCMABB, agent_base=agents.UCB1)
        g.set_params_sweep(w=[0, 0.5, 1], dummy=[1, 2])
        g.run_experiments(T=20)
        expected_shape = [96, 7]
        metrics = g.get_pareto_metrics(quantile_bin=True, n_bins=3)
        print(metrics.shape)
        np.allclose(metrics.shape, expected_shape)
