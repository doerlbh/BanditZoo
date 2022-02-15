"""Test for the worlds"""
from unittest import TestCase

import sys

sys.path.insert(0, "..")

from examples import epidemic
from examples import multi_armed_bandit
from examples import two_rewards
from examples import evobandit


class TestExamples(TestCase):
    def test_examples(self):
        epidemic.test()
        multi_armed_bandit.test()
        two_rewards.test()
        evobandit.test()
