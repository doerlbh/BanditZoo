"""Test for the utils"""
from unittest import TestCase
import numpy as np

from banditzoo import utils


class TestUtils(TestCase):
    def test_the_default_obj_works(self):
        rewards = [1, 2]
        w = 0.5
        r = utils.default_obj(rewards, w)
        expected_r = w * rewards[0] + (1 - w) / rewards[1]
        self.assertEqual(expected_r, expected_r)
