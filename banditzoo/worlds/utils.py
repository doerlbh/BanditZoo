#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions and classes related to worlds
"""


def print_progress(t, T, bar_length=20):
    percent = float(t) * 100 / T
    arrow = "-" * int(percent / 100 * bar_length - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    print("run progress: [%s%s] %d %%" % (arrow, spaces, percent), end="\r")
