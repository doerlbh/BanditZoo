# Copyright 2021 Baihan Lin
#
# Licensed under the GNU General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions and classes related to worlds
"""

import numpy as np


def print_progress(t: int, T: int, bar_length: int = 20):
    percent = float(t) * 100 / T
    arrow = "-" * int(percent / 100 * bar_length - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    print("run progress: [%s%s] %d %%" % (arrow, spaces, percent), end="\r")


def check_and_correct_dimensions(
    name: str,
    means: np.ndarray,
    stds: np.ndarray,
    dimension: int,
    target_dimension: int,
):
    # TODO add tests
    if not np.array_equal(means.shape, stds.shape):
        raise ValueError(
            "Please specify the same shape for "
            + name
            + " means and stds, now "
            + str(means.shape)
            + " and "
            + +str(stds.shape)
        )
    if means.ndim == target_dimension - 1:
        means = np.expand_dims(means, -1)
        stds = np.expand_dims(stds, -1)

    if dimension != means.shape[-1]:
        if dimension != target_dimension - 1:
            raise ValueError(
                "Please specify the same shape for "
                + name
                + "dimension and the "
                + "dimensions of its mean, now "
                + str(dimension)
                + " and "
                + +str(means.shape[-1])
            )
        dimension = means.shape[-1]

    return means, stds, dimension
