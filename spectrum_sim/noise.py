# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""Measurement noise law."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Additive Gaussian noise with std = (c0 + c1 * |signal|) / sqrt(time).

    The mean does not depend on acquisition time (intensity is a rate); only the spread shrinks
    as 1/sqrt(time), and it is larger on peaks than on the baseline.
    """

    c0: float = 0.03
    c1: float = 0.07

    def sigma(self, signal, time):
        return (self.c0 + self.c1 * np.abs(signal)) / np.sqrt(time)
