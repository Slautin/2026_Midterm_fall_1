# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""RMSE scoring and the server-side baseline used to normalise it."""
from functools import lru_cache

import numpy as np

from .noise import NoiseModel
from .rand import HashStream
from .spectrum import E_GRID, E_MAX, E_MIN, Spectrum

N_BASELINE_POINTS = 250  # evenly spaced wavelengths, equal time, Gaussian-kernel smoothing
BASELINE_BANDWIDTH = 0.04  # near the best width for a simple scan of the default (narrow) spectra
N_REFERENCE_DRAWS = 32  # noise realisations averaged into the reference
REQUEST_COST = 5.0  # time charged for every request, however many queries it holds; the baseline pays it too


def rmse(predicted, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(truth)) ** 2)))


def _smoothing_weights(wavelengths, bandwidth: float) -> np.ndarray:
    """(512, n) matrix whose rows average n equal-time measurements into one grid channel."""
    kernel = np.exp(-0.5 * ((E_GRID[:, None] - np.asarray(wavelengths)[None, :]) / bandwidth) ** 2)
    return kernel / kernel.sum(axis=1, keepdims=True)


def smooth_to_grid(wavelengths, values, bandwidth: float = BASELINE_BANDWIDTH) -> np.ndarray:
    """Gaussian-kernel regression of equal-time measurements onto the 512-channel grid."""
    return _smoothing_weights(wavelengths, bandwidth) @ np.asarray(values, dtype=float)


BASELINE_WAVELENGTHS = np.linspace(E_MIN, E_MAX, N_BASELINE_POINTS)
_BASELINE_WEIGHTS = _smoothing_weights(BASELINE_WAVELENGTHS, BASELINE_BANDWIDTH)


# Fixed standard-normal noise for the reference draws, built from the hash stream so the
# reference, and therefore every grade, cannot drift with a numpy version.
_REFERENCE_Z = np.array(
    [[HashStream("reference", f"{i}|{j}").normal() for j in range(N_BASELINE_POINTS)]
     for i in range(N_REFERENCE_DRAWS)]
)


def baseline_prediction(truth: Spectrum, time_budget: float, noise: NoiseModel, z,
                        request_cost: float = REQUEST_COST) -> np.ndarray:
    """The simple strategy students are compared against: a uniform smoothed scan in one request.

    z is one standard-normal noise value per scanned wavelength. The single request pays its cost.
    """
    time_each = (time_budget - request_cost) / N_BASELINE_POINTS
    signal = truth(BASELINE_WAVELENGTHS)
    values = signal + noise.sigma(signal, time_each) * np.asarray(z)
    return _BASELINE_WEIGHTS @ values


@lru_cache(maxsize=512)
def reference_rmse(truth: Spectrum, time_budget: float, noise: NoiseModel, request_cost: float = REQUEST_COST) -> float:
    """Mean baseline RMSE on this spectrum over fixed noise draws. Deterministic for a given spectrum."""
    errors = [rmse(baseline_prediction(truth, time_budget, noise, z, request_cost), truth.on_grid) for z in _REFERENCE_Z]
    return float(np.mean(errors))
