# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""Hidden ground-truth spectra: a few peaks on a gently sloping baseline, built from a seed."""
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .rand import HashStream

E_MIN, E_MAX, N_CHANNELS = 0.0, 10.0, 512
E_GRID = np.linspace(E_MIN, E_MAX, N_CHANNELS)

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_PLACEMENT_TRIES = 200


@dataclass(frozen=True)
class GeneratorConfig:
    """Difficulty knobs. Changing any default changes every spectrum, including the frozen test set."""

    n_peaks: tuple[int, int] = (2, 5)
    amplitude: tuple[float, float] = (0.3, 1.2)  # peak height above the baseline
    sigma: tuple[float, float] = (0.06, 0.2)  # Gaussian width of the peak core: narrow lines, so design matters
    lorentz: tuple[float, float] = (0.0, 0.5)  # Lorentzian fraction (pseudo-Voigt mix)
    edge_margin: float = 0.5  # peak centres stay this far from the ends
    min_separation: float = 0.6  # minimum distance between peak centres
    baseline: tuple[float, float] = (0.05, 0.12)  # level at the middle of the range
    slope: tuple[float, float] = (-0.01, 0.01)  # baseline tilt per unit wavelength


@dataclass(frozen=True)
class Spectrum:
    centers: tuple[float, ...]
    amplitudes: tuple[float, ...]
    sigmas: tuple[float, ...]
    lorentz: tuple[float, ...]
    base: float
    slope: float

    @classmethod
    def from_seed(cls, seed, domain: str, config: GeneratorConfig = GeneratorConfig()) -> "Spectrum":
        """Build the spectrum for (domain, seed). The same inputs always give the same spectrum."""
        stream = HashStream(domain, seed)
        n = stream.integer(*config.n_peaks)
        lo, hi = E_MIN + config.edge_margin, E_MAX - config.edge_margin

        centers: list[float] = []
        for _ in range(n):
            candidate = stream.uniform(lo, hi)
            for _ in range(_PLACEMENT_TRIES):
                if all(abs(candidate - c) >= config.min_separation for c in centers):
                    break
                candidate = stream.uniform(lo, hi)
            centers.append(candidate)  # a crowded draw is accepted as is; impossible for default config
        centers.sort()

        amplitudes = [stream.uniform(*config.amplitude) for _ in range(n)]
        sigmas = [stream.uniform(*config.sigma) for _ in range(n)]
        lorentz = [stream.uniform(*config.lorentz) for _ in range(n)]
        base = stream.uniform(*config.baseline)
        slope = stream.uniform(*config.slope)
        return cls(tuple(centers), tuple(amplitudes), tuple(sigmas), tuple(lorentz), base, slope)

    def __call__(self, energy):
        """Noiseless intensity at any wavelength (scalar or array). Not restricted to the 512 channels."""
        e = np.asarray(energy, dtype=float)
        out = self.base + self.slope * (e - 0.5 * (E_MIN + E_MAX))
        for c, a, s, eta in zip(self.centers, self.amplitudes, self.sigmas, self.lorentz):
            gauss = np.exp(-0.5 * ((e - c) / s) ** 2)
            half_width = 0.5 * _FWHM_PER_SIGMA * s
            lorentz = 1.0 / (1.0 + ((e - c) / half_width) ** 2)
            out = out + a * ((1.0 - eta) * gauss + eta * lorentz)
        return out

    @cached_property
    def on_grid(self) -> np.ndarray:
        """Noiseless truth on the 512-channel evaluation grid (read-only)."""
        values = np.asarray(self(E_GRID), dtype=float)
        values.setflags(write=False)
        return values
