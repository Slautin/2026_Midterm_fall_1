# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""Deterministic, version-stable randomness for building hidden spectra.

numpy's Generator streams are not guaranteed identical across numpy versions, and the hidden
test spectra must never change between deployments. Spectrum parameters are therefore drawn
from a SHA-256 counter stream instead of numpy's generators.
"""
import hashlib
import math


class HashStream:
    """Reproducible stream of uniform numbers keyed by (domain, seed)."""

    def __init__(self, domain: str, seed) -> None:
        self._prefix = f"{domain}\x00{seed}\x00".encode()
        self._counter = 0

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        digest = hashlib.sha256(self._prefix + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        u = (int.from_bytes(digest[:8], "big") >> 11) / 2**53  # exact, in [0, 1)
        return lo + (hi - lo) * u

    def integer(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], both inclusive."""
        return lo + int(self.uniform() * (hi - lo + 1))

    def normal(self) -> float:
        """Standard normal draw (Box-Muller from two uniforms)."""
        u1, u2 = 1.0 - self.uniform(), self.uniform()  # u1 is in (0, 1], so the log is finite
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
