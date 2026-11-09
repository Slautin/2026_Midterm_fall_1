# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""One measurement session: budget accounting, noisy queries and scoring on submit."""
import math
import secrets
from dataclasses import dataclass

import numpy as np

from .noise import NoiseModel
from .rand import HashStream
from .scoring import REQUEST_COST, reference_rmse, rmse
from .spectrum import E_GRID, E_MAX, E_MIN, N_CHANNELS, Spectrum

_BUDGET_TOLERANCE = 1e-9


class SessionError(Exception):
    """Base class for errors the API turns into client errors."""


class InvalidQuery(SessionError, ValueError):
    pass


class BudgetExceeded(SessionError):
    pass


class CallLimitExceeded(SessionError):
    pass


class RoundLimitExceeded(SessionError):
    pass


class AlreadySubmitted(SessionError):
    pass


class InvalidSubmission(SessionError, ValueError):
    pass


@dataclass(frozen=True)
class Measurement:
    wavelength: float
    time: float
    intensity: float


@dataclass(frozen=True)
class Result:
    rmse: float
    reference_rmse: float
    ratio: float  # rmse / reference_rmse; below 1 beats the baseline
    time_used: float
    n_calls: int
    n_rounds: int = 0


class Session:
    """A run against one hidden spectrum.

    A run has a time budget and a cap on the number of queries. Each request (measure, or
    measure_many with any number of queries) is one round, and rounds are counted. There is no
    limit on rounds unless max_rounds is set, so a policy may decide one query at a time or a whole
    stage at a time. Every request also costs request_cost time units, however many queries it
    holds: a big stage is cheaper per query, a small one lets you adapt sooner, and each extra look
    at the data has a price. The cost comes out of the same time budget as the measurements.

    The noise of a query is a deterministic function of (noise_key, wavelength, time, how many
    times that exact query was made before). The same code on the same spectrum therefore gets
    the same values and the same grade, while repeating a query still gives independent noise.
    The duration is part of the input on purpose: if the noise depended only on the wavelength, a
    student could query it at two durations and solve for the truth exactly.
    Without a noise_key a random one is drawn, so every such session has fresh noise.
    """

    def __init__(
        self,
        truth: Spectrum,
        time_budget: float = 500.0,  # calibrated in scripts/calibrate.py (the hackathon used 30)
        max_calls: int = 1000,
        noise: NoiseModel = NoiseModel(),
        noise_key: str | None = None,
        max_rounds: int | None = None,
        request_cost: float = REQUEST_COST,
    ) -> None:
        self._truth = truth
        self._noise = noise
        self._noise_key = secrets.token_hex(16) if noise_key is None else str(noise_key)
        self._repeats: dict[tuple[float, float], int] = {}
        self.time_budget = float(time_budget)
        self.max_calls = int(max_calls)
        self.max_rounds = None if max_rounds is None else int(max_rounds)
        self.rounds_used = 0
        self.request_cost = float(request_cost)
        if self.request_cost < 0:
            raise ValueError("request_cost cannot be negative")
        self.used_time = 0.0
        self.history: list[Measurement] = []
        self.submitted = False

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.time_budget - self.used_time)

    @property
    def rounds_remaining(self) -> int | None:
        """Rounds left, or None when rounds are not limited."""
        return None if self.max_rounds is None else max(0, self.max_rounds - self.rounds_used)

    def measure(self, wavelength, time) -> dict:
        """One query. It uses up one round."""
        self._check_open()
        wl, t = self._query(wavelength, time)
        self._check_capacity([(wl, t)])
        self.used_time += self.request_cost
        return self._end_round([self._take(wl, t)])[0]

    def measure_many(self, queries) -> list[dict]:
        """A whole stage of (wavelength, time) pairs, in order. It counts as one round, however many
        queries it holds. All or nothing: if any query is invalid, or the stage would exceed the
        budget, the call cap or the round limit (if one is set), nothing is spent."""
        self._check_open()
        parsed = [self._query(w, t) for w, t in queries]
        if not parsed:
            raise InvalidQuery("a batch needs at least one query")
        self._check_capacity(parsed)
        self.used_time += self.request_cost
        return self._end_round([self._take(w, t) for w, t in parsed])

    def _check_capacity(self, parsed) -> None:
        if self.max_rounds is not None and self.rounds_used >= self.max_rounds:
            raise RoundLimitExceeded(f"Round limit of {self.max_rounds} reached")
        if len(self.history) + len(parsed) > self.max_calls:
            raise CallLimitExceeded(f"Call limit of {self.max_calls} would be exceeded")
        if self.used_time + self.request_cost + sum(t for _, t in parsed) > self.time_budget + _BUDGET_TOLERANCE:
            raise BudgetExceeded("Measurement-time budget exceeded")

    def _take(self, wl: float, t: float) -> dict:
        signal = float(self._truth(wl))
        repeat = self._repeats.get((wl, t), 0)
        z = HashStream("noise", f"{self._noise_key}|{wl!r}|{t!r}|{repeat}").normal()
        intensity = signal + float(self._noise.sigma(signal, t)) * z
        self._repeats[(wl, t)] = repeat + 1
        self.used_time += t
        self.history.append(Measurement(wl, t, intensity))
        return {
            "wavelength": wl,
            "measurement_time": t,
            "intensity": intensity,
            "time_used": self.used_time,
            "time_remaining": self.time_remaining,
        }

    def _end_round(self, results: list[dict]) -> list[dict]:
        self.rounds_used += 1
        for result in results:
            result["rounds_remaining"] = self.rounds_remaining
        return results

    def submit(self, predicted, energy=None) -> Result:
        """Score a 512-channel prediction and lock the session. A rejected submission does not lock it."""
        self._check_open()
        pred = self._vector(predicted, "predicted")
        if energy is not None and not np.allclose(self._vector(energy, "energy"), E_GRID, atol=1e-7, rtol=0):
            raise InvalidSubmission("Invalid spectral grid")

        error = rmse(pred, self._truth.on_grid)
        reference = reference_rmse(self._truth, self.time_budget, self._noise, self.request_cost)
        self.submitted = True
        return Result(error, reference, error / reference, self.used_time, len(self.history), self.rounds_used)

    def _check_open(self) -> None:
        if self.submitted:
            raise AlreadySubmitted("Episode already submitted")

    def _query(self, wavelength, time) -> tuple[float, float]:
        wl = self._number(wavelength, "wavelength")
        t = self._number(time, "time")
        if not E_MIN <= wl <= E_MAX:
            raise InvalidQuery(f"wavelength must be in [{E_MIN}, {E_MAX}]")
        if not t > 0:
            raise InvalidQuery("time must be positive")
        return wl, t

    @staticmethod
    def _number(value, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InvalidQuery(f"{name} must be a number") from None
        if not math.isfinite(number):
            raise InvalidQuery(f"{name} must be finite")
        return number

    @staticmethod
    def _vector(values, name: str) -> np.ndarray:
        try:
            vector = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise InvalidSubmission(f"{name} must be a list of numbers") from None
        if vector.shape != (N_CHANNELS,) or not np.all(np.isfinite(vector)):
            raise InvalidSubmission(f"{name} must be {N_CHANNELS} finite numbers")
        return vector
