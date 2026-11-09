# Copied from the server repository by scripts/sync_sim.py. Do not edit here: change the source, then copy again.
"""A local training environment with the server's exact rules, for dynamic programming and RL.

    env = SpectrumEnv()
    obs = env.reset(seed=7)                        # the spectrum and noise the server serves for train seed 7
    obs, reward, done, info = env.step([(3.0, 5.0), (3.2, 5.0)])   # one round: (wavelength, time) pairs
    result = env.submit(prediction)                # the score; a natural terminal reward is -log(result.ratio)

An episode ends when the time budget or the query cap is used up (or the round limit, if you set one
with max_rounds; by default rounds are only counted). A step may be a single query or a whole stage,
and every step pays env.request_cost out of the time budget, so a step is a real decision: look
again soon and adapt, or send a bigger stage and save the overhead.
The observation is everything measured so far plus what is left. In training the truth is visible
(env.truth, env.error) so you can shape a dense reward; on the server it never is.
"""
import math

import numpy as np

from .noise import NoiseModel
from .scoring import REQUEST_COST, rmse
from .session import Result, Session
from .spectrum import GeneratorConfig, Spectrum


class SpectrumEnv:
    def __init__(self, time_budget: float = 500.0, max_rounds: int | None = None, max_calls: int = 1000,
                 noise: NoiseModel = NoiseModel(), config: GeneratorConfig = GeneratorConfig(),
                 request_cost: float = REQUEST_COST) -> None:
        self.time_budget, self.max_rounds, self.max_calls = float(time_budget), max_rounds, int(max_calls)
        self.request_cost = float(request_cost)
        self.noise, self.config = noise, config
        self.session: Session | None = None
        self.truth: Spectrum | None = None
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode on a training spectrum. The same seed always gives the same spectrum and noise."""
        self.seed = int(np.random.default_rng().integers(0, 2**31)) if seed is None else int(seed)
        self.truth = Spectrum.from_seed(self.seed, "train", self.config)
        self.session = Session(self.truth, self.time_budget, self.max_calls, self.noise,
                               noise_key=f"train:{self.seed}", max_rounds=self.max_rounds,
                               request_cost=self.request_cost)
        self._last_error = float(np.sqrt(np.mean(self.truth.on_grid ** 2)))  # error of the all-zero estimate
        return self._observation()

    def step(self, queries, estimate=None):
        """Spend one round on a list of (wavelength, time) pairs.

        Returns (observation, reward, done, info). The reward is 0 unless you also pass your current
        512-point `estimate`; then it is the log of how much the true error shrank (dense shaping).
        An invalid stage raises, as on the server, and costs nothing.
        """
        session = self._need_session()
        session.measure_many(np.asarray(queries, dtype=float).reshape(-1, 2).tolist())
        reward = 0.0
        if estimate is not None:
            error = max(self.error(estimate), 1e-12)
            reward, self._last_error = math.log(self._last_error / error), error
        return self._observation(), reward, self.done, {"rounds_left": session.rounds_remaining,
                                                        "time_left": session.time_remaining}

    def submit(self, prediction) -> Result:
        """Score the final 512-point prediction (the same score the server reports) and end the episode."""
        return self._need_session().submit(prediction)

    def error(self, prediction) -> float:
        """True RMSE of a prediction. Privileged: only the local environment can tell you this."""
        return rmse(prediction, self._need_session()._truth.on_grid)

    @property
    def done(self) -> bool:
        s = self._need_session()
        return s.rounds_remaining == 0 or s.time_remaining <= 1e-9 or len(s.history) >= s.max_calls

    def _observation(self) -> dict:
        s = self._need_session()
        measurements = np.array([(m.wavelength, m.time, m.intensity) for m in s.history], dtype=float).reshape(-1, 3)
        return {"measurements": measurements, "time_left": s.time_remaining, "rounds_used": s.rounds_used,
                "rounds_left": s.rounds_remaining, "calls_left": s.max_calls - len(s.history),
                "request_cost": s.request_cost}

    def _need_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("call reset() first")
        return self.session


def evaluate(strategy, seeds=range(20), **session_kwargs) -> np.ndarray:
    """Run strategy(session) on training seeds with the server's rules, fast and offline. Returns the ratios."""
    ratios = []
    for seed in seeds:
        s = Session(Spectrum.from_seed(seed, "train"), noise_key=f"train:{seed}", **session_kwargs)
        ratios.append(s.submit(strategy(s)).ratio)
    return np.array(ratios)
