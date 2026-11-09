"""Client for the spectrum sampling server. Needs only numpy and requests.

    server = Server("https://...")
    s = server.train_session(seed=7)            # a training spectrum you can repeat
    results = s.measure_many([(3.0, 1.0), ...])  # (wavelength, time) pairs
    s.submit(prediction)                         # 512 numbers on E_GRID

A strategy is a function strategy(session) -> 512 numbers. train_eval() and run_test() call it.
"""
from __future__ import annotations

import json
import math
import time as _time
from dataclasses import dataclass

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

E_GRID = np.linspace(0.0, 10.0, 512)  # the 512 wavelengths your prediction must cover
_TOLERANCE = 1e-9


class ServerError(Exception):
    """The server refused a request, or could not be reached (status 0)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}" if status else detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Measurement:
    wavelength: float
    time: float
    intensity: float


def _detail(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):  # validation errors
        return "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}" for e in detail)
    return str(detail)


def _check_prediction(prediction) -> np.ndarray:
    try:
        values = np.asarray(prediction, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("your strategy must return a list of 512 numbers") from None
    if values.shape != (len(E_GRID),) or not np.all(np.isfinite(values)):
        raise ValueError(f"your strategy must return 512 finite numbers, one per point of E_GRID; got shape {values.shape}")
    return values


def _geomean(values) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(values, dtype=float), 1e-12)))))


class Server:
    def __init__(self, url: str, wait: float = 90.0, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._http = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=None, connect=3, read=0, status=0, backoff_factor=0.5,
                                                allowed_methods=None))  # only retries that never reached the server
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)
        self.info = self._wait_for_info(wait)
        limit = self.info.get("max_rounds")
        rounds = "" if limit is None else f" in {limit} rounds"
        print(f"connected to {self.url}: time budget {self.info['time_budget']:g}, each request costs "
              f"{self.info.get('request_cost', 0):g}, up to {self.info['max_calls']} queries{rounds} per run, "
              f"test mode {'on' if self.info['test_enabled'] else 'off'}")

    def _request(self, method: str, path: str, body=None, session_id: str | None = None):
        headers = {"X-Session-ID": session_id} if session_id else {}
        try:
            response = self._http.request(method, self.url + path, json=body, headers=headers, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            raise ServerError(0, f"could not reach the server at {self.url} ({type(exc).__name__})") from exc
        if not response.ok:
            raise ServerError(response.status_code, _detail(response))
        return response.json()

    def _wait_for_info(self, wait: float) -> dict:
        deadline, announced = _time.time() + wait, False
        while True:
            try:
                return self._request("GET", "/info")
            except ServerError as exc:
                if exc.status != 0 or _time.time() >= deadline:
                    raise
                if not announced:
                    print("waiting for the server to wake up ...")
                    announced = True
                _time.sleep(2)

    def train_session(self, seed: int) -> "Session":
        """A training spectrum. The same seed always gives the same spectrum and the same noise."""
        return Session(self, self._request("POST", "/session", {"mode": "train", "seed": int(seed)}))

    def _test_session(self, attempt_id: str, index: int) -> "Session":
        return Session(self, self._request("POST", "/session", {"mode": "test", "attempt_id": attempt_id, "index": index}))


class Session:
    """One run against one hidden spectrum. Same interface as the server's own Session.

    A run has a time budget and a cap on queries. Each request is one round (counted in rounds_used):
    measure() is one, and measure_many() is one however many queries it holds. Every request also costs
    request_cost out of the same budget, so a big stage is cheaper per query and a small one lets you
    adapt sooner. Rounds are not limited unless the server sets a limit.
    """

    def __init__(self, server: Server, created: dict):
        self._server = server
        self._id = created["session_id"]
        self.mode = created["mode"]
        self.time_budget = float(created["time_budget"])
        self.max_calls = int(created["max_calls"])
        self.request_cost = float(created.get("request_cost", 0.0))  # charged for every request
        self.max_rounds = created.get("max_rounds")  # None: rounds are not limited
        self.rounds_remaining = created.get("rounds_remaining")
        self.rounds_used = 0
        self.time_remaining = float(created.get("time_remaining", self.time_budget))
        self.history: list[Measurement] = []
        self._calls = 0
        if created.get("resumed"):
            state = self._server._request("GET", "/budget", session_id=self._id)
            self._calls, self.rounds_used = state["measurements"], state["rounds_used"]

    @property
    def used_time(self) -> float:
        return self.time_budget - self.time_remaining

    def measure(self, wavelength, time) -> dict:
        """One query, which counts as one round."""
        if self.rounds_remaining is not None and self.rounds_remaining < 1:
            raise ValueError(f"Round limit of {self.max_rounds} reached")
        result = self._server._request("POST", "/pixel", {"wavelength": float(wavelength), "time": float(time)}, self._id)
        self._record(result)
        self.rounds_used += 1
        return result

    def measure_many(self, queries) -> list[dict]:
        """Measure a stage of (wavelength, time) pairs: one round. Checked before anything is sent, so a
        bad query or an over-budget stage spends nothing (and so does a missing round, if the server
        limits rounds). If the server limits the size of one request, a long list is split into several
        requests, and each one counts as a round."""
        queries = [(float(w), float(t)) for w, t in queries]
        low, high = self._server.info["wavelength_range"]
        for w, t in queries:
            if not (np.isfinite(w) and np.isfinite(t) and low <= w <= high and t > 0):
                raise ValueError(f"invalid query (wavelength {w}, time {t}): wavelength must be in [{low:g}, {high:g}], time > 0")
        size, results = self._server.info["max_batch"], []
        requests_needed = math.ceil(len(queries) / size)
        need = sum(t for _, t in queries) + self.request_cost * requests_needed
        if need > self.time_remaining + _TOLERANCE:
            raise ValueError(f"Measurement-time budget exceeded: the queries need {need:g} (including {requests_needed} "
                             f"request(s) at {self.request_cost:g} each), {self.time_remaining:g} is left")
        if self._calls + len(queries) > self.max_calls:
            raise ValueError(f"Call limit of {self.max_calls} would be exceeded")
        if self.rounds_remaining is not None and requests_needed > self.rounds_remaining:
            raise ValueError(f"Round limit: this list needs {requests_needed} round(s) but {self.rounds_remaining} "
                             f"of {self.max_rounds} are left")
        for i in range(0, len(queries), size):
            body = {"queries": [{"wavelength": w, "time": t} for w, t in queries[i:i + size]]}
            chunk = self._server._request("POST", "/pixels", body, self._id)["results"]
            for result in chunk:
                self._record(result)
            self.rounds_used += 1
            results += chunk
        return results

    def submit(self, predicted) -> dict:
        values = _check_prediction(predicted)
        return self._server._request("POST", "/submit", {"predicted": values.tolist()}, self._id)

    def _record(self, result: dict) -> None:
        self.history.append(Measurement(result["wavelength"], result["measurement_time"], result["intensity"]))
        self.time_remaining = float(result["time_remaining"])
        self.rounds_remaining = result.get("rounds_remaining")
        self._calls += 1


def train_eval(server: Server, strategy, seeds=range(5)) -> float:
    """Run a strategy on several training seeds; prints the ratios and returns their geometric mean."""
    ratios = []
    print(f"{'seed':>5s} {'queries':>8s} {'rounds':>7s} {'ratio':>7s}")
    for seed in seeds:
        s = server.train_session(seed)
        ratios.append(s.submit(strategy(s))["ratio"])
        print(f"{seed:5d} {len(s.history):8d} {s.rounds_used:7d} {ratios[-1]:7.3f}")
    print(f"{'geometric mean':>22s} {_geomean(ratios):7.3f}")
    return _geomean(ratios)


def run_test(server: Server, strategy, name: str, class_code: str | None = None, preflight: bool = True) -> dict:
    """Run a strategy on every hidden spectrum and print the result table you hand in.

    Uses one of your limited attempts. A preflight run on a training spectrum comes first, so a bug in
    your strategy is caught before an attempt is used.
    """
    name = " ".join(str(name).split())
    if len(name) < 2 or name.lower() == "surname name":
        raise ValueError("set my_name to your own surname and name")

    if preflight:
        print("preflight: running your strategy once on a training spectrum ...")
        s = server.train_session(seed=0)
        ratio = s.submit(_check_prediction(strategy(s)))["ratio"]
        print(f"preflight passed (ratio {ratio:.3f} on training seed 0)\n")

    body = {"name": name}
    if class_code:
        body["class_code"] = class_code
    start = server._request("POST", "/test/start", body)
    n = start["n_spectra"]
    print(f"attempt {start['attempt']} of {start['max_attempts']}" + (" (resumed)" if start["resumed"] else ""))

    for i in range(1, n + 1):
        try:
            s = server._test_session(start["attempt_id"], i)
        except ServerError as exc:
            if exc.status == 409 and "already submitted" in exc.detail:
                print(f"  spectrum {i:2d}/{n}  already done")
                continue
            raise
        try:
            ratio = s.submit(_check_prediction(strategy(s)))["ratio"]
        except Exception:
            print(f"\nYour strategy failed on spectrum {i}. That spectrum stays open with {s.time_remaining:g} of "
                  f"{s.time_budget:g} time left. Fix the problem and run this cell again: the attempt resumes there.")
            raise
        rounds = s.rounds_used
        print(f"  spectrum {i:2d}/{n}  ratio {ratio:.3f}  ({len(s.history)} queries, {rounds} round{'' if rounds == 1 else 's'})")

    done = server._request("POST", "/test/finish", {"attempt_id": start["attempt_id"]})
    table, code = done["table"], done["code"]
    print(f"\n{table['name']}  (attempt {table['attempt']} of {start['max_attempts']})")
    for i, ratio in enumerate(table["ratios"], 1):
        print(f"  spectrum {i:2d}   ratio {ratio:.3f}")
    print(f"  geometric mean      {table['geomean']:.3f}")
    print(f"  verification code   {code}")
    print("SUBMISSION " + json.dumps({"table": table, "code": code}, sort_keys=True))
    return done
