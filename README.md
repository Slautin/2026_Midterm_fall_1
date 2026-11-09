- `spectrum_client.py`: the client for the server, in one file (needs only numpy and requests).
- `spectrum_sim/`: the server's own simulator, to train offline. It is copied from the server repository; do not edit it here.
- `starter.ipynb`: a Colab-ready notebook built around a **surrogate** (data so far → estimate of the spectrum) and a **policy** (state → next stage of queries) joined by a loop. It has worked examples (a smoother, a peak fit, a two-stage policy), an offline comparison on the simulator, and the final test cell. It uses scipy and matplotlib, which Colab already has; elsewhere install them too.


## Students: the interface in one minute

A run has a time budget and a cap on queries. You decide how often to look at the results: one query at a time, or a whole stage in one request. **Every request costs a little of the same budget** (`session.request_cost`), however many queries it holds, so looking often lets you adapt sooner but leaves less time to measure.

```python
import spectrum_client as sc
server = sc.Server("https://...")
s = server.train_session(seed=7)                      # a training spectrum you can repeat
results = s.measure_many([(3.0, 1.0), (3.1, 1.0)])    # one stage of (wavelength, time) pairs = one request
s.time_remaining, s.request_cost, s.rounds_used       # what is left, what a request costs, requests so far
s.submit(prediction)                                  # 512 numbers, one per point of sc.E_GRID
sc.train_eval(server, my_strategy, seeds=range(5))    # try a strategy on several training seeds
sc.run_test(server, my_strategy, name="Surname Name") # the graded run on the hidden spectra
```

A strategy is a function `my_strategy(session)` that returns 512 numbers. In the starter it is two functions and a loop: the policy picks a stage, the session measures it, the surrogate learns from it, and the surrogate's last estimate is submitted.

```python
def my_strategy(session):
    return run(session, my_policy, my_surrogate)   # run, my_policy and my_surrogate are defined in starter.ipynb
```

## Train offline

```python
from spectrum_sim import SpectrumEnv, evaluate

env = SpectrumEnv()
obs = env.reset(seed=7)                               # the same spectrum and noise as server train seed 7
obs, reward, done, info = env.step([(3.0, 5.0), (3.2, 5.0)])   # one request: pays env.request_cost plus 10
ratios = evaluate(my_strategy, seeds=range(50))       # 50 training spectra, no network
```

The state is what you have measured plus the time left, an action is the next query or stage (each request pays the cost), and the reward is the final score. `env.step(queries, estimate=...)` also returns a dense reward from the true error, which only the local simulator can compute. Plan with dynamic programming, learn with RL, or design it by hand, then test on the server.
