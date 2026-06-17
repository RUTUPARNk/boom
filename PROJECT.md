# boom — the story

This is where the satellite obsession got serious.

The trilogy: [`moon`](../moon/PROJECT.md) tracked where satellites *are*.
[`stone`](../stone/PROJECT.md) asked, in the browser, *will they hit?* **`boom` is
the answer done for real** — not a pretty globe, but the actual astrodynamics that
satellite operators run before deciding to fire a thruster and dodge.

The name says it: moon → stone → **boom**. The collision.

And he didn't fake the math. boom implements genuine, textbook-correct conjunction
assessment:
- Orbit **propagation with J2** (Earth's equatorial bulge) and **atmospheric
  drag** — the two perturbations that actually matter in low Earth orbit.
- **Time of closest approach** found by numerical optimization, not eyeballing.
- **Probability of collision** computed *two* ways — a **Monte Carlo** estimate
  (sampling the relative-position covariance through a Cholesky factorization) and
  the **Foster 1992** analytic method, which is the canonical approach used in real
  operational conjunction screening (NASA/CARA and friends).
- Unit tests on the TCA and Pc paths.

That's the tell. A lot of his projects are big and fast and broad; **this one is
deep and correct.** He went and learned the real aerospace method — covariance,
hard-body radius, Foster's 2D projection — and implemented it properly, with tests,
for the satisfaction of getting it *right*. The wonder of looking up at moon, taken
all the way down to the equations underneath.

## What it is

`src/sim/` — `propagator` + `dynamics` (J2 + drag), `conjunction` (broad-phase +
TCA), `pc_estimate` (Monte Carlo + Foster 1992), `io`, `common`; a `cli.py` driver
over `data/sats.csv` and `config/defaults.yaml`; `tests/test_tca_pc.py`.

## Cleanup done in this pass (2026)

- Replaced the README (which was just a pasted console-output dump) with a real
  description of the pipeline and the methods. Repo otherwise clean.
