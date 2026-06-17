# boom

A from-scratch **satellite conjunction-assessment engine** in Python — the kind of
math real operators use to decide whether to dodge. Propagate orbits, find when two
objects are closest, and estimate the probability they collide.

## Pipeline
1. **Propagate** (`propagator.py`, `dynamics.py`) — orbital motion with **J2**
   gravitational perturbation and **atmospheric drag**.
2. **Broad-phase screening** (`conjunction.py`) — find candidate close-approach
   pairs over a time window.
3. **TCA** — refine each candidate's *time of closest approach* with
   `scipy.optimize.minimize_scalar`.
4. **Pc — probability of collision** (`pc_estimate.py`), two ways:
   - **Monte Carlo** — sample the relative position covariance (via Cholesky) and
     count how many draws fall inside the combined hard-body radius.
   - **Foster 1992** — the classic 2D-projection analytic method used in
     operational conjunction assessment.
5. **Output** — per-pair alerts (TCA, miss distance, Pc) to `alerts.csv`.

## Run
```bash
pip install -r requirements.txt
python -m src.cli            # uses data/sats.csv + config/defaults.yaml
```

Tests: `pytest src/tests/test_tca_pc.py` (covers TCA + Pc).
