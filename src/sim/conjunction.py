import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize_scalar
from .common import SatState, Conjunction, PropagationParams
from .propagator import propogate_state, propogate_to_time

def broad_phase_positions(sat_states: List[SatState], t_grid: np.ndarray, threshold_m: float, params: PropagationParams) -> List[Tuple[str, str, float, float]]:
    # Broad phase screening using simple position checks
    candidates = []

    for i, sat1 in enumerate(sat_states):
        for j, sat2 in enumerate(sat_states[i+1:], i+1):
            # checking positn are close at any time in t_grid
            for t in t_grid:
                r1 = propogate_state(sat1.r0, sat1.v0, sat1.epoch, np.array([t]), params, sat1.area_to_mass)[0]
                r2 = propogate_state(sat2.r0, sat2.v0, sat2.epoch, np.array([t]), params, sat2.area_to_mass)[0]
                distance = np.linalg.norm(r1 - r2)
                if distance < threshold_m:
                    # found potential conjunction, add time window 
                    candidates.append((sat1.sat_id, sat2.sat_id, max(t-300, t_grid[0]), min(t+300, t_grid[-1])))
                    break
    return candidates
def find_tca(satA: SatState, satB: SatState, t0:float, t1: float, params: PropagationParams, tol_s: float = 1.0) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # fidn time of closest approach using minimization
    def distance_squared(t):
        rA = propogate_to_time(satA.r0, satA.v0, satA.epoch, np.array([t]), params, satA.area_to_mass)[0]
        rB = propogate_to_time(satB.r0, satB.v0, satB.epoch, np.array([t]), params, satB.area_to_mass)[0]
        return np.sum((rA - rB) ** 2)
    result = minimize_scalar(distance_squared, bounds=(t0, t1), method='bounded', tol=tol_s)

    if result.success:
        tca = result.x
        miss_distance = np.sqrt(result.fun)
        rA = propogate_to_time(satA.r0, satA.v0, satA.epoch, np.array([tca]), params, satA.area_to_mass)[0]
        rB = propogate_to_time(satB.r0, satB.v0, satB.epoch, np.array([tca]), params, satB.area_to_mass)[0]

        dt = 1.0
        rA_next = propogate_to_time(satA.r0, satA.v0, satA.epoch, np.array([tca + dt]), params, satA.area_to_mass)[0]
        rB_next = propogate_to_time(satB.r0, satB.v0, satB.epoch, np.array([tca + dt]), params, satB.area_to_mass)[0]

        vA = (rA_next - rA) / dt
        vB = (rB_next - rB) / dt

        return tca, miss_distance, rA - rB, vA - vB
    raise ValueError(f"TCA minimization failed between {t0} and {t1} for {satA.sat_id} and {satB.sat_id}")