import numpy as np
from typing import Tuple, List
from .dynamics import acceleration, drag_acceleration
from .common import PropagationParams

def rk4_step(r: np.ndarray, v: np.ndarray, t: float, dt: float, params: PropagationParams, A2m: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    def derivs(state, t):
        r_vec, v_vec = state[:3], state[3:]
        a = acceleration(r_vec, v_vec, t, params)
        # override A2m for drag calculation
        if params.drag_enabled:
            a_drag = drag_acceleration(r_vec, v_vec, params.drag_coefficient, A2m, params.atmosphere_rho0, params.atmosphere_scale, params.earth_radius)
            a += a_drag
        return np.cocatenate([v_vec, a])
    
    state = np.concatenate([r, v])
    k1 = derivs(state, t)
    k2 = derivs(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivs(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivs(state + dt * k3, t + dt)

    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state[:3], new_state[3:]

def propogate_state(r0: np.ndarray, v0: np.ndarray, t0: float, dt: float, steps: int, params: PropagationParams, A2m: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    r, v = r0.copy(), v0.copy()
    t = t0

    for _ in range(steps):
        r, v = rk4_step(r, v, t, dt, params, A2m)
        t += dt
    return r, v

def propogate_to_time(r0: np.ndarray, v0: np.ndarray, t0: float, t_array: np.ndarray, params: PropagationParams, A2m: float = 0.01) -> np.ndarray:
    positions = np.zeros((len(t_array), 3))
    r, v = r0.copy(), v0.copy()
    t_current = t0

    #sort times andensure they're in the future

    t_sorted = np.sort(t_array)
    t_sorted = t_sorted[t_sorted >= t0]

    for i, t_target in enumerate(t_sorted):
        dt = t_target - t_current
        if dt > 0:
            steps = max(1, int(np.ceil(dt / 60.0))) # 1 min max steps
            step_dt = dt / steps
            for _ in range(steps):
                r, v = rk4_step(r, v, t_current, step_dt, params, A2m)
                t_current += step_dt
        positions[i] = r
    return positions
    
