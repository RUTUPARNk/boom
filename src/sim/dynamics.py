import numpy as np
from numba import jit
from .common import PropagationParams

@jit(nopython=True)
def two_body_acceleration(r: np.ndarray, mu: float) -> np.ndarray:
    # gravitational acceleration due to two-body problem
    r_norm = np.linalg.norm(r)
    return -mu * r / (r_norm ** 3)

@jit(nopython=True)
def j2_acceleration(r: np.ndarray, mu: float, earth_radius: float, j2_constant: float) -> np.ndarray:
    x, y, z = r
    r_norm = np.linalg.norm(r)

    factor = 1.5 * j2_constant * mu * (earth_radius ** 2) / (r_norm ** 5)
    z_factor = 5 * (z ** 2) / (r_norm ** 2)

    ax = factor * x * (z_factor - 1)
    ay = factor * y * (z_factor - 1)
    az = factor * z * (z_factor - 3)
    return np.array([ax, ay, az])

@jit(nopython=True)
def drag_acceleration(r: np.ndarray, v: np.ndarray, Cd: float, A2_m: float, rho0: float, H: float, Re: float) -> np.ndarray:
    # Simple exponential atmosphere model for drag
    r_norm = np.linalg.norm(r)
    altitude = r_norm - Re

    if altitude < 0:
        return np.zeros(3)

    # exponential atmosphere model
    rho = rho0 * np.exp(-altitude / H)
    v_norm = np.linalg.norm(v)

    if v_norm == 0:
        return np.zeros(3)
    
    drag_magnitude = 0.5 *Cd * A2_m * rho * (v_norm ** 2)
    return -drag_magnitude * (v / v_norm)

def acceleration(r: np.ndarray, v: np.ndarray, t: float, params: PropagationParams) -> np.ndarray:
    # returns acceleration vector (m/s^2) = two body + J2 + drag
    accel = two_body_acceleration(r, params.mu)
    if params.j2_enabled:
        accel += j2_acceleration(r, params.mu, params.earth_radius, params.j2_constant)

    if params.drag_enabled:
        accel += drag_acceleration(r, v, params.drag_coefficient, 0.01, # A2m will be overridden by satellite specific value
                                    params.atmosphere_rho0, params.atmoshpere_scale, params.earth_radius)
    return accel
    
    