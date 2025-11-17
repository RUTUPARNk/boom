from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class SatState:
    sat_id: str
    epoch: float # seconds since reference
    r0: np.ndarray  # initial positn [x,y,z]
    v0: np.ndarray # initial velocity [vx,vy,vz]
    radius: float # collision radius
    area_to_mass: float # area to mass ratio
    cov_json: str # covariance as json format

    def __post_init__(self):
        self.r0 = np.array(self.r0, dtype=np.float64)
        self.v0 = np.array(self.v0, dtype=np.float64)

@dataclass
class Conjunction:
    sat1: str
    sat2: str
    tca: float
    miss_distance: float
    pc: float
    r_rel: np.ndarray
    v_rel: np.ndarray

@dataclass
class PropagationParams:
    mu: float
    earth_radius: float
    j2_constant: float
    drag_coefficient: float
    atmosphere_rho0: float
    atmosphere_scale: float
    j2_enabled: bool
    drag_enabled: bool
