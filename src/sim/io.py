import pandas as pd
import json
import numpy as np
from typing import List
from .common import SatState

def load_sats_csv(path: str) -> List[SatState]:
    # loading part
    df = pd.read_csv(path)
    satellites = []

    for _, row in df.iterrows():
        from datetime import datetime
        epoch_dt = datetime.fromisoformat(row['epoch_iso'].replace('Z', '+00:00'))
        ref_dt = datetime(2025, 1, 1, 0, 0, 0) # reference epoch
        epoch_sec = (epoch_dt.replace(tzinfo=None) - ref_dt).total_seconds()

        # parsing covariance matrix
        cov_matrix = np.array(json.loads(row['P_cov_json']))

        sat = SatState(
            sat_id = row['sat_id'],
            epoch = epoch_sec,
            r0 = [row['x_m'], row['y_m'], row['z_m']],
            v0 = [row['vx_mps'], row['vy_mps'], row['vz_mps']],
            radius=row['r_m'],
            area_to_mass=row.get('a2m', 0.01),
            cov_json=row['P_cov_json']
        )
        satellites.append(sat)
    return satellites

def save_alerts(alerts: List[dict], path: str):
    if alerts:
        df = pd.DataFrame(alerts)
        df.to_csv(path, index=False)

def save_conjunctions(conjunctions: List[dict], path: str):
    data = []
    for conj in conjunctions:
        data.append({
            'sat1': conj.sat1,
            'sat2': conj.sat2,
            'tca': conj.tca,
            'miss_distance_m': conj.miss_distance,
            'pc': conj.pc
        })
    if data:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)