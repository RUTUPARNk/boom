import argparse
import yaml
import numpy as np
from datetime import datetime, timedelta
from typing import List
from sim.io import load_sats_csv, save_alerts, save_conjunctions
from sim.common import PropagationParams
from sim.conjunction import broad_phase_positions, find_tca
from sim.pc_estimate import monte_carlo_pc

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config_path: str):
    config = load_config(config_path)

    # Load params
    params = PropagationParams(
        mu = config['mu'],
        earth_radius = config['earth_radius'],
        j2_constant = config['j2_constant'],
        drag_coefficient = config['drag_coefficient'],
        atmosphere_rho0 = config['atmosphere_rho0'],
        atmosphere_scale = config['atmosphere_scale'],
        j2_enabled = config['j2_enabled'],
        drag_enabled = config['drag_enabled']
    )
    # Load satellites
    sats = load_sats_csv('data/sats.csv')

    # create time grid for broad phase

    time_window = config['time_window_seconds'] * 3600
    t_grid = np.arange(0, time_window, config['broad_phase_dt_seconds'])

    print(f"Running broad phase propagation for {len(sats)} satellites...")
    candidates = broad_phase_positions(sats, t_grid, config['miss_distance_alert_m']* 5, params)
    print(f"Found {len(candidates)} candidate conjunctions in broad phase.")

    # Narrow phase processing
    alerts = []
    conjunctions = []
    for sat1_id, sat2_id, t0, t1 in candidates:
        sat1 = next(s for s in sats if s.sat_id == sat1_id)
        sat2 = next(s for s in sats if s.sat_id == sat2_id)

        try:
            tca, miss_distance, r_rel, v_rel = find_tca(sat1, sat2, t0, t1, params)
            if miss_distance < config['miss_distance_alert_m']:
                # calculate probability of collision
                # for now, using covariance should propogate covariance
                P_rel = np.eye(3) * 100
                pc = monte_carlo_pc(r_rel, P_rel, sat1.radius + sat2.radius, config['mc_samples'])
                if pc > config['pc_threshold']:
                    alert = {
                        'sat1': sat1_id,
                        'sat2': sat2_id,
                        'tca': tca,
                        'miss_distance_m': miss_distance,
                        'pc': pc,
                        'alert_level': 'HIGH' if pc > 1e-3 else 'MEDIUM'
                    }
                    alerts.append(alert)
                    print(f"ALERT: {sat1_id}-{sat2_id} | TCA: {tca} | Miss Dist: {miss_distance:.1f} m | Pc: {pc:.2e}")
        except Exception as e:
            print(f"Error processing conjunction between {sat1_id} and {sat2_id}: {e}")
    # save results
    save_alerts(alerts, 'alerts.csv')
    print(f"saved {len(alerts)} alerts to alerts.csv")

def main():
    parser = argparse.ArgumentParser(description='Collision Simulation CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)
    run_parser = subparsers.add_parser('run', help='Run full simulation')
    run_parser.add_argument('--config', default='config/defaults.yaml', help='Path to config YAML file')
    pair_parser = subparsers.add_parser('simulate-pair', help='Simulate specific pair')
    pair_parser.add_argument('--sat1', required=True, help='First satellite ID')
    pair_parser.add_argument('--sat2', required=True, help='Second satellite ID')
    pair_parser.add_argument('--window', type=float, default=3600, help='Time window in seconds')

    # list candidates command
    candidates_parser = subparsers.add_parser('list-candidates', help='List conjunction candidates')
    candidates_parser.add_argument('--start', type=float, default=0, help='Start time (s)')
    candidates_parser.add_argument('--end', type=float, default=172800, help='End time (s)')

    args =parser.parse_args()
    if args.command == 'run':
        run_simulation(args.config)
    elif args.command == 'simulate-pair':
        print("Simulate pair {args.sat1}-{args.sat2}")
    elif args.command == 'list-candidates':
        print("List candidates from {args.start}s to {args.end}s")

if __name__ == '__main__':
    main()