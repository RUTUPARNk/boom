import numpy as np
from sim.dynamics import acceleration
from sim.propagator import propagate_state
from sim.pc_estimate import monte_carlo_pc

def test_acceleration():
    """Test acceleration calculations"""
    r = np.array([6378137 + 500000, 0, 0])  # 500 km altitude
    v = np.array([0, 7660, 0])
    
    class MockParams:
        mu = 3.986004418e14
        earth_radius = 6378137.0
        j2_constant = 1.08262668e-3
        drag_coefficient = 2.2
        atmosphere_rho0 = 1.225
        atmosphere_scale = 72000.0
        j2_enabled = True
        drag_enabled = False
    
    accel = acceleration(r, v, 0, MockParams())
    assert accel.shape == (3,)
    print("Acceleration test passed")

def test_pc_estimation():
    """Test probability of collision estimation"""
    r_rel = np.array([100, 0, 0])
    P_rel = np.eye(3) * 50
    Rc = 5.0
    
    pc = monte_carlo_pc(r_rel, P_rel, Rc, n_samples=1000)
    assert 0 <= pc <= 1
    print(f"Pc estimation test passed: {pc:.4f}")

if __name__ == '__main__':
    test_acceleration()
    test_pc_estimation()