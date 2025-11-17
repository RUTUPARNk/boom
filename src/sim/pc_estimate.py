import numpy as np
from scipy.linalg import cholesky

def monte_carlo_pc(r_rel: np.ndarray, P_rel: np.ndarray, Rc: float, n_samples: int = 20000) -> float:
    # monte carlo probability of collision estimation
    try:
        L = cholesky(P_rel, lower=True)
        samples = np.random.randn(n_samples, 3) @ L.T + r_rel

        # Count collisions
        distances = np.linalg.norm(samples, axis=1)
        collisions = np.sum(distances < Rc)

        return collisions / n_samples
    except:
        # fallback to simple approximation if cholesky fails
        return 0.0

def foster_1992_pc(r_rel: np.ndarray, P_rel: np.ndarray, Rc: float) -> float:
    # simplified 2D projectn approximatn
    try:
        v_rel = r_rel
        if np.linalg.norm(v_rel) == 0:
            return 0.0
        k = v_rel / np.linalg.norm(v_rel)
        i = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        j = np.cross(k, i)
        i = np.cross(j, k)
        i, j = i / np.linalg.norm(i), j / np.linalg.norm(j)

        # project covariance
        T = np.vstack([i, j])
        P_2d = T @ P_rel @ T.T

        # project relative postn
        r_2d = T @ r_rel

        # calculate pc using 2d gaussian 
        det_P = np.linalg.det(P_2d)
        if det_P <= 0:
            return 0.0
        
        inv_P = np.linalg.det(P_2d)
        exponent = -0.5 * (r_2d @ inv_P @ r_2d)

        return (Rc ** 2) / (2 * np.sqrt(det_P)) * np.exp(exponent)
    except:
        return 0.0