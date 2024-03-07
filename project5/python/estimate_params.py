import numpy as np
from scipy.linalg import svd, qr, rq

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    # K, R, t = None, None, None
    U, S, Vt = np.linalg.svd(P)
    c = Vt[-1]
    c = c / c[-1]
    M = P[:, :3]
    K, R = rq(M)
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)
    R = np.dot(T, R) 
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        K[:, 2] *= -1
    t = -np.dot(R, c[:3])

    return K, R, t

