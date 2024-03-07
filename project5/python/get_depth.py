import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """

    if t1.ndim == 1:
        t1 = t1[:, np.newaxis]
    if t2.ndim == 1:
        t2 = t2[:, np.newaxis]
    E1 = np.hstack((R1, t1))
    E2 = np.hstack((R2, t2))
    c1 = -np.linalg.inv(E1[:, :3]) @ E1[:, 3]
    c2 = -np.linalg.inv(E2[:, :3]) @ E2[:, 3]
    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]
    depthM = np.divide(b * f, dispM, out=np.zeros_like(dispM), where=dispM!=0)
    
    return depthM

