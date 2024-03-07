import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    N = x.shape[1]
    A = []
    for i in range(N):
        Xx, Xy, Xz = X[:, i]
        xx, xy = x[:, i]
        A.append([Xx, Xy, Xz, 1, 0, 0, 0, 0, -xx*Xx, -xx*Xy, -xx*Xz, -xx])
        A.append([0, 0, 0, 0, Xx, Xy, Xz, 1, -xy*Xx, -xy*Xy, -xy*Xz, -xy])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    P = Vh[-1].reshape(3, 4)
    return P 
