import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'

    n = pts1.shape[0]
    pts1 = pts1 / M
    pts2 = pts2 / M

    A = np.zeros((n,9))
    for i in range(n):
        x1,y1 = pts1[i]
        x2,y2 = pts2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2,1]

    U, S, Vt = svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = svd(F)
    S[-1] = 0 
    F = U @ (np.diag(S) @ Vt)
    F = refineF(F,pts1, pts2)

    scale = np.diag([1/M, 1/M, 1])
    F = scale.T @ F @ scale
    return F


# def normalize(p):
#     cen = np.mean(p, axis=0)
#     translated = p - cen

#     scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(translated**2, axis=1)))
#     transfor = np.array([[scale, 0, -scale * cen[0]],
#                                       [0, scale, -scale * cen[1]],
#                                       [0, 0, 1]])

#     normal = np.dot(transfor, np.vstack((p.T, np.ones(p.shape[0]))))

#     return normal[:2].T, transfor

