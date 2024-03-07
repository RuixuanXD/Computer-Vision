import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE
    # t1.reshape(-1, 1)
    # t2.reshape(-1, 1)
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    r1 = (c1 - c2) / np.sqrt(np.sum((c1 - c2) ** 2))
    r2 = np.cross(R1[2, :], r1)
    r3 = np.cross(r1, r2)
    K = K2
    R = np.vstack([-r1, -r2, r3])
    t1n = -R @ c1
    t2n = -R @ c2

    M1 = K @ R @ np.linalg.inv(K1 @ R1)
    M2 = K @ R @ np.linalg.inv(K2 @ R2)

    K1n = K1
    K2n = K2
    R1n = R
    R2n = R
    # M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

