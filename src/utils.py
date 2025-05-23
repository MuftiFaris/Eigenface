import numpy as np

def power_iteration(matrix, tol=1e-6, max_iter=1000):
    b = np.random.rand(matrix.shape[1])
    b /= np.linalg.norm(b)
    for _ in range(max_iter):
        b1 = matrix @ b
        n1 = np.linalg.norm(b1)
        if n1 == 0: break
        b1 /= n1
        if np.linalg.norm(b1 - b) < tol:
            break
        b = b1
    eigenvalue = b @ (matrix @ b)
    return eigenvalue, b