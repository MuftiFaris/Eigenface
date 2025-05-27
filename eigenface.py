import numpy as np
from utils import power_iteration

def compute_eigenfaces(A, max_components=20):

    mean_face = np.mean(A, axis=1, keepdims=True)
    B = A - mean_face

    L = B.T @ B

    eigenvalues, eigenvectors = [], []
    Lc = L.copy()
    for _ in range(min(max_components, L.shape[0])):
        val, vec = power_iteration(Lc)
        if val <= 0: break
        eigenvalues.append(val)
        eigenvectors.append(vec)
        Lc -= val * np.outer(vec, vec)
    eigenvectors = np.array(eigenvectors)

    U = np.zeros((A.shape[0], eigenvectors.shape[0]))
    for i, v in enumerate(eigenvectors):
        u = B @ v
        U[:,i] = u / np.linalg.norm(u)

    weights = U.T @ B
    
    return mean_face, U, weights, eigenvalues