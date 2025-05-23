import numpy as np
from utils import power_iteration

def compute_eigenfaces(A, max_components=20):

    # 1. Mean face dan data terpusat
    mean_face = np.mean(A, axis=1, keepdims=True)
    B = A - mean_face

    # 2. Matriks kecil L
    L = B.T @ B

    # 3. Power iteration + deflation
    eigenvalues, eigenvectors = [], []
    Lc = L.copy()
    for _ in range(min(max_components, L.shape[0])):
        val, vec = power_iteration(Lc)
        if val <= 0: break
        eigenvalues.append(val)
        eigenvectors.append(vec)
        Lc -= val * np.outer(vec, vec)
    eigenvectors = np.array(eigenvectors)

    # 4. Bangun eigenfaces
    U = np.zeros((A.shape[0], eigenvectors.shape[0]))
    for i, v in enumerate(eigenvectors):
        u = B @ v
        U[:,i] = u / np.linalg.norm(u)

    # 5. Project dataset
    weights = U.T @ B
    
    # Kembalikan juga eigenvalues untuk analisis
    return mean_face, U, weights, eigenvalues