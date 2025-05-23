import numpy as np

def find_nearest_face(weights_train, weights_test, paths, threshold=800.0):
    dists = np.linalg.norm(weights_train - weights_test, axis=0)
    idx = np.argmin(dists)
    return paths[idx], dists[idx], dists[idx] < threshold