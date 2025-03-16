import numpy as np

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Measure the euclidean distance of multidimensional points

    Args:
        x1 (np.ndarray): An array of dimension BxD (number of points, number of features)
        x2 (np.ndarray): Same as x1

    Returns:
        np.ndarray: An array of dimensions BxB, containing the distances between the points
    """
    return np.linalg.norm(x1[:, np.newaxis, :] - x2[np.newaxis, :, :], ord=2, axis=2)