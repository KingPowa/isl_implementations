"""
Implementing KNN in numpy based on ISL mathematical description
"""
from typing import Callable
import numpy as np

from .core import Model
from .utils.distances import euclidean_distance

class KNN(Model):

    def __init__(self, 
                 k: int = 1,
                 distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
                 ):
        super().__init__()
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, x, y):
        """Fit the KNN model for a task T.
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """Predict the labels based on the nearest neighboor
        """
        raise NotImplementedError
    
class KNNClassification(KNN):

    def fit(self, x, y):
        super().fit(x, y)
        self.classes = np.unique(y)
        self.class_masks = [np.where(self.x_train == c) for c in self.classes]

    def predict(self, x):
        distance_matrix = self.distance_fn(self.x_train, x)
        # Select the closest points
        partitioned_indices = np.argpartition(distance_matrix, self.k, axis=0)[:self.k, :]
        # Matrix is (self.k, M). We sort the indices based on the value
        sorted_indices = np.argsort(-distance_matrix[partitioned_indices, np.arange(distance_matrix.shape[1])], axis=0)
        # Matrix is (self.k, M), now get ordered indices
        top_k_indices = partitioned_indices[sorted_indices, np.arange(distance_matrix.shape[1])]
        # Get labels from train
        labels_train = self.y_train[top_k_indices]
        # Return majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=labels_train)
        
