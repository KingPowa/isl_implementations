"""
Core class for the models
"""
import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Internally fit the model

        Args:
            x (np.ndarray): training samples
            y (np.ndarray): training labels
        """
        raise NotImplementedError
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the labels

        Args:
            x (np.ndarray): testing samples

        Returns:
            np.ndarray: y predicted labels
        """
        raise NotImplementedError
    