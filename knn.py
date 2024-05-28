import numpy as np
from statistics import mode
from typing import List, Tuple
import pandas as pd

class KNN:
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        self.X_train, self.y_train = self.prepare_data(train_set)
        self.X_test, self.y_test = self.prepare_data(test_set)

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    @staticmethod
    def prepare_data(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = dataframe.iloc[:, :-1].values  # features
        y = dataframe.iloc[:, -1].values  # labels
        return X, y

    def get_neighbourhood(self, point: np.ndarray, K: int) -> int:
        distances = [(self.euclidean_distance(point, x_train), y) 
                     for x_train, y in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda x: x[0])
        neighbours = [y for _, y in distances[:K]]
        return mode(neighbours)

    def get_accuracy(self, predictions: List[int]) -> float:
        correct = sum(p == t for p, t in zip(predictions, self.y_test))
        return correct / len(predictions)

    def knn_main_code(self, K: int) -> List[int]:
        if K <= 0:
            raise ValueError("K must be a positive integer")
        if K > len(self.X_train):
            raise ValueError("K cannot be greater than the number of training samples")
        
        predictions = [self.get_neighbourhood(x, K) for x in self.X_test]
        return predictions

# Example usage:
# Assuming train_set and test_set are pandas DataFrames with the last column being the labels

def main(train_set: pd.DataFrame, test_set: pd.DataFrame, K: int):
    knn = KNN(train_set, test_set)
    predictions = knn.knn_main_code(K)
    accuracy = knn.get_accuracy(predictions)
    print(f'Accuracy: {accuracy}')

# Uncomment and modify the following lines to use your own data
# train_set = pd.read_csv('path_to_train_set.csv')
# test_set = pd.read_csv('path_to_test_set.csv')
# K = 3
# main(train_set, test_set, K)
