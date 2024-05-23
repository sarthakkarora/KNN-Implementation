import numpy as np
from statistics import mode

class KNN:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
    
    def euclidean_distance(self, vec1, vec2):
        return np.sum((vec1 - vec2) ** 2)

    def get_neighbourhood(self, X_train, y_train, point, K):
        distances = []
        for x_train, y in zip(X_train, y_train):
            distance = self.euclidean_distance(point, x_train)
            distances.append((distance, y))
        distances.sort(key=lambda x: x[0])
        neighbours = [y for _, y in distances[:K]]
        return mode(neighbours)
  
    def get_accuracy(self, pred, y_test):
        correct = sum(p == t for p, t in zip(pred, y_test))
        return correct / len(pred)
    
    def knn_main_code(self, X_train, X_test, y_train, K):
        pred = [self.get_neighbourhood(X_train, y_train, x, K) for x in X_test]
        return pred

# Example usage:
# Assuming train_set and test_set are pandas DataFrames with the last column being the labels

def prepare_data(dataframe):
    X = dataframe.iloc[:, :-1].values  # features
    y = dataframe.iloc[:, -1].values  # labels
    return X, y

# train_set and test_set should be your pandas DataFrame containing the training and testing data respectively
# knn = KNN(train_set, test_set)
# X_train, y_train = prepare_data(train_set)
# X_test, y_test = prepare_data(test_set)

# K = 3
# predictions = knn.knn_main_code(X_train, X_test, y_train, K)
# accuracy = knn.get_accuracy(predictions, y_test)
# print(f'Accuracy: {accuracy}')
