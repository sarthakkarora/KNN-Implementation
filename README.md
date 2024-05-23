# KNN-Implementation

**Concept:**

KNN is a fundamental supervised learning algorithm used for classification and, in some cases, regression tasks. It's a non-parametric method, meaning it doesn't make assumptions about the underlying data distribution.

**Core principle:**

The algorithm assumes that similar data points tend to have similar labels (classification) or values (regression).
To make a prediction for a new data point, KNN identifies the K nearest neighbors in the training data based on a distance metric (e.g., Euclidean distance).
In classification tasks, the predicted label is the most frequent class among the K nearest neighbors.


**Imports:**

* `csv`: for reading CSV files (possibly for loading data)
* `random`: for generating random numbers (might be used for data manipulation)
* `math`: for mathematical functions (likely used in the distance calculation)
* `operator`: for mathematical operations (might be used for sorting)
* `statistics`: for statistical functions (used to find the mode)
* `pandas` (pd): for data manipulation and analysis (powerful library for working with data)
* `numpy` (np): for numerical computations (another powerful library for numerical operations)

**Class Definition: KNN**

* This code defines a class named `KNN`.
* The class takes two arguments in its constructor (`__init__` function):
    * `train_set`: The training data set used to train the KNN model.
    * `test_set`: The testing data set used to evaluate the model's performance.

**Methods:**

* **euclidean_distance(self, vec1, vec2):**
    * This function calculates the Euclidean distance between two vectors (data points).
    * It iterates through each dimension (index) of the vectors and squares the difference between corresponding values.
    * The squares are summed, and the square root is not explicitly calculated (follows the mathematical formula for Euclidean distance).
* **get_neighbourhood(self, X_train, y_train, point, K):**
    * This function finds the K nearest neighbors for a given data point (`point`) in the training set (`X_train`).
    * It uses the `zip` function to combine the training data (`X_train`) and labels (`y_train`) into pairs.
    * It sorts the pairs based on the Euclidean distance between the `point` and the first element (data point) of each pair.
    * It selects the top `K` closest neighbors using slicing and returns the most frequent label (mode) among those neighbors using the `mode` function.
* **get_accuracy(self, pred, y_test):**
    * This function calculates the accuracy of the KNN model's predictions.
    * It iterates through the predicted labels (`pred`) and the actual labels (`y_test`).
    * It counts the number of correct predictions (where the predicted label matches the actual label).
    * It divides the number of correct predictions by the total number of predictions and returns the accuracy as a float.
* **knn_main_code(self, X_train, X_test, y_train, K):**
    * This function is the core of the KNN algorithm.
    * It takes the training data (`X_train`), testing data (`X_test`), training labels (`y_train`), and the number of neighbors (`K`) as arguments.
    * It initializes an empty list `pred` to store the predicted labels.
    * It iterates through each data point in the testing set (`X_test`).
    * For each data point, it calls the `get_neighbourhood` function to find the K nearest neighbors in the training set.
    * It appends the predicted label (mode of the neighbors' labels) for each test data point to the `pred` list.
    * Finally, it returns the list of predicted labels.

**Overall Functionality:**

This code implements a KNN classifier that can be used for classification tasks. It trains the model on the provided training data and then uses it to predict the labels for unseen data points in the testing set. The `get_accuracy` function allows you to evaluate the model's performance on the testing data.

**Note:**

* This code assumes the data is already preprocessed and formatted appropriately for the KNN algorithm.
* Additional functionalities like data loading, preprocessing, and hyperparameter tuning (finding the optimal value for K) might be needed for practical applications.
