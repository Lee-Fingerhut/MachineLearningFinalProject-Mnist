import matplotlib.pyplot as plt
import numpy as np

from pca import reduce_pca_int

from sklearn.linear_model import LogisticRegression
from tsne import reduce_tsne_int
from utils import create_2d_meshgrid


def lr_clf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    classifies the test data based on training on the train data,
    using a Linear Regression classifier.
    :param X_train: train features.
    :param X_test: test features.
    :param y_train: train labels.
    :param y_test: test labels.
    """

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    prediction = lr.predict(X_train)
    train_accuracy = 100 * np.sum(prediction == y_train) / len(y_train)

    prediction = lr.predict(X_test)
    test_accuracy = 100 * np.sum(prediction == y_test) / len(y_test)

    print('Logistic Regressor: train accuracy = {:.2f}%, '
            'test accuracy = {:.2f}%'.format(train_accuracy, test_accuracy))
