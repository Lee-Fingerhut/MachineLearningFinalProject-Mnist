import numpy as np

from sklearn.neural_network import MLPClassifier


def mlp_clf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_sizes: tuple = (64, 64, 64),
    activation: str = 'relu',
    solver: str = 'adam',
    alpha: float = 1e-4,
    batch_size: int = 32,
    learning_rate_init: float = 1e-3,
    max_iter: int = 50
) -> None:
    """
    classifies the test data based on training on the train data,
    using an SVM classifier.
    :param X_train: train features.
    :param X_test: test features.
    :param y_train: train labels.
    :param y_test: test labels.
    :param kernel: kernel method for SVM.
        {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’linear’.
    """
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,solver=solver,
                        alpha=alpha, batch_size=batch_size,
                        learning_rate_init=learning_rate_init,
                        max_iter=max_iter, shuffle=True, random_state=2021)
    mlp.fit(X_train, y_train)

    prediction = mlp.predict(X_train)
    train_accuracy = 100 * np.sum(prediction == y_train) / len(y_train)

    prediction = mlp.predict(X_test)
    test_accuracy = 100 * np.sum(prediction == y_test) / len(y_test)

    print('MLP: train accuracy = {:.2f}%, '
          'test accuracy = {:.2f}%'.format(train_accuracy, test_accuracy))