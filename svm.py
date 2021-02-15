import matplotlib.pyplot as plt
import numpy as np

from pca import reduce_pca_int
from sklearn.svm import SVC
from tsne import reduce_tsne_int
from utils import create_2d_meshgrid


def svm_clf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel: str = 'linear'
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
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

    prediction = svm.predict(X_train)
    train_accuracy = 100 * np.sum(prediction == y_train) / len(y_train)

    prediction = svm.predict(X_test)
    test_accuracy = 100 * np.sum(prediction == y_test) / len(y_test)

    print('SVM: train accuracy = {:.2f}%, '
          'test accuracy = {:.2f}%'.format(train_accuracy, test_accuracy))


def reduced_svm_clf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    method: str = 'pca',
    ndims: int = 2,
    kernel: str = 'linear',
    save_as: str = ''
) -> None:
    """
    classifies the test data based on training on the train data,
    using an SVM classifier.
    :param X_train: train features.
    :param X_test: test features.
    :param y_train: train labels.
    :param y_test: test labels.
    :param method: dimensionality-reduction method {'pca', 'tsne'}.
    :param ndims: number of dims in reduced dimension.
    :param kernel: kernel method for SVM.
        {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’linear’.
    :param save_as: save plot to path.
    """
    if method == 'pca':
        X_train = reduce_pca_int(X_train, ndims)
        X_test = reduce_pca_int(X_test, ndims)
    elif method == 'tsne':
        X_train = reduce_tsne_int(X_train, ndims)
        X_test = reduce_tsne_int(X_test, ndims)
    else:
        raise ValueError(method)

    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

    prediction = svm.predict(X_train)
    train_accuracy = 100 * np.sum(prediction == y_train) / len(y_train)

    prediction = svm.predict(X_test)
    test_accuracy = 100 * np.sum(prediction == y_test) / len(y_test)

    if ndims == 2:
        X, Y = create_2d_meshgrid(X_train[:, 0], X_train[:, 1], samples=100)
        Z = svm.predict(np.c_[X.ravel(), Y.ravel()])
        Z = Z.reshape(X.shape)

        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].contourf(X, Y, Z, cmap=plt.cm.Spectral,  alpha=0.8)
        s = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, s=20, edgecolors='k')
        legend = axs[0].legend(*s.legend_elements(), title='digits', loc='best')
        axs[0].add_artist(legend)
        axs[0].set_xticks(())
        axs[0].set_yticks(())
        axs[0].set_title('SVM on trainset. accuracy {:.2f}%'.format(train_accuracy))

        axs[1].contourf(X, Y, Z, cmap=plt.cm.Spectral, alpha=0.8)
        s = axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral, s=20, edgecolors='k')
        legend = axs[1].legend(*s.legend_elements(), title='digits', loc='best')
        axs[1].add_artist(legend)
        axs[1].set_xticks(())
        axs[1].set_yticks(())
        axs[1].set_title('SVM on testset. accuracy {:.2f}%'.format(test_accuracy))

        if save_as != '':
            plt.savefig(save_as)
        plt.show()

    else:
        print(method + ' (dim={}) + SVM, train accuracy = {:.2f}%, '
              'test accuracy = {:.2f}%'.format(ndims, train_accuracy, test_accuracy))
