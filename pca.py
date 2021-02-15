import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


def reduce_pca_int(X: np.ndarray, ndim: int = 2) -> np.ndarray:
    pca = PCA(n_components=ndim)
    return pca.fit_transform(X)


def reduce_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_as: str = ''
) -> None:
    X_train = reduce_pca_int(X_train)
    X_test = reduce_pca_int(X_test)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    s = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                cmap=plt.cm.Spectral, s=20, edgecolors='k')
    legend = axs[0].legend(*s.legend_elements(), title='digits', loc='best')
    axs[0].add_artist(legend)
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    axs[0].set_title('trainset projected on $\mathbb{R}^2$ using PCA')

    s = axs[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                cmap=plt.cm.Spectral, s=20, edgecolors='k')
    legend = axs[1].legend(*s.legend_elements(), title='digits', loc='best')
    axs[1].add_artist(legend)
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    axs[1].set_title('testset projected on $\mathbb{R}^2$ using PCA')

    if save_as != '':
        plt.savefig(save_as)
    plt.show()
