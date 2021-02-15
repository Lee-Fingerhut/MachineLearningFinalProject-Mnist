import matplotlib.pyplot as plt
import numpy as np
import utils

from cnn import cnn_clf
from knn import knn_clf, reduced_knn_clf
from logistic_regression import lr_clf
from mlp import mlp_clf
from pca import reduce_pca
from svm import svm_clf, reduced_svm_clf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tsne import reduce_tsne
from utils import normalize


if __name__ == '__main__':
    batch_size = 64
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5, 0.5)
    ])
    mnist_trainset = MNIST(root='./traindata', train=True, download=True, transform=transforms)
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    mnist_testset = MNIST(root='./testdata', train=False, download=True, transform=transforms)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True, num_workers=0)

    im_shape = mnist_trainset.data.data[0].shape

    _, axs = plt.subplots(8, 8, figsize=(10, 10))
    for sample in train_loader:
        X, y = sample
        for i in range(batch_size):
            im = X[i, ...].reshape(im_shape)
            row = i // 8
            col = i % 8
            axs[row, col].imshow(im, cmap='gray')
            axs[row, col].set_title(y[i].item(), fontsize=20)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
        break
    plt.suptitle('samples from the MNIST dataset', fontsize=20)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
    plt.show()

    # CNN CLASSIFIER
    cnn_clf(train_loader, test_loader)

    n = np.prod(im_shape)
    X_train = utils.torch_to_numpy(mnist_trainset.data.data).reshape((len(mnist_trainset), n))
    y_train = utils.torch_to_numpy(mnist_trainset.targets.data).reshape((len(mnist_trainset),))
    X_test = utils.torch_to_numpy(mnist_testset.data.data).reshape((len(mnist_testset), n))
    y_test = utils.torch_to_numpy(mnist_testset.targets.data).reshape((len(mnist_testset),))

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # MLP CLASSIFIER
    mlp_clf(X_train, X_test, y_train, y_test)

    # Linear Regression CLASSIFIER
    lr_clf(X_train, X_test, y_train, y_test)

    # from here and on we sample 10% of the data and run the algorithms
    num_train = len(mnist_trainset) // 10
    num_test = len(mnist_testset) // 10
    train_indices = np.random.randint(0, len(mnist_trainset), num_train)
    test_indices = np.random.randint(0, len(mnist_testset), num_test)

    X_train = X_train[train_indices, :]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices, :]
    y_test = y_test[test_indices]

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(y_train, bins=np.arange(0, 11), rwidth=0.7, align='left')
    axs[0].set_xticks(np.arange(0, 11))
    axs[0].set_title('trainset after sampling')
    axs[1].hist(y_test, bins=np.arange(0, 11), rwidth=0.7, align='left')
    axs[1].set_xticks(np.arange(0, 11))
    axs[1].set_title('testset after sampling')
    plt.show()

    # KNN CLASSIFIER
    knn_clf(X_train, X_test, y_train, y_test)

    # SVM CLASSIFIER
    svm_clf(X_train, X_test, y_train, y_test)

    # PCA
    reduce_pca(X_train, X_test, y_train, y_test)

    # TSNE
    reduce_tsne(X_train, X_test, y_train, y_test)

    # Reduce dimension and classify using KNN
    reduced_knn_clf(X_train, X_test, y_train, y_test, method='pca', neigh=3, ndims=2)

    reduced_knn_clf(X_train, X_test, y_train, y_test, method='tsne', neigh=3, ndims=2)

    # Reduce dimension and classify using SVM
    reduced_svm_clf(X_train, X_test, y_train, y_test, method='pca', ndims=2)
