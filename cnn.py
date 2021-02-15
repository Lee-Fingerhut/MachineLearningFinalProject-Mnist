import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x


def eval(test_dl: DataLoader, net: CNN) -> float:
    with torch.no_grad():
        num_samples = 0
        num_success = 0
        for i, sample in enumerate(test_dl):

            X, y = sample
            outputs = net(X)
            pred = torch.argmax(outputs, axis=1)
            num_samples += len(y)
            num_success += torch.sum(pred == y)
    return 100. * num_success / num_samples


def cnn_clf(
        train_dl: DataLoader,
        test_dl: DataLoader,
        learning_rate: float = 1e-3,
        max_iter: int = 10,
        save_as: str = ''
) -> None:

    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = []
    test_acc = []
    for iter in range(max_iter):
        for i, sample in enumerate(train_dl):
            optimizer.zero_grad()
            X, y = sample
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if i % 100 == 0:
                print('[{:3}, {:5}] loss: {:.3f}'.format(iter, i, train_loss[-1]))

        acc = eval(test_dl, net)
        test_acc.append(acc)
        print('[{:3}] accuracy: {:.3f}%'.format(iter, test_acc[-1]))

    _, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].plot(np.arange(0, len(train_loss)), train_loss)
    axs[0].set_title('train loss')
    axs[0].set_xlabel('#batches')
    axs[0].set_ylabel('loss')

    axs[1].plot(np.arange(0, len(test_acc)), test_acc)
    axs[1].set_title('test accuracy')
    axs[1].set_xlabel('#epochs')
    axs[1].set_ylabel('accuracy[%]')

    if save_as != '':
        plt.savefig(save_as)
    plt.show()
