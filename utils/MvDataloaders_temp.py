import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import scipy.io as scio


def Get_dataloaders(batch_size=128, path_to_data='./utils/DATA/', DATANAME='MNIST-Sobel.mat', seed=15):

    torch.manual_seed(seed)

    """Dataloader with (32, 32) images."""
    DATA = scio.loadmat(path_to_data + DATANAME)
    view = len(DATA) - 3 - 1
    X1 = DATA['X1']
    X2 = DATA['X2']
    print('X1 Shape')
    print(X1.shape)
    print('X2 Shape')
    print(X2.shape)

    if view == 3:
        X3 = DATA['X3']
        print('X3 Shape')
        print(X3.shape)

    y = DATA['Y']
    size = y.shape[1]
    print('Y Shape')
    print(y.shape[1])

    cluster = np.unique(y)
    print(cluster)
    print('Cluster K:' + str(len(cluster)))

    x1 = torch.from_numpy(X1).float()
    X1 = []
    x2 = torch.from_numpy(X2).float()
    X2 = []

    if view == 3:
        x3 = torch.from_numpy(X3).float()
        X3 = []

    y = torch.from_numpy(y[0])

    if view == 2:
        dataset = TensorDataset(x1, x2, y)
    elif view == 3:
        dataset = TensorDataset(x1, x2, x3, y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, view, len(cluster), size
