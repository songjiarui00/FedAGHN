import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=False, is_test=False, is_val=False):

    if is_val:
        val_data_dir = os.path.join('../dataset', dataset, 'val/')

        val_file = val_data_dir + str(idx) + '.npz'
        with open(val_file, 'rb') as f:
            val_data = np.load(f, allow_pickle=True)['data'].tolist()

        return val_data

    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    if is_test:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=False, is_test=False, is_val=False):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_val:
        val_data = read_data(dataset, idx, is_val=is_val, is_train=False, is_test=False)
        X_val = torch.Tensor(val_data['x']).type(torch.float32)
        y_val = torch.Tensor(val_data['y']).type(torch.int64)
        val_data = [(x, y) for x, y in zip(X_val, y_val)]
        return val_data

    if is_train:
        train_data = read_data(dataset, idx, is_train=is_train, is_test=False, is_val=False)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data

    if is_test:
        test_data = read_data(dataset, idx, is_test=is_test, is_val=False, is_train=False)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

