import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import random

random.seed(666)
os.environ['PYTHONHASHSEED'] = str(666)
np.random.seed(666)
#torch.manual_seed(666)

batch_size = 64
train_val_size = 0.8
val_size = 0.125
test_size = 0.2
least_samples = 1
alpha = 0.1

def check(config_path, train_path, test_path, val_path, num_clients, num_classes, niid=False,
        balance=True, partition=None):

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(val_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        # get the number of all samples in the dataset
        all_samples_num = len(dataset_label)
        # get index for each class
        idxs = np.array(range(len(dataset_label)))
        # get num of samples in each class
        idx_for_each_class = []
        num_samples_each_class = []
        for i in range(num_classes):
            temp_index = idxs[dataset_label == i]
            np.random.shuffle(temp_index)
            idx_for_each_class.append(temp_index)
            num_samples_each_class.append(len(temp_index))
        # needs (k * client_num) shards
        all_shards_num = class_per_client * num_clients
        # get shard size
        print("num_samples_each_class",num_samples_each_class)
        shard_num_for_each_class = int(all_shards_num / num_classes)
        shard_size = int(np.floor(min(num_samples_each_class) / shard_num_for_each_class))
        print("shard_size",shard_size)

        # get shards_list
        shards_list = []
        for i in range(num_classes):
            for j in range(shard_num_for_each_class):
                temp_shard = idx_for_each_class[i][j*shard_size:j*shard_size+shard_size]
                shards_list.append(temp_shard)

        # index of shards
        shards_index = list(np.arange(len(shards_list)))
        print("shards_index",shards_index)
        np.random.shuffle(shards_index)
        print("shuffled_shards_index", shards_index)
        # assign shards to clients
        for client in range(num_clients):
            # select k class
            selected_shards = shards_index[client*class_per_client:client*class_per_client+class_per_client]
            for shards in selected_shards:
                if client not in dataidx_map.keys():
                    dataidx_map[client] = shards_list[shards]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], shards_list[shards], axis=0)
            print("selected_shards",selected_shards)

    # case study settings
    elif partition == 'group':
        # get the number of all samples in the dataset
        all_samples_num = len(dataset_label)
        from collections import Counter
        print(Counter(dataset_label))
        # get index for each class
        idxs = np.array(range(len(dataset_label)))
        # get num of samples in each class
        idx_for_each_class = []
        num_samples_each_class = []
        idx_for_each_class_iid = []
        num_samples_each_class_iid = []
        # noniid_rate denotes the rate of sample numbers for domain class
        noniid_rate = 0.8
        for i in range(num_classes):
            temp_index = idxs[dataset_label == i]
            np.random.shuffle(temp_index)
            noniid_num = int(noniid_rate * len(temp_index))
            print("noniid_num", noniid_num)
            # get non-iid index and iid index
            idx_for_each_class.append(temp_index[0:noniid_num])
            num_samples_each_class.append(len(temp_index[0:noniid_num]))
            idx_for_each_class_iid.append(temp_index[noniid_num:])
            num_samples_each_class_iid.append(len(temp_index[noniid_num:]))
            print("iid_num", len(temp_index[noniid_num:]))

        # group num
        group_num = int(num_classes / class_per_client)
        client_num_per_group = int(num_clients / group_num)

        # shuffle idx
        shuffled_class_index = np.arange(len(idx_for_each_class), dtype='int')
        np.random.shuffle(shuffled_class_index)
        print("shuffled_class_index", shuffled_class_index)
        idx_for_each_class = np.array(idx_for_each_class)[shuffled_class_index]
        num_samples_each_class = np.array(num_samples_each_class)[shuffled_class_index]

        # select shards for clients in a group
        group_shards = {}
        for i in range(0,num_clients,client_num_per_group):
            # select num_class_per_client shards
            group_shards_list = []
            for j in range(class_per_client):
                temp_size = int(num_samples_each_class[int(i*class_per_client/client_num_per_group)+j] / client_num_per_group)
                for q in range(client_num_per_group):
                    if q == client_num_per_group-1:
                        temp_group_shards = idx_for_each_class[int(i*class_per_client/client_num_per_group) + j][q * temp_size:]
                    else:
                        temp_group_shards = idx_for_each_class[int(i*class_per_client/client_num_per_group) + j][q * temp_size:q * temp_size + temp_size]
                    group_shards_list.append(temp_group_shards)
            for t in range(client_num_per_group):
                group_shards[i + t] = []
                for p in range(class_per_client):
                    print(p * client_num_per_group + t)
                    group_shards[i+t].append(group_shards_list[p*client_num_per_group+t])
            print("i",i)
        print("group_shards",group_shards)

        # assign non-iid shards to clients
        for client in range(num_clients):
            selected_shards = group_shards[client]
            for shards in selected_shards:
                if client not in dataidx_map.keys():
                    dataidx_map[client] = shards
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], shards, axis=0)
            print("selected_shards",selected_shards)

        # assing iid shards to clients
        for client in range(num_clients):
            for class_i in range(num_classes):
                per_num = int(num_samples_each_class_iid[class_i]/num_clients)
                temp_class_index = idx_for_each_class_iid[class_i][per_num*client:per_num*client+per_num]
                dataidx_map[client] = np.append(dataidx_map[client], temp_class_index, axis=0)
                print("iid_classnum",temp_class_index)

    # Practical
    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data according to above data split strategy
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))


    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data, val_data = [], [], []
    num_samples = {'train':[], 'test':[], 'val':[]}

    for i in range(len(y)):
        # data split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X[i], y[i], test_size=test_size, shuffle=True, random_state=666)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, shuffle=True, random_state=666)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))
        val_data.append({'x': X_val, 'y': y_val})
        num_samples['val'].append(len(y_val))


    print("Total number of samples:", sum(num_samples['train'] + num_samples['test'] + num_samples['val']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print("The number of val samples:", num_samples['val'])
    del X, y
    # gc.collect()

    return train_data, test_data, val_data

def save_file(config_path, train_path, test_path, val_path, train_data, test_data, val_data, num_clients,
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    for idx, val_dict in enumerate(val_data):
        with open(val_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=val_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
