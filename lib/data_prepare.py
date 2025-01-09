import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import torch.utils.data as data_utils
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, SubsetRandomSampler,Subset

def get_dataloaders1(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])


    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )
    train_val_set = ConcatDataset([trainset, valset])
    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,drop_last=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size,shuffle=False,drop_last=True
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,shuffle=False,drop_last=True
    )
    

    return trainset_loader, valset_loader, testset_loader, scaler
def get_dataloaders2(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    data = data[..., features]      
    index = np.load(os.path.join(data_dir, "my_index1.npz"))
    index_test = np.load(os.path.join(data_dir, "my_index2.npz"))
    index = index['index1']
    index_test = index_test['index2']
    
    x_index = vrange(index[:, 0], index[:, 1])
    y_index = vrange(index[:, 1], index[:, 2])
    x_data = data[x_index]
    y_data = data[y_index][..., :1]
    x_test_index = vrange(index_test[:, 0], index_test[:, 1])
    y_test_index = vrange(index_test[:, 1], index_test[:, 2])
    x_test_data = data[x_test_index]
    y_test_data = data[y_test_index][..., :1]
    scaler = StandardScaler(mean=x_data[..., 0].mean(), std=x_data[..., 0].std())
    x_data[..., 0] = scaler.transform(x_data[..., 0])

    x_test_data[..., 0] = scaler.transform(x_test_data[..., 0])

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_data), torch.FloatTensor(y_data))


    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_data), torch.FloatTensor(y_test_data))
    total_samples = len(dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    train_size = int(0.875 * total_samples) 
    val_size = total_samples - train_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,drop_last=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,drop_last=True)


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False,drop_last=True)
    

    return train_loader, val_loader, test_loader, scaler
