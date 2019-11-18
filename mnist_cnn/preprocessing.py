import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings


# Define torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_h5_data(path, columns):
    ''' Load mnist h5 data '''
    f = h5py.File(path, 'r')
    x = np.abs(np.array(f[columns[0]]))
    y = np.abs(np.array(f[columns[1]]))
    return x, y


def prepare_dataset(x_train, y_train, x_valid, y_valid, log=False):
    ''' Preprocessing dataset:
    split
    log (optional)
    create ArrayDataset
    '''
    if log is True:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        x_train = np.log(x_train)
        x_valid = np.log(x_valid)

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train,
                                                            x_valid, y_valid))
    train_ds = ArrayDataset(x_train, y_train)
    valid_ds = ArrayDataset(x_valid, y_valid)

    assert x_train.shape, y_train.shape != (50000, 4096)
    assert x_valid.shape, y_valid.shape != (10000, 4096)
    assert train_ds.c, valid_ds.c != 4096
    return train_ds, valid_ds


class ArrayDataset():
    ''' Sample array dataset '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.c = x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i].float(), self.y[i].float()


def get_dls(train_ds, valid_ds, bs, **kwargs):
    ''' Define data loaders '''
    return(DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
           pin_memory=False, **kwargs),
           DataLoader(valid_ds, batch_size=bs*2, shuffle=False,
           drop_last=True, pin_memory=False, **kwargs))


class DataBunch():
    ''' Define data bunch '''
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset
