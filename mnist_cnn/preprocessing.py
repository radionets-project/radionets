import sys
sys.path.append('..')
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sampling.uv_simulations import sample_freqs


# Define torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_h5_data(path, columns):
    ''' Load mnist h5 data '''
    f = h5py.File(path, 'r')
    x = np.abs(np.array(f[columns[0]]))
    y = np.abs(np.array(f[columns[1]]))
    return x, y


def prepare_dataset(x_train, y_train, x_valid, y_valid, log=False, freq_samp=False, quantile=False,
                    positive=False):
    ''' Preprocessing dataset: 
    split
    normalize
    log (optional)
    freq_sampling (optional)
    create ArrayDataset
    '''
    if log is True:
        x_train = np.log(x_train)
        x_valid = np.log(x_valid)

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    x_train, x_valid = noramlize_data(x_train, x_valid, quantile, positive)
    train_ds = ArrayDataset(x_train, y_train)
    valid_ds = ArrayDataset(x_valid, y_valid)

    assert x_train.shape, y_train.shape != (50000, 4096)
    assert x_valid.shape, y_valid.shape != (10000, 4096)
    assert train_ds.c, valid_ds.c != 4096
    # print(train_ds.c)
    return train_ds, valid_ds


def noramlize_data(x_train, x_valid, quantile=False, positive=False):
    ''' Normalize dataset excluding 0.1 and 0.9 qunatile '''
    if quantile is True:
        mask = quantile_mask(x_train)
        train_mean,train_std = x_train[mask].mean(),x_train[mask].std()
    else:
        train_mean,train_std = x_train.mean(),x_train.std()
    if positive is True:
        mask = x_train > 0
        train_mean,train_std = x_train[mask].mean(),x_train[mask].std()
    else:
        train_mean,train_std = x_train.mean(),x_train.std()
    x_train[torch.isinf(x_train)] = train_mean
    x_valid[torch.isinf(x_valid)] = train_mean
    # from IPython import embed
    # embed()
    # train_std = x_train.std()
    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    print(train_mean, train_std)
    print('Normalization')
    print(x_train.mean(), x_train.std())
    print(x_valid.mean(), x_valid.std())
    return x_train, x_valid

def normalize(x, m, s): return (x-m)/s

def quantile_mask(ar):
    ''' Generating 0.1 and 0.9 quantile mask '''
    l = np.quantile(ar, 0.1)
    print(l)
    h = np.quantile(ar, 0.9)
    mask = (l < ar) & (ar < h)
    return mask

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
    return(DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, pin_memory=False, **kwargs),
           DataLoader(valid_ds, batch_size=bs*2, shuffle=False,  drop_last=True, pin_memory=False, **kwargs))

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