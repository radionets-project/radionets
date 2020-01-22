from torch.utils.data import DataLoader
import torch
import h5py
import numpy as np


def normalize(x, m, s):
    return (x-m)/s


def do_normalisation(x, norm):
    """
    :param x        Object to be normalized
    :param norm     Pandas Dataframe which includes the normalisation factors
    """
    train_mean_real = torch.tensor(norm['train_mean_real'].values[0]).float()
    train_std_real = torch.tensor(norm['train_std_real'].values[0]).float()
    train_mean_imag = torch.tensor(norm['train_mean_imag'].values[0]).float()
    train_std_imag = torch.tensor(norm['train_std_imag'].values[0]).float()
    x[:, 0] = normalize(x[:, 0], train_mean_real, train_std_real)
    x[:, 1] = normalize(x[:, 1], train_mean_imag, train_std_imag)
    assert not torch.isinf(x).any()
    return x


class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset


def save_bundle(path, bundle, counter, name='gs_bundle'):
    with h5py.File(path + str(counter) + '.h5', 'w') as hf:
        hf.create_dataset(name,  data=bundle)
        hf.close()


def open_bundle(path):
    f = h5py.File(path, 'r')
    bundle = np.array(f['gs_bundle'])
    return bundle
