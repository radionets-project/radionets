import sys
sys.path.append('..')
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
# from sampling.uv_simulations import sample_freqs
import warnings


# Define torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_h5_data(path, columns):
    ''' Load mnist h5 data '''
    f = h5py.File(path, 'r')
    x = np.abs(np.array(f[columns[0]]))
    y = np.abs(np.array(f[columns[1]]))
    return x, y


def prepare_dataset(x_train, y_train, x_valid, y_valid, log=False,
                    use_mask=False):
    ''' Preprocessing dataset:
    split
    normalize
    log (optional)
    freq_sampling (optional)
    create ArrayDataset
    '''
    if log is True:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        x_train = np.log(x_train)
        x_valid = np.log(x_valid)

    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train,
                                                            x_valid, y_valid))
    x_train, x_valid = noramlize_data(x_train, x_valid, use_mask)
    train_ds = ArrayDataset(x_train, y_train)
    valid_ds = ArrayDataset(x_valid, y_valid)

    assert x_train.shape, y_train.shape != (50000, 4096)
    assert x_valid.shape, y_valid.shape != (10000, 4096)
    assert train_ds.c, valid_ds.c != 4096
    return train_ds, valid_ds


def noramlize_data(x_train, x_valid, use_mask=False):
    ''' Normalize dataset setting inf pixel to mean '''
    if use_mask is True:
        mask = create_mask(x_train)
        train_mean, train_std = x_train[mask].mean(), x_train[mask].std()
    else:
        train_mean, train_std = x_train.mean(), x_train.std()

    mask = create_mask(x_valid)
    valid_mean = x_valid[mask].mean()
    x_train[torch.isinf(x_train)] = train_mean
    x_valid[torch.isinf(x_valid)] = valid_mean
    train_std = x_train.std()
    # assert len(x_train[torch.isinf(x_train)]) != 0
    # assert len(x_valid[torch.isinf(x_valid)]) != 0

    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    if not np.isclose(x_train.mean(), 0, atol=1e-1):
        print('Training mean is ', x_train.mean())
    if not np.isclose(x_train.std(), 1, atol=1e-1):
        print('Training std is ', x_train.std())
    if not np.isclose(x_valid.mean(), 0, atol=1e-1):
        print('Valid mean is ', x_valid.mean())
    if not np.isclose(x_valid.std(), 1, atol=1e-1):
        print('Valid std is ', x_valid.std())
    return x_train, x_valid


def normalize(x, m, s): return (x-m)/s


def create_mask(ar):
    ''' Generating mask with min and max value != inf'''
    val = ar.clone()
    val[torch.isinf(val)] = 0
    l = val.min()
    h = val.max()
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
