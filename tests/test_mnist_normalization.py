import warnings

def test_normalization():
    from mnist_cnn.utils import get_h5_data, normalize
    import numpy as np
    import os
    import pandas as pd
    train_path = 'mnist_cnn/data/mnist_samp_train.h52'
    valid_path = 'mnist_cnn/data/mnist_samp_valid.h5'
    normalization = 'mnist_cnn/data/normalization_factors.csv'
    if (os.path.exists(train_path) & os.path.exists(valid_path)
        & os.path.exists(normalization)):
        norm_values = pd.read_csv(normalization)
        train_mean = norm_values['mean'].values
        train_std = norm_values['std'].values
        x_train, _ = get_h5_data(train_path, columns=['x_train', 'y_train'])
        x_valid, _ = get_h5_data(valid_path, columns=['x_valid', 'y_valid'])

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
    else:
        warnings.warn(UserWarning('Test can only be run after mnist_fft_samp creation!'))
        pass
