import warnings


def test_normalization():
    from dl_framework.data import normalize
    from mnist_cnn.utils import get_h5_data
    import numpy as np
    import os
    import pandas as pd
    # from mnist_cnn.utils import create_mask
    train_path = 'mnist_cnn/data/mnist_samp_train.h5'
    valid_path = 'mnist_cnn/data/mnist_samp_valid.h5'
    normalization = 'mnist_cnn/data/normalization_factors.csv'
    if (os.path.exists(train_path) & os.path.exists(valid_path)
                                   & os.path.exists(normalization)):
        norm_values = pd.read_csv(normalization)
        train_mean = norm_values['train_mean'].values
        valid_mean = norm_values['valid_mean'].values
        train_std = norm_values['train_std'].values
        x_train, _ = np.log(get_h5_data(train_path,
                            columns=['x_train', 'y_train']))
        x_valid, _ = np.log(get_h5_data(valid_path,
                            columns=['x_valid', 'y_valid']))

        x_train[np.isinf(x_train)] = train_mean
        x_valid[np.isinf(x_valid)] = valid_mean

        assert not np.isinf(x_train).any()
        assert not np.isinf(x_valid).any()

        x_train = normalize(x_train, train_mean, train_std)
        x_valid = normalize(x_valid, train_mean, train_std)

        assert np.isclose(x_train.mean(), 0, atol=1e-1)
        assert np.isclose(x_train.std(), 1, atol=1e-1)
        assert np.isclose(x_valid.mean(), 0, atol=1e-1)
        assert np.isclose(x_valid.std(), 1, atol=1e-1)
    else:
        warnings.warn(UserWarning(
            'Test can only be run after mnist_fft_samp creation!'))
        pass
