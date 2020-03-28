import warnings


def test_normalization():
    from dl_framework.data import normalize
    from mnist_cnn.utils import get_h5_data
    import numpy as np
    import os
    import pandas as pd
    from mnist_cnn.utils import split_real_imag, combine_and_swap_axes
    # from mnist_cnn.utils import create_mask
    train_path = 'mnist_cnn/data/mnist_samp_train.h5'
    valid_path = 'mnist_cnn/data/mnist_samp_valid.h5'
    normalization = 'mnist_cnn/data/normalization_factors.csv'
    if (os.path.exists(train_path) & os.path.exists(valid_path)
                                   & os.path.exists(normalization)):
        norm_values = pd.read_csv(normalization)
        train_mean = norm_values['train_mean_real'].values
        train_std = norm_values['train_std_real'].values
        x_train, _ = get_h5_data(train_path,
                                 columns=['x_train', 'y_train'])
        x_valid, _ = get_h5_data(valid_path,
                                 columns=['x_valid', 'y_valid'])

        x_train_real, x_train_imag = split_real_imag(x_train)
        x_valid_real, x_valid_imag = split_real_imag(x_valid)

        x_train = combine_and_swap_axes(x_train_real, x_train_imag)
        x_valid = combine_and_swap_axes(x_valid_real, x_valid_imag)

        x_train[:, 0][np.isinf(x_train[:, 0])] = train_mean
        x_valid[:, 0][np.isinf(x_valid[:, 0])] = train_mean

        assert not np.isinf(x_train[:, 0]).any()
        assert not np.isinf(x_valid[:, 0]).any()

        x_train[:, 0] = normalize(x_train[:, 0], train_mean, train_std)
        x_valid[:, 0] = normalize(x_valid[:, 0], train_mean, train_std)

        assert np.isclose(x_train[:, 0].mean(), 0, atol=1e-1)
        assert np.isclose(x_train[:, 0].std(), 1, atol=1e-1)
        assert np.isclose(x_valid[:, 0].mean(), 0, atol=1e-1)
        assert np.isclose(x_valid[:, 0].std(), 1, atol=1e-1)
    else:
        warnings.warn(UserWarning(
            'Test can only be run after mnist_fft_samp creation!'))
        pass
