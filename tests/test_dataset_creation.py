import sys
sys.path.append('.')

# def test_open_mnist():
#     from mnist_cnn.create_mnist_fft import open_mnist

#     path = '../resources/mnist.pkl.gz'
#     x_train, x_valid = open_mnist(path)
#     assert x_train.shape == (50000, 784)
#     assert x_valid.shape == (10000, 784)

# def test_process_image():
#     from mnist_cnn.create_mnist_fft import open_mnist
#     from mnist_cnn.create_mnist_fft import process_img
#     from mnist_cnn.create_mnist_fft import write_h5
#     import numpy as np
#     import os

#     path = '../resources/mnist.pkl.gz'
#     x_train, x_valid = open_mnist(path)
#     x_train = x_train[0:10]
#     x_valid = x_valid[0:10]
#     processed_train = np.concatenate([process_img(img) for img in x_train])
#     processed_valid = np.concatenate([process_img(img) for img in x_valid])

#     assert processed_train.shape == (20, 4096)
#     assert processed_valid.shape == (20, 4096)

#     y_train = processed_train[0::2]
#     x_train = processed_train[1::2]

#     assert x_train.shape == (10, 4096)
#     assert y_train.shape == (10, 4096)

#     y_valid = processed_valid[0::2]
#     x_valid = processed_valid[1::2]

#     assert x_valid.shape == (10, 4096)
#     assert y_valid.shape == (10, 4096)

#     outpath = 'mnist.h5'
#     write_h5(outpath, x_train, y_train)

#     assert os.path.exists('./mnist.h5')

#     os.remove('./mnist.h5')


def test_create_mnist_fft():
    import numpy as np
    from mnist_cnn.utils import open_mnist, process_img
    from tqdm import tqdm

    path = './resources/mnist.pkl.gz'
    train_x, valid_x = open_mnist(path)
    train_x = train_x[0:50]
    valid_x = valid_x[0:50]
    # Process train images, split into x and y
    all_train = np.concatenate([process_img(img) for img in tqdm(train_x)])
    y_train = all_train[0::2]
    x_train = all_train[1::2]

    # Process valid images, split into x and y
    all_valid = np.concatenate([process_img(img) for img in tqdm(valid_x)])
    y_valid = all_valid[0::2]
    x_valid = all_valid[1::2]

    assert y_train.shape == (50, 4096)
    assert x_train.shape == (50, 4096)
    assert y_valid.shape == (50, 4096)
    assert x_valid.shape == (50, 4096)
