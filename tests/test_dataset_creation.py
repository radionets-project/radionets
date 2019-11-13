import sys
sys.path.append('.')


def test_open_mnist():
    """
    Test the open_mnist function from utils and check if the shapes
    are as expected.
    """
    from mnist_cnn.utils import open_mnist

    path = './resources/mnist.pkl.gz'
    x_train, x_valid = open_mnist(path)
    assert x_train.shape == (50000, 784)
    assert x_valid.shape == (10000, 784)
    return x_train, x_valid


def test_process_img():
    """"
    Test the process_img function from utils. Check the expected shapes.
    """
    import numpy as np
    from skimage.transform import resize

    x_train, x_valid = test_open_mnist()
    image = x_train[0]
    img_reshaped = image.reshape(28, 28)

    assert img_reshaped.shape == (28, 28)

    img_rescaled = resize(img_reshaped, (64, 64), anti_aliasing=True,
                          mode='constant')

    assert img_rescaled.shape == (64, 64)

    img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))

    assert img_fft.shape == (64, 64)
    assert img_rescaled.reshape(4096).shape == (4096,)
    assert img_fft.reshape(4096).shape == (4096,)


def test_train_valid_split():
    """"
    Validate the shape of the train and valid datasets. Furthermore,
    check if the h5 file for saving is created.
    """
    import numpy as np
    import os
    from mnist_cnn.utils import open_mnist, process_img, write_h5

    path = './resources/mnist.pkl.gz'
    x_train, x_valid = open_mnist(path)
    x_train = x_train[0:10]
    x_valid = x_valid[0:10]
    processed_train = np.concatenate([process_img(img) for img in x_train])
    processed_valid = np.concatenate([process_img(img) for img in x_valid])

    assert processed_train.shape == (20, 4096)
    assert processed_valid.shape == (20, 4096)

    y_train = processed_train[0::2]
    x_train = processed_train[1::2]

    assert x_train.shape == (10, 4096)
    assert y_train.shape == (10, 4096)

    y_valid = processed_valid[0::2]
    x_valid = processed_valid[1::2]

    assert x_valid.shape == (10, 4096)
    assert y_valid.shape == (10, 4096)

    outpath = 'mnist.h5'
    write_h5(outpath, x_train, y_train)

    assert os.path.exists('./mnist.h5')

    os.remove('./mnist.h5')
