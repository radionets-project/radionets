import numpy as np
import os


def test_open_mnist():
    from mnist_cnn.utils import open_mnist

    path = "./resources/mnist.pkl.gz"
    x_train, x_valid = open_mnist(path)
    assert x_train.shape == (50000, 784)
    assert x_valid.shape == (10000, 784)


def test_adjust_outpath():
    from mnist_cnn.utils import adjust_outpath

    path = "this/is/a/path"
    out = adjust_outpath(path, "test")

    assert type(path) == type(out)
    assert out.split("/")[-1] == "fft_bundle_test0.h5"


def test_prepare_mnist_bundles():
    from mnist_cnn.utils import prepare_mnist_bundles

    bundle = np.ones((10, 3, 3))
    build = "./tests/build"
    os.mkdir('./tests/build')

    assert prepare_mnist_bundles(bundle, build, "test", noise=True, pixel=5) is None

    os.remove(build + "/fft_bundle_test0.h5")
    os.rmdir("./tests/build")

# def test_train_valid_split():
#     """"
#     Validate the shape of the train and valid datasets. Furthermore,
#     check if the h5 file for saving is created.
#     """
#     import numpy as np
#     import os
#     from mnist_cnn.utils import open_mnist, process_img, write_h5

#     path = './resources/mnist.pkl.gz'
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
