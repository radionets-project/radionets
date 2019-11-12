import gzip
import pickle
import numpy as np
import h5py
from skimage.transform import resize


def open_mnist(path):
    with gzip.open(path, 'rb') as f:
        ((train_x, _), (valid_x, _), _) = pickle.load(f, encoding='latin-1')
    return train_x, valid_x


def process_img(image):
    ''' Resize images to 64 x 64 and calculate fft
        Return images as vector
    '''
    img_reshaped = image.reshape(28, 28)
    img_rescaled = resize(img_reshaped, (64, 64), anti_aliasing=True,
                          mode='constant')
    img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))
    return img_rescaled.reshape(4096), img_fft.reshape(4096)


def reshape_img(img, size=64):
    return img.reshape(size, size)


def write_h5(path, x, y, name_x='x_train', name_y='y_train'):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset(name_x,  data=x)
        hf.create_dataset(name_y,  data=y)
        hf.close()


def get_h5_data(path, columns):
    ''' Load mnist h5 data '''
    f = h5py.File(path, 'r')
    x = np.abs(np.array(f[columns[0]]))
    y = np.abs(np.array(f[columns[1]]))
    return x, y


def normalize(x, m, s): return (x-m)/s


def create_mask(ar):
    ''' Generating mask with min and max value != inf'''
    val = ar.copy()
    val[np.isinf(val)] = 0
    l = val.min()
    h = val.max()
    mask = (l < ar) & (ar < h)
    return mask