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
    img_rescaled = resize(img_reshaped, (64, 64), anti_aliasing=True, mode='constant')
    img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))
    return img_rescaled.reshape(4096), img_fft.reshape(4096)


def reshape_img(img, size=64):
    return img.reshape(size, size)


def write_h5(path, x, y, name_x, name_y):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset(name_x,  data=x)
        hf.create_dataset(name_y,  data=y)
        hf.close()

def main():
    # Load MNIST dataset
    path = './data/mnist/mnist.pkl.gz'
    train_x, valid_x = open_mnist(path)

    # Process train images, split into x and y
    all_train = np.concatenate([process_img(img) for img in train_x]) 
    y_train = all_train[0::2]
    x_train = all_train[1::2]

    # Process valid images, split into x and y
    all_valid = np.concatenate([process_img(img) for img in valid_x]) 
    y_valid = all_valid[0::2]
    x_valid = all_valid[1::2]

    # Write data to h5 file
    outpath_train = 'data/mnist_train.h5'
    write_h5(outpath_train, x_train, y_train, 'x_train', 'y_train')
    outpath_valid = 'data/mnist_valid.h5'
    write_h5(outpath_valid, x_valid, y_valid, 'x_valid', 'y_valid')


if __name__ == '__main__':
    main()