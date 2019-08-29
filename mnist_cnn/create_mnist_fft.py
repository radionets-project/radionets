import gzip
import pickle
import numpy as np
import h5py
from skimage.transform import resize


# Load MNIST dataset
path = './data/mnist/'
with gzip.open(path+'mnist.pkl.gz', 'rb') as f:
    ((train_x, train_y), (valid_x, valid_y), _) = pickle.load(f, encoding='latin-1')

def process_img(image):
    ''' Resize images to 64 x 64 and calculate fft 
        Return images as vector
    '''
    img_reshaped = image.reshape(28, 28)
    img_rescaled = resize(img_reshaped, (64, 64))
    img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))
    return img_rescaled.reshape(4096), img_fft.reshape(4096)


def reshape_img(img, size=64):
    return img.reshape(size, size)


# Process train images, split into x and y
all_train = np.concatenate([process_img(img) for img in train_x]) 
y_train = all_train[0::2]
x_train = all_train[1::2]

# Process valid images, split into x and y
all_valid = np.concatenate([process_img(img) for img in valid_x]) 
y_valid = all_valid[0::2]
x_valid = all_valid[1::2]

# Write data to h5 file
with h5py.File('mnist.h5', 'w') as hf:
    hf.create_dataset('x_train',  data=x_train)
    hf.create_dataset('y_train',  data=y_train)
    hf.create_dataset('x_valid',  data=x_valid)
    hf.create_dataset('y_valid',  data=y_valid)


