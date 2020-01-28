import gzip
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

def open_mnist(path):
    with gzip.open(path, 'rb') as f:
        ((train_x, _), (valid_x, _), _) = pickle.load(f, encoding='latin-1')
    return train_x, valid_x


def process_img(image, noise=False):
    ''' Resize images to 64 x 64 and calculate fft
        Return images as vector
    '''
    if noise:
        image = add_noise(image)
    img_reshaped = image.reshape(64,64)
    img_rescaled = resize(img_reshaped, (64, 64), anti_aliasing=True,
                          mode='constant')
    img_fft = np.fft.fftshift(np.fft.fft2(img_rescaled))
    return img_rescaled.reshape(4096), img_fft.reshape(4096)


def add_noise(image, mean=0, std=1, index=0, plotting=False):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked.
    """
    noise = np.random.normal(mean, std, size=image.shape)*0.15
    image_noised = image + noise
    image = image.reshape(28, 28)
    image_noised = image_noised.reshape(28, 28)

    if plotting:
        print('Plotting')
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

        ax1.set_title(r'Original')
        im1 = ax1.imshow(image)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2.set_title(r"Noised")
        im2 = ax2.imshow(image_noised)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        fig.savefig('data/plots/input_plot_{}.pdf'.format(index), pad_inches=0,
                    bbox_inches='tight')

    image_noised = image_noised.reshape(784,)
    return image_noised


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
    x = np.array(f[columns[0]])
    y = np.array(f[columns[1]])
    return x, y


def create_mask(ar):
    ''' Generating mask with min and max value != inf'''
    val = ar.copy()
    val[np.isinf(val)] = 0
    low = val.min()
    high = val.max()
    mask = (low < ar) & (ar < high)
    return mask


def split_real_imag(array):
    """
    takes a complex array and returns the real and the imaginary part
    """
    return array.real, array.imag


def mean_and_std(array):
    return array.mean(), array.std()


def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 1, 2)

def pointsources(num_img, pi, targets):
    """
    generate images with random pointsources
    N is the number of images
    pi is the number of pixels
    targets is the number of pointsources
    """
    a = np.zeros((num_img,pi), dtype=np.int8)
    for i in range(num_img):
        for j in range(targets):
            x = np.random.randint(0,pi)
            a[i,x] = 1
    return a

def gauss(mx,my,sx,sy):
    x=np.arange(64)[None].astype(np.float)
    y=x.T
    return np.exp(-(y-my)**2/sy).dot(np.exp(-(x-mx)**2/sx))

def create_gauss(N,sources,spherical):
    img = np.zeros((N,64,64))
    mx = np.random.randint(1,64,size=(N,sources))
    my = np.random.randint(1,64,size=(N,sources))
    
    if spherical:
        sx = np.random.randint(1,15,size=(N,sources))
        sy = sx
    else:
        sx = np.random.randint(1,15,size=(N,sources))
        sy = np.random.randint(1,15,size=(N,sources))
        theta = np.random.randint(0,360,size=(N,sources))
    
    for i in range(N):
        targets = np.random.randint(2,sources+1)
        for j in range(targets):
            g = gauss(mx[i,j],my[i,j],sx[i,j],sy[i,j])
            if spherical:
                img[i] += g
            else:
                # rotation around center of the source
                padX = [g.shape[0] - mx[i,j], mx[i,j]]
                padY = [g.shape[1] - my[i,j], my[i,j]]
                imgP = np.pad(g, [padY, padX], 'constant')
                imgR = ndimage.rotate(imgP, theta[i,j], reshape=False)
                imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
                img[i] += imgC
    return img

