# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from simulations.source_simulations import simulate_gaussian_source
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
import h5py


def create_grid(pixel):
    ''' Create a square 2d grid

    pixel: number of pixel in x and y
    '''
    x = np.linspace(0, pixel-1, num=pixel)
    y = np.linspace(0, pixel-1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.zeros(X.shape), X, Y])
    return grid


def create_rot_mat(alpha):
    '''
    Create 2d rotation matrix for given alpha
    alpha: rotation angle in rad
    '''
    rot_mat = np.array([[np.cos(alpha), np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot=0, center=None):
    ''' Create a gaussian component on a 2d grid

    x: x coordinates of 2d grid
    y: y coordinates of 2d grid
    flux: peak amplitude of component
    x_fwhm: full-width-half-maximum in x direction
    y_fwhm: full-width-half-maximum in y direction
    rot: rotation of component
    center: center of component
    '''
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(np.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2

    gauss = flux * np.exp(-((x_0 - x)**2/(2*(x_fwhm)**2) +
                          (y_0 - y)**2 / (2*(y_fwhm)**2)))
    return gauss


def add_gaussian(grid, amp, x, y, sig_x, sig_y, rot):
    '''
    Takes a grid and adds a Gaussian component relative to the center
    
    grid: 2d grid 
    amp: amplitude
    x: x position, will be calculated rel. to center
    y: y position, will be calculated rel. to center
    sig_x: standard deviation in x
    sig_y: standard deviation in y
    '''
    cent = np.array([len(grid[0])//2 + x, len(grid[0])//2 + y])
    X = grid[1]
    Y = grid[2]
    gaussian = grid[0]
    gaussian += gaussian_component(
                                X,
                                Y,
                                amp,
                                sig_x,
                                sig_y,
                                rot,
                                center=cent,
    )

    return gaussian


def create_gaussian_source(comps, amp, x, y, sig_x, sig_y, rot, grid, sides=0, blur=True):
    '''
    takes grid
    side: one-sided or two-sided
    core dominated or lobe dominated
    number of components
    angle of the jet
    
    components should not have too big gaps between each other
    '''
    
    if sides == 1:
        comps += comps-1
        amp = np.append(amp, amp[1:])
        x = np.append(x, -x[1:])
        y = np.append(y, -y[1:])
        sig_x = np.append(sig_x, sig_x[1:])
        sig_y = np.append(sig_y, sig_y[1:])
        
    
    for i in range(comps):
        source = add_gaussian(
            grid = grid,
            amp = amp[i],
            x = x[i],
            y = y[i],
            sig_x = sig_x[i],
            sig_y = sig_y[i],
            rot = rot,
        )
    if blur is True:
        source = gaussian_filter(source, sigma=1.5)
    return source







def gauss_paramters():
    '''
    
    '''
    # random number of components between 4 and 9
    comps = np.random.randint(4, 10)
    
    # start amplitude between 10 and 1e-3
    amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # if start amp is 0, draw a new number
    while amp_start == 0:
        amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # logarithmic decrease to outer components
    amp = np.array([amp_start/np.exp(i) for i in range(comps)])
    
    # linear distance bestween the components
    x = np.arange(0, comps) * 5
    y = np.zeros(comps)
    
    # extension of components
    # random start value between 1 - 0.375 and 1 - 0 
    # linear distance between components
    # distances scaled by factor between 0.25 and 0.5
    # randomnized for each sigma
    off1 = (np.random.random() + 0.5) / 4
    off2 = (np.random.random() + 0.5) / 4
    fac1 = (np.random.random() + 1) / 4
    fac2 = (np.random.random() + 1) / 4
    sig_x = (np.arange(1, comps+1) - off1 ) * fac1
    sig_y = (np.arange(1, comps+1) - off2 ) * fac2
    
    # jet rotation
    rot = np.random.randint(0, 360)
    # jet one- or two-sided
    sides = np.random.randint(0, 2)
    
    return comps, amp, x , y, sig_x, sig_y, rot, sides


def gaussian_source(pixel):
    grid = create_grid(pixel)
    comps, amp, x, y, sig_x, sig_y, rot, sides = gauss_paramters()
    s = create_gaussian_source(comps, amp, x, y, sig_x, sig_y, rot, grid, sides, blur=True)
    return s


s = gaussian_source(128)
plt.imshow(s, norm=LogNorm(vmin=1e-8, vmax=10))
plt.colorbar()


def save_bundle(path, bundle, counter, name='gs_bundle'):
    with h5py.File(path + str(counter) + '.h5', 'w') as hf:
        hf.create_dataset(name,  data=bundle)
        hf.close()


def open_bundle(path):
    f = h5py.File(path, 'r')
    bundle = np.array(f['gs_bundle'])
    return bundle


def running_stats(path, num_bundles):
    means = np.array([])
    stds = np.array([])
    
    for i in range(num_bundles):
        bundle_path = path + str(i) + '.h5'
        bundle = open_bundle(bundle_path)
        bundle_mean = bundle.mean()
        bundle_std = bundle.std()
        means = np.append(bundle_mean, means)
        stds = np.append(bundle_std, stds)
    mean = means.mean()
    std = stds.mean()
    return mean, std


# %%time
for i in range(1024):
    grid = create_grid(128)
    comps, amp, x, y, sig_x, sig_y, rot, side = gauss_paramters()
    s = create_gaussian_source(comps, amp, x, y, sig_x, sig_y, rot, grid, side, blur=True)

# %%time
bundle = np.array([gaussian_source() for i in range(1024)])
print(bundle.shape)

# %%time
path = 'gaussian_sources/bundle_'
for j in range(20):
    bundle = np.array([gaussian_source() for i in range(1024)])
    save_bundle(path, bundle, j)

mean, std = running_stats('gaussian_sources/bundle_', 20)
mean,std

from pathlib import Path


def get_bundles(path):
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


path = 'gaussian_sources'
bundles = get_bundles(path)
bundles


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


from dl_framework.data import DataLoader
from functools import partial


class h5_dataset():
    def __init__(self, bundles_x, bundles_y):
        self.x = bundles_x
        self.y = bundles_y

    def __call__(self):
        return print('This is the h5_dataset class.')
        
    def __len__(self):
        return len(self.x) * len(self.open_bundle(self.x[0]))

    def __getitem__(self, i):
        x = self.open_image(self.x, i)
        y = self.open_image(self.y, i)
        return x, y

    def open_bundle(self, bundle):
        bundle = h5py.File(bundle, 'r')
        data = bundle['gs_bundle']
        return data
    
    def open_image(self, bundle, i):
        bundle_i = i // 1024
        image_i = i - bundle_i * 1024
        bundle = h5py.File(bundle[bundle_i], 'r')
        data = bundle['gs_bundle'][image_i]
        return data


train_ds = h5_dataset(bundles, bundles)

len(train_ds)

train_ds()

bs = 64

next(iter(train_ds.x))

val = next(iter(train_ds))
plt.imshow(val[0])

data = DataBunch(*get_dls(train_ds, train_ds, bs))

x, y = next(iter(data.train_dl))
plt.imshow(x[0])

# +
#loader = get_dls(train_ds, train_ds, bs)

# +
#loader

# +
#x = next(iter(loader[0]))
#img = x[0][0]
#plt.imshow(img)
# -

data.train_ds[125]






























