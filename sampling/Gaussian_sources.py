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

from sampling.source_simulations import simulate_gaussian_source
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


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


def create_gaussian_source(grid, sides=1, blur=True):
    '''
    takes grid
    side: one-sided or two-sided
    core dominated or lobe dominated
    number of components
    angle of the jet
    
    components should not have too big gaps between each other
    '''
    source = grid[0]
    amp = np.array([2, 1])
    x = np.array([10, 30])
    y = np.array([0, 0])
    sig_x = np.array([2, 4])
    sig_y = np.array([2, 4])
    rot = 45
    
    for i in range(1):
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


grid = create_grid(128)
s = create_gaussian_source(grid)
s.shape
plt.imshow(s)
plt.colorbar()

74, 64





def gaussian_source(X, Y, blur=False):
    ''' Creates a gaussian source consisting of different gaussian componentes
        on a 2d grid

    X: x coordinates of 2d grid
    Y: y coordinates of 2d grid
    blur: smear the components with a gaussian filter
    '''
    source = np.zeros(X.shape)
    cent = len(X)//2

    center = np.array([[cent, cent], [cent+10, cent], [cent-10, cent],
                       [cent+20, cent], [cent-20, cent]])
    intens = np.array([5, 2, 2, 1, 1])
    fwhm = np.array([2, 4, 4, 6, 6])

    for i in range(0, 5):
        source += gaussian_component(X, Y, intens[i], fwhm[i], fwhm[i],
                                     center=center[i])

    if blur is True:
        source = gaussian_filter(source, sigma=3)

    return source



source = simulate_gaussian_source(128)

plt.imshow(source)


