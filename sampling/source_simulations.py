import numpy as np
from scipy.ndimage import gaussian_filter


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
        x_0 = center[0]
        y_0 = center[1]

    rotation = np.deg2rad(rot)
    x_rot = (np.cos(rotation) * x - np.sin(rotation) * y) 
    y_rot = (np.sin(rotation) * x + np.cos(rotation) * y) 

    gauss = flux * np.exp(- ((x_rot - x_0)**2/(2*(x_fwhm)**2) +
                          (y_rot - y_0)**2 / (2*(y_fwhm)**2)))
    return gauss


def create_grid(pixel):
    ''' Create a square 2d grid

    pixel: number of pixel in x and y
    '''
    x = np.linspace(0, pixel-1, num=pixel)
    y = np.linspace(0, pixel-1, num=pixel)
    X, Y = np.meshgrid(x, y)
    return X, Y


def gaussian_source(X, Y, blur=False):
    ''' Creates a gaussian source consisting of different gaussian componentes on a 2d grid

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


def simulate_gaussian_source(pixel, blur=False):
    ''' Creates a (extended) gaussian source on a 2d grid
        using create_grid and gaussian_source

    pixel: grid size in x and y
    blur: smear the components with a gaussian filter 
    
    number of components
    list of components?
    2 sided /  1 sided
    list of fluxes?
    fr1 and fr2
    seperation depended on fwhm ?
    '''
    X, Y = create_grid(pixel)
    source = gaussian_source(X, Y, blur=blur)
    return source
