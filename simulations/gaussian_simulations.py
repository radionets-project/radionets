import numpy as np
from scipy.ndimage import gaussian_filter
from dl_framework.data import save_bundle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_rot_mat(alpha):
    """
    Create 2d rotation matrix for given alpha

    Parameters
    ----------
    alpha: float
        rotation angle in rad

    Returns
    -------
    rot_mat: 2darray
        2d rotation matrix
    """
    rot_mat = np.array([[np.cos(alpha), np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot=0, center=None):
    """
    Adds a gaussian component to a 2d grid.

    Parameters
    ----------
    x: 2darray
        x coordinates of 2d meshgrid
    y: 2darray
        y coordinates of 2d meshgrid
    flux: float
        peak amplitude of component
    x_fwhm: float
        full-width-half-maximum in x direction
    y_fwhm: float
        full-width-half-maximum in y direction
    rot: int
        rotation of component in degree
    center: 2darray
        enter of component

    Returns
    -------
    gauss: 2darray
        2d grid with gaussian component
    """
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(np.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2

    gauss = flux * np.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    return gauss


def create_grid(pixel):
    """
    Creates a square 2d grid.

    Parameters
    ----------
    pixel: int
        number of pixel in x and y

    Returns
    -------
    grid: ndarray
        2d grid with 1e-10 pixels, X meshgrid, Y meshgrid
    """
    x = np.linspace(0, pixel - 1, num=pixel)
    y = np.linspace(0, pixel - 1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.zeros(X.shape) + 1e-10, X, Y])
    return grid


def add_gaussian(grid, amp, x, y, sig_x, sig_y, rot):
    """
    Takes a grid and adds n Gaussian component relative to the center.

    Parameters
    ----------
    grid: 2darray
        2d grid
    amp: float
        amplitude of gaussian component
    x: float
        x position, will be calculated rel. to center
    y: float
        y position, will be calculated rel. to center
    sig_x: float
        standard deviation in x
    sig_y: float
        standard deviation in y
    rot: int
        rotation in degree

    Returns
    -------
    gaussian: 2darray
        grid with gaussian component
    """
    cent = np.array([len(grid[0]) // 2 + x, len(grid[0]) // 2 + y])
    X = grid[1]
    Y = grid[2]
    gaussian = grid[0]
    gaussian += gaussian_component(X, Y, amp, sig_x, sig_y, rot, center=cent,)

    return gaussian


def create_gaussian_source(
    grid, comps, amp, x, y, sig_x, sig_y, rot, sides=0, blur=True
):
    """
    Combines Gaussian components on a 2d grid to create a Gaussian source

    takes grid
    side: one-sided or two-sided
    core dominated or lobe dominated
    number of components
    angle of the jet

    Parameters
    ----------
    grid: ndarray
        2dgrid + X and Y meshgrid
    comps: int
        number of components
    amp: 1darray
        amplitudes of components
    x: 1darray
        x positions of components
    y: 1darray
        y positions of components
    sig_x: 1darray
        standard deviations of components in x
    sig_y: 1darray
        standard deviations of components in y
    rot: int
        rotation of the jet in degree
    sides: int
        0 one-sided, 1 two-sided jet
    blur: bool
        use Gaussian filter to blur image

    Returns
    -------
    source: 2darray
        2d grid containing Gaussian source

    Comments
    --------
    components should not have too big gaps between each other
    """
    if sides == 1:
        print(amp)
        comps += comps - 1
        amp = np.append(amp, amp[1:])
        x = np.append(x, -x[1:])
        y = np.append(y, -y[1:])
        sig_x = np.append(sig_x, sig_x[1:])
        sig_y = np.append(sig_y, sig_y[1:])

    for i in range(comps):
        source = add_gaussian(
            grid=grid,
            amp=amp[i],
            x=x[i],
            y=y[i],
            sig_x=sig_x[i],
            sig_y=sig_y[i],
            rot=rot,
        )
    if blur is True:
        source = gaussian_filter(source, sigma=1.5)
    return source


def gauss_paramters():
    """
    Generate a random set of Gaussian parameters.

    Parameters
    ----------
    None

    Returns
    -------
    comps: int
        Number of components
    amp: float
        Amplitude of the core component
    x: array
        x positions of components
    y: array
        y positions of components
    sig_x:
        standard deviation in x
    sig_y:
        standard deviation in y
    rot: int
        rotation in degree
    sides: int
        0 for one-sided and 1 for two-sided jets
    """
    # random number of components between 4 and 9
    comps = np.random.randint(4, 7)  # decrease for smaller images

    # start amplitude between 10 and 1e-3
    amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # if start amp is 0, draw a new number
    while amp_start == 0:
        amp_start = (np.random.randint(0, 100) * np.random.random()) / 10
    # logarithmic decrease to outer components
    amp = np.array([amp_start / np.exp(i) for i in range(comps)])

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
    sig_x = (np.arange(1, comps + 1) - off1) * fac1
    sig_y = (np.arange(1, comps + 1) - off2) * fac2

    # jet rotation
    rot = np.random.randint(0, 360)
    # jet one- or two-sided
    sides = np.random.randint(0, 2)

    return comps, amp, x, y, sig_x, sig_y, rot, sides


def gaussian_source(img_size):
    """
    Creates grid, random Gaussian source parameters and returns an image
    of a Gaussian source.

    Parameters
    ----------
    img_size: int
       number of pixel in x and y

    Returns
    -------
    s: 2darray
       Image containing a simulated Gaussian source.
    """
    grid = create_grid(img_size)
    comps, amp, x, y, sig_x, sig_y, rot, sides = gauss_paramters()
    s = create_gaussian_source(
        grid, comps, amp, x, y, sig_x, sig_y, rot, sides, blur=True
    )
    return s


def create_bundle(img_size, bundle_size):
    """
    Creates a bundle of Gaussian sources.

    Parameters
    ----------
    img_size: int
        pixel size of the image
    bundle_size: int
        number of images in the bundle

    Returns
    -------
    bundle ndarray
        bundle of Gaussian sources
    """
    bundle = np.array([gaussian_source(img_size) for i in range(bundle_size)])
    return bundle


def create_n_bundles(num_bundles, bundle_size, img_size, out_path):
    """
    Creates n bundles of Gaussian sources and saves each to hdf5 file.

    Parameters
    ----------
    num_bundles: int
        number of bundles to be created
    bundle_size: int
        number of sources in one bundle
    img_size: int
        pixel size of the image
    out_path: str
        path to save bundle

    Returns
    -------
    None
    """
    for j in tqdm(range(num_bundles)):
        bundle = create_bundle(img_size, bundle_size)
        save_bundle(out_path, bundle, j)


def get_noise(image, scale, mean=0, std=1):
    """
    Calculate random noise values for all image pixels.

    Parameters
    ----------
    image: 2darray
        2d image
    scale: float
        scaling factor to increase noise
    mean: float
        mean of noise values
    std: float
        standard deviation of noise values

    Returns
    -------
    out: ndarray
        array with noise values in image shape
    """
    return np.random.normal(mean, std, size=image.shape) * scale


def add_noise(bundle, preview=False, num=1):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked. Using 0.05 * max(image) as scaling factor.

    Parameters
    ----------
    bundle: path
        path to hdf5 bundle file
    preview: bool
        enable/disable showing 10 preview images

    Returns
    -------
    bundle_noised hdf5_file
        bundle with noised images
    """
    bundle_noised = np.array(
        [img + get_noise(img, (img.max() * 0.05)) for img in bundle]
    )

    if preview:
        for i in range(num):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

            ax1.set_title(r"Original")
            im1 = ax1.imshow(bundle[i])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im1, cax=cax, orientation="vertical")

            ax2.set_title(r"Noised")
            im2 = ax2.imshow(bundle_noised[i])
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im2, cax=cax, orientation="vertical")
            plt.show()

    return bundle_noised
