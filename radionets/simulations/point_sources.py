from radionets.simulations.gaussians import create_grid, create_rot_mat
import numpy as np
from tqdm import tqdm
import h5py


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot, center=None):
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
        full-width-half-maximum in x direction (sigma_x)
    y_fwhm: float
        full-width-half-maximum in y direction (sigma_y)
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
    params = np.array([x_0, y_0, x_fwhm, y_fwhm])
    return gauss, params


def gauss_parameters():
    # random number of components between 4 and 9
    comps = np.random.randint(4, 7)  # decrease for smaller images

    rng = np.random.default_rng()
    # start amplitude between 1 and 10
    amp_start = rng.uniform(5, 10)
    # logarithmic decrease to outer components
    amp = np.array([amp_start / (np.exp(i * 0.6)) for i in range(comps)])

    # linear distance bestween the components
    x = np.arange(0, comps) * 5
    y = np.zeros(comps)

    sig_start = rng.uniform(1, 1.2)
    fac = rng.uniform(0.25, 0.5)
    sig = (np.arange(0, comps) * fac) + sig_start

    # jet rotation
    rot = np.random.randint(0, 360)
    # jet one- or two-sided
    sides = np.random.randint(0, 2)

    return comps, amp, x, y, sig, sig, rot, sides


def add_gaussian(grid, amp, x, y, sig_x, sig_y, rot):
    cent = np.array([len(grid[0]) // 2 + x, len(grid[0]) // 2 + y])
    X = grid[1]
    Y = grid[2]
    gaussian = grid[0]
    comp, params = gaussian_component(
        X,
        Y,
        amp,
        sig_x,
        sig_y,
        rot,
        center=cent,
    )
    gaussian += comp

    return gaussian, params


def create_gaussian_source(grid, comps, amp, x, y, sig_x, sig_y, rot, sides=0):
    if sides == 1:
        comps += comps - 1
        amp = np.append(amp, amp[1:])
        x = np.append(x, -x[1:])
        y = np.append(y, -y[1:])
        sig_x = np.append(sig_x, sig_x[1:])
        sig_y = np.append(sig_y, sig_y[1:])

    params = np.array([])
    for i in range(comps):
        source, param = add_gaussian(
            grid=grid,
            amp=amp[i],
            x=x[i],
            y=y[i],
            sig_x=sig_x[i],
            sig_y=sig_y[i],
            rot=rot,
        )
        params = np.append(params, param)
    return source, params


def gaussian_source(grid):
    comps, amp, x, y, sig_x, sig_y, rot, sides = gauss_parameters()
    s, params = create_gaussian_source(grid, comps, amp, x, y, sig_x, sig_y, rot, sides)
    return s, params


def gauss(img_size, mx, my, sx, sy, amp=0.01):
    x = np.arange(img_size)[None].astype(np.float)
    y = x.T
    return amp * np.exp(-((y - my) ** 2) / sy).dot(np.exp(-((x - mx) ** 2) / sx))


def create_gauss(img, num_sources, source_list, img_size=63):
    mx = np.random.randint(1, img_size, size=(num_sources))
    my = np.random.randint(1, img_size, size=(num_sources))
    rng = np.random.default_rng()
    amp = rng.uniform(1, 10, num_sources)
    sx = np.random.randint(
        round(1 / 8 * (img_size ** 2) / 720),
        1 / 2 * (img_size ** 2) / 360,
        size=(num_sources),
    )
    sy = sx
    idx = []
    for n in range(num_sources):
        if img[mx[n], my[n]] <= 5e-10:
            g = gauss(img_size, mx[n], my[n], sx[n], sy[n], amp[n])
            img += g
        else:
            idx.append(n)
    mx = np.delete(mx, idx)
    my = np.delete(my, idx)
    sx = np.delete(sx, idx)
    sy = np.delete(sy, idx)
    # assert np.isnan(img).any() == False
    if source_list:
        return img, [mx, my], [sx, sy]
    else:
        return img


def create_point_source_img(
    img_size, bundle_size, num_bundles, path, option, extended=False
):
    for num_bundle in tqdm(range(num_bundles)):
        with h5py.File(path + "/fft_" + option + str(num_bundle) + ".h5", "w") as hf:
            for num_img in range(bundle_size):
                grid = create_grid(img_size, 1)
                num_point_sources = np.random.randint(2, 5)

                if extended:
                    gs, params_extended = gaussian_source(grid[0])
                    x_off = np.random.randint(1, 20)
                    y_off = np.random.randint(1, 20)
                    params_extended[0::4] += x_off
                    params_extended[1::4] += y_off
                    gs = np.pad(gs, ((y_off, 0), (x_off, 0)), constant_values=(1e-10))[
                        :-y_off, :-x_off
                    ]
                    tag_ext = np.ones(len(params_extended) // 4)

                g, p_point, s_point = create_gauss(
                    gs, num_point_sources, True, img_size
                )

                tag_point = np.zeros(len(p_point[0]))

                comps = np.array(
                    [
                        np.concatenate([p_point[0], params_extended[0::4]]),
                        np.concatenate([p_point[1], params_extended[1::4]]),
                        np.concatenate([s_point[0], params_extended[2::4]]),
                        np.concatenate([s_point[1], params_extended[3::4]]),
                        np.concatenate([tag_point, tag_ext]),
                    ]
                )

                # crop image size
                mask = (
                    (comps[0] >= 0)
                    & (comps[0] <= img_size - 1)
                    & (comps[1] >= 0)
                    & (comps[1] <= img_size - 1)
                )
                list_x = comps[0][mask]
                list_y = comps[1][mask]
                list_sx = comps[2][mask]
                list_sy = comps[3][mask]
                list_tag = comps[4][mask]
                assert (
                    list_x.shape
                    == list_y.shape
                    == list_sx.shape
                    == list_sy.shape
                    == list_tag.shape
                )

                source_list = np.array([list_x, list_y, list_sx, list_sy, list_tag])
                g_fft = np.array(np.fft.fftshift(np.fft.fft2(g.copy())))
                hf.create_dataset("x" + str(num_img), data=g_fft)
                hf.create_dataset("y" + str(num_img), data=g)
                hf.create_dataset("z" + str(num_img), data=source_list)

        hf.close()
