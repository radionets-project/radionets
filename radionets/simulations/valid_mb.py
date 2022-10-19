import torch


def create_rot_mat(alpha):
    rot_mat = torch.tensor(
        [[torch.cos(alpha), -torch.sin(alpha)], [torch.sin(alpha), torch.cos(alpha)]]
    )
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot, center=None):
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(torch.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2
    gauss = flux * torch.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    return gauss


def create_grid(pixel):
    x = torch.linspace(0, pixel - 1, steps=pixel)
    y = torch.linspace(0, pixel - 1, steps=pixel)
    X, Y = torch.meshgrid(x, y)
    X.unsqueeze_(0)
    Y.unsqueeze_(0)
    mesh = torch.cat((X, Y))
    grid = torch.tensor((torch.zeros(X.shape) + 1e-10))
    grid = torch.cat((grid, mesh))
    return grid
