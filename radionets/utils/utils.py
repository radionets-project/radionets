import numpy as np
import torch
import torch.nn.functional as F


def symmetry(image, key):
    """
    Symmetry function to complete the images

    Parameters
    ----------
    image : torch.Tensor
        (stack of) half images

    Returns
    -------
    torch.Tensor
        quadratic images after utilizing symmetry
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    if len(image.shape) == 3:
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    half_image = image.shape[-1] // 2
    upper_half = image[:, :, : half_image + 1, :].clone()
    a = torch.rot90(upper_half, 2, dims=[-2, -1])

    image[:, 0, half_image + 1 :, 1:] = a[:, 0, 1:-1, :-1]
    image[:, 0, half_image + 1 :, 0] = a[:, 0, 1:-1, -1]
    image[:, 0, half_image : half_image + 1, half_image + 1 :] = a[
        :, 0, 0:1, half_image:-1
    ]

    if key == "unc":
        image[:, 1, half_image + 1 :, 1:] = a[:, 1, :-1, :-1]
        image[:, 1, half_image + 1 :, 0] = a[:, 1, :-1, -1]
    else:
        image[:, 1, half_image + 1 :, 1:] = -a[:, 1, 1:-1, :-1]
        image[:, 1, half_image + 1 :, 0] = -a[:, 1, 1:-1, -1]
        image[:, 1, half_image : half_image + 1, half_image + 1 :] = -a[
            :, 1, 0:1, half_image:-1
        ]

    return image


def apply_symmetry(img_dict, overlap=1):
    """
    Pads and applies symmetry to half images. Takes a dict as input

    Parameters
    ----------
    img_dict : dict
        input dict which contains the half images

    Returns
    -------
    dict
        input dict with quadratic images
    """
    for key in img_dict:
        if key != "indices":
            if isinstance(img_dict[key], np.ndarray):
                img_dict[key] = torch.tensor(img_dict[key])
            half_image = img_dict[key].shape[-1] // 2
            output = F.pad(
                input=img_dict[key],
                pad=(0, 0, 0, half_image - overlap),
                mode="constant",
                value=0,
            )
            output = symmetry(output, key)
            img_dict[key] = output

    return img_dict
