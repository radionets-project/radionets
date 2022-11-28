import astropy.units as u


def pixel2coordinate(header, x_pixel: float, y_pixel: float, img_size: int, relative: bool = True, units: bool = True):
    """Transform pixel values of one image to astronomical units

    Parameters
    ----------
    header: header of MOVAJE fits file
        header of one image
    x_pixel: floats
        x positions
    y_pixel: floats
        y positions
    img_size: int
        size of the input image of x_pixel and y_pixel
    relative: bool
        calculate relative or absolute position
    units: bool
        wether to return with units or without

    Returns
    -------
    x, y: tuple of floats
        coordinates, in mas (milliarcsecond) if units is set True
    """
    size_proportion = header["NAXIS1"] / 2 -  img_size / 2

    x_ref_pixel = header["CRPIX1"]  # center in pixel
    x_ref_value = (header["CRVAL1"] * u.degree).to(u.mas)  # center in mas
    x_inc = (header["CDELT1"] * u.degree).to(u.mas)  # increament in mas per pixel

    y_ref_pixel = header["CRPIX2"]  # center as pixel
    y_ref_value = (header["CRVAL2"] * u.degree).to(u.mas)  # center in mas
    y_inc = (header["CDELT2"] * u.degree).to(u.mas)  # increament in mas per pixel

    x = ((x_pixel + size_proportion - x_ref_pixel) * x_inc).astype(float)
    y = ((y_pixel + size_proportion - y_ref_pixel) * y_inc).astype(float)

    if not relative:
        x += x_ref_value
        y += y_ref_value

    if not units:
        x = x.value
        y = y.value

    return x, y