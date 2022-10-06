import astropy.units as u


def pixel2coordinate(header, x_pixel, y_pixel, relative=True):
    """Transform pixel values of one image to astronomical units

    Parameters
    ----------
    header: header of MOVAJE fits file
        header of one image
    x_pixel: floats
        x positions
    y_pixel: floats
        y positions

    Returns
    -------
    x, y: tuple of floats
        coordinates in mas (milli arc second)
    """
    x_ref_pixel = header["CRPIX1"]  # center in pixel
    x_ref_value = (header["CRVAL1"] * u.degree).to(u.mas)  # center in mas
    x_inc = (header["CDELT1"] * u.degree).to(u.mas)  # increament in mas per pixel

    y_ref_pixel = header["CRPIX2"]  # center as pixel
    y_ref_value = (header["CRVAL2"] * u.degree).to(u.mas)  # center in mas
    y_inc = (header["CDELT2"] * u.degree).to(u.mas)  # increament in mas per pixel

    x = ((x_pixel - x_ref_pixel) * x_inc).astype(float)
    y = ((y_pixel - y_ref_pixel) * y_inc).astype(float)

    if not relative:
        x += x_ref_value
        y += y_ref_value

    return x, y
