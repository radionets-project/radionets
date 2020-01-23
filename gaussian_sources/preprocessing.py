import numpy as np


def split_real_imag(array):
    """
    takes a complex array and returns the real and the imaginary part
    """
    return array.real, array.imag


def split_amp_phase(array):
    """
    takes a complex array and returns the amplitude and the phase
    """
    amp = np.abs(array)
    phase = np.angle(array)
    return amp, phase


def mean_and_std(array):
    return array.mean(), array.std()


def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 2, 0)
