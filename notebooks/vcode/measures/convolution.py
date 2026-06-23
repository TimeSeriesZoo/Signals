"""Functions for computing convolution measures."""

import numpy as np

from neurodsp.utils import create_samples

###################################################################################################
###################################################################################################

def compute_convolution(sig, kernel):
    """Custom function for computing convolution.

    Parameters
    ----------
    sig : 1d array
        Signal.
    kernel : 1d array
        Kernel to convolve.

    Returns
    -------
    convolved : 1d array
        Output signal.
    """

    samps = create_samples(len(kernel))
    convolved = np.ones(len(sig)) * np.nan
    halfwid = int(np.ceil(len(kernel) / 2))
    for ind in range(0, len(sig) - len(kernel) + 1, 1):
        convolved[ind + halfwid] = np.dot(sig[samps + ind], kernel)

    return convolved
