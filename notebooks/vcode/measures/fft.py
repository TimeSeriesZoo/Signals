"""Functions for computing FFT measures."""

import cmath

import numpy as np

###################################################################################################
###################################################################################################

def decompose_fft(sig, threshold=0., max_n_waves=np.inf):
    """Decompose the components of an FFT.

    Parameters
    ----------
    sig : 1d array
        Signal to compute FFT from.
    threshold : float, optional, default: 0.
        Minimum power threshold to keep a sine wave.
    max_n_waves : int, optional, default: inf
        Maximum number of sine waves to compute.

    Returns
    -------
    sines : 2d array
        Component sine waves.
    freqs : 1d array
        Frequency values for each sine wave.
    phases : 1d array
        Phase values for each sine wave.
    powers : 1d array
        Power values for each sine wave.

    Notes
    -----
    Adapted from:
    https://stackoverflow.com/questions/59725933/plot-fft-as-a-set-of-sine-waves-in-python
    """

    fft3 = np.fft.fft(sig)
    xs = np.arange(0, 10, 10 / len(sig))
    freqs = np.fft.fftfreq(len(xs), .01)

    sines = np.empty([0, len(xs)])
    phases = []
    powers = []

    for ind, value in enumerate(fft3):

        if ind > max_n_waves:
            break

        power = abs(value)
        phase = cmath.phase(value)
        coeff = 2 if ind == 0 else 1

        if power / len(xs) > threshold:

            sinewave = 1/(len(xs)*coeff/2) * \
                (power * np.cos(freqs[ind]*2*np.pi*xs+phase))

            sines = np.vstack([sines, sinewave])

            phases.append(phase)
            powers.append(power)

    return sines, freqs, np.array(phases), np.array(powers)
