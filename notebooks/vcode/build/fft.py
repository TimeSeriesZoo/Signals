"""Builder functions for FFT visualizers."""

import cmath

import numpy as np
import matplotlib.pyplot as plt

from vcode.plts.fft import plot_sines, plot_recomb, plot_powers, plot_phases

# Import sigviz code
from sigviz.gif import clear_output, animate_plot

###################################################################################################
###################################################################################################

## MEASURES

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


### AXES

def make_axes_params():
    """Make axes for the parameters part of the FFT visualizer."""

    fig = plt.figure()
    ax1 = fig.add_axes([0, 0.0, 0.5, 0.5])
    ax2 = fig.add_axes([0, 0.6, 0.5, 0.5], polar=True)

    return fig, [ax1, ax2]


def make_axes_sigs():
    """Make axes for the signals part of the FFT visualizer."""

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.6, 1.3, 0.5])
    ax2 = fig.add_axes([0.0, 0.0, 1.3, 0.5])

    return fig, [ax1, ax2]


def make_axes_fft():
    """Make axes for combined FFT visualizer."""

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.6, 1.3, 0.5])
    ax2 = fig.add_axes([0.0, 0.0, 1.3, 0.5])
    ax3 = fig.add_axes([1.5, 0.0, 0.5, 0.5])
    ax4 = fig.add_axes([1.5, 0.6, 0.5, 0.5], polar=True)

    return fig, [ax1, ax2, ax3, ax4]

## PLOTS

...

### BUILDERS: COMPONENTS

def build_sines(sines, n_build=np.inf, sleep=0.05):
    """Build the sine wave plot (animated)."""

    for ind in range(min(sines.shape[0], n_build)):

        clear_output(wait=True)
        plot_sines(sines[0:ind, :])
        animate_plot(plt.gcf(), False, ind, sleep=sleep)


def build_recomb(sines, data, n_build=np.inf, sleep=0.05):
    """Build the recombined wave plot (animated)."""

    for ind in range(min(sines.shape[0], n_build)):

        clear_output(wait=True)
        plot_recomb(sines[0:ind, :], data)
        animate_plot(plt.gcf(), False, ind, sleep=sleep)


def build_powers(freqs, powers, n_build=np.inf, sleep=0.05):
    """Build the powers plot (animated)."""

    for ind in range(min(len(powers), n_build)):

        clear_output(wait=True)
        plot_powers(freqs[0:ind], powers[0:ind])
        animate_plot(plt.gcf(), False, ind, sleep=sleep)


def build_phases(phases, n_build=np.inf, sleep=0.05):
    """Build the phase plot (animated)."""

    for ind in range(min(len(phases), n_build)):

        clear_output(wait=True)
        plot_phases(phases[0:ind])
        animate_plot(plt.gcf(), False, ind, sleep=sleep)


def build_params(freqs, phases, powers, n_build=np.inf, sleep=0.05):
    """Build param plots together."""

    for ind in range(min(n_build, len(phases))):

        clear_output(wait=True)

        fig, axes = make_axes_params()

        plot_powers(freqs[0:ind], powers[0:ind], ax=axes[0])
        plot_phases(phases[0:ind], ax=axes[1])
        animate_plot(fig, False, ind, sleep=sleep)


def build_sigs(sines, data, n_build=np.inf, sleep=0.05):
    """Build signal plots together."""

    for ind in range(min(n_build, sines.shape[0])):

        clear_output(wait=True)

        fig, axes = make_axes_sigs()

        plot_sines(sines[0:ind, :], ax=axes[0])
        plot_recomb(sines[0:ind, :], data, ax=axes[1])
        animate_plot(fig, False, ind, sleep=sleep)


### BUILDERS: COMBINED

def build_fft(sines, data, freqs, phases, powers, n_build=np.inf,
              sleep=0.05, save=False, label='fft'):
    """Build all plots together."""

    for ind in range(min(n_build, sines.shape[0])):

        clear_output(wait=True)

        fig, axes = make_axes_fft()

        plot_sines(sines[0:ind, :], ax=axes[0])
        plot_recomb(sines[0:ind, :], data, ax=axes[1])
        plot_powers(freqs[0:ind], powers[0:ind], log_powers=True, ax=axes[2])
        plot_phases(phases[0:ind], ax=axes[3])

        animate_plot(fig, save, ind, label=label, sleep=sleep)
