"""Builder functions for FFT visualizers."""

import numpy as np
import matplotlib.pyplot as plt

from vcode.plts.fft import plot_sines, plot_recomb, plot_powers, plot_phases
from vcode.plts.utils import clear_output, animate_plot

###################################################################################################
###################################################################################################

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

### BUILDERS

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
