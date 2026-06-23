"""Builder functions for convolution."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts.utils import check_ax
from neurodsp.utils import create_samples

# Import sigviz code
from sigviz.gif import clear_output, animate_plot

###################################################################################################
###################################################################################################

### MEASURES

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


### AXES

def make_axes_convolution():
    """Make axes for combined plot."""

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.3, 1.85, 0.25])
    ax2 = fig.add_axes([0.0, 0.0, 1.85, 0.25])

    return fig, [ax1, ax2]


### PLOTS

def plot_sig_kernel(sig, samps, kernel, ax=None, **kwargs):
    """Plot a signal with an overlying kernel."""

    ax = check_ax(ax, [12, 2])

    ax.plot(sig, color='black', alpha=0.25)
    ax.plot(samps, sig[samps], marker='.', markersize=2.5, linewidth=0, color='blue')
    ax.plot(samps, kernel*25, color='red', alpha=0.75)

    ax.set(xlim=[0, len(sig)], ylim=kwargs.pop('ylim', [-3.5, 3.5]))
    ax.set(xticks=[], yticks=[], xlabel='', ylabel='')


def plot_convolution(samples, convolved, ax=None, **kwargs):
    """Plot the output of a convolution."""

    ax = check_ax(ax, [12, 2])

    ax.plot(samples, convolved, alpha=0.5, color='green')

    ind = np.where(~np.isnan(convolved))[0][-1]
    ax.plot(samples[ind], convolved[ind], '.', markersize=12, color='green', alpha=0.75)

    ax.set(xlim=[0, len(samples)], ylim=kwargs.pop('ylim', [-3.5, 3.5]))
    ax.set(xticks=[], yticks=[], xlabel='', ylabel='')


### BUILDERS: COMPONENTS

def build_kernel_slide(sig, kernel, sleep=0.025):
    """Build kernel slide plot for convolution visualizer."""

    samps = create_samples(len(kernel))

    for ind in range(0, len(sig)-len(kernel) + 1, 1):
        clear_output(wait=True)
        plot_sig_kernel(sig, samps+ind, kernel)
        animate_plot(plt.gcf(), False, ind, sleep=sleep)


def build_convolution_output(sig, kernel, sleep=0.025):
    """Build convolution output for convolution visualizer."""

    samps = create_samples(len(kernel))
    samples = create_samples(len(sig))
    convolved = np.ones(len(sig)) * np.nan
    halfwid = int(np.ceil(len(kernel)/2))

    for ind in range(0, len(sig)-len(kernel) + 1, 1):

        clear_output(wait=True)

        convolved[ind+halfwid] = np.dot(sig[samps+ind], kernel)
        plot_convolution(samples, convolved)

        animate_plot(plt.gcf(), False, ind, sleep=sleep)


### BUILDERS: COMBINED

def build_convolution(sig, kernel,
                      sleep=0.025, save=False, label='conv', **kwargs):
    """Build all plots together for convolution visualizer."""

    samps = create_samples(len(kernel))
    samples = create_samples(len(sig))
    convolved = np.ones(len(sig)) * np.nan
    halfwid = int(np.ceil(len(kernel)/2))

    for ind in range(0, len(sig)-len(kernel) + 1, 1):

        clear_output(wait=True)
        fig, axes = make_axes_convolution()

        convolved[ind+halfwid] = np.dot(sig[samps+ind], kernel)

        plot_sig_kernel(sig, samps+ind, kernel, ax=axes[0], **kwargs)
        plot_convolution(samples, convolved, ax=axes[1], **kwargs)

        animate_plot(fig, save, ind, label=label, sleep=sleep)
