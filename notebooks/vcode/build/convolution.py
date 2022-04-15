"""Builder functions for convolution."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.utils import create_samples

from vcode.plts.convolution import *
from vcode.plts.utils import clear_output, animate_plot

###################################################################################################
###################################################################################################

### AXES

def make_axes_convolution():
    """Make axes for combined plot."""

    fig = plt.figure()
    #ax1 = fig.add_axes([0.0, 0.5, 1.7, 0.45])
    #ax2 = fig.add_axes([0.0, 0.0, 1.7, 0.45])

    ax1 = fig.add_axes([0.0, 0.3, 1.85, 0.25])
    ax2 = fig.add_axes([0.0, 0.0, 1.85, 0.25])

    return fig, [ax1, ax2]


### BUILDERS

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

        animate_plot(fig, save, ind, label='03_' + label, sleep=sleep)
