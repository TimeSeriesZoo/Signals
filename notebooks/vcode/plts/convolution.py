"""Convolution plot functions."""

import numpy as np

from neurodsp.plts.utils import check_ax

###################################################################################################
###################################################################################################

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
