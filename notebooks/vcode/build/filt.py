"""Builder functions for filters."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.filt.fir import design_fir_filter, apply_fir_filter
from neurodsp.filt.utils import compute_frequency_response
from neurodsp.plts.filt import plot_frequency_response, plot_impulse_response

from vcode.plts.base import plot_data
from vcode.plts.filt import add_filter_text

# Import sigviz code
from sigviz.gif import clear_output, animate_plot
from sigviz.utils import incrementer

###################################################################################################
###################################################################################################

def passer(*args, **kwargs):
    """Helper function to pass through for custom styling."""
    pass


### AXES

def make_axes_filter_properties():
    """Make axes for filter properties plot."""

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.8, 0.15, 0.75])
    ax1.axis('off')
    ax2 = fig.add_axes([0.325, 0.8, 0.75, 0.75])
    ax3 = fig.add_axes([1.3, 0.8, 0.75, 0.75])

    return fig, [ax1, ax2, ax3]


def make_axes_filter_output():
    """Make axes for filter output plot."""

    fig = plt.figure()

    ax1 = fig.add_axes([0.0, 0.8, 0.35, 0.6])

    ax2 = fig.add_axes([0.45, 1.15, 1.15, 0.325])
    ax3 = fig.add_axes([0.45, 0.75, 1.15, 0.325])

    ax4 = fig.add_axes([1.725, 0.75, 0.15, 0.7])
    ax4.axis('off')

    return fig, [ax1, ax2, ax3, ax4]


### BUILDERS

def build_filter_properties(fs, pass_type, f_range, n_cycles, field, values,
                            fr_xlim=[0, 150], fr_ylim=[-100, 0], imp_xlim=None, imp_ylim=None,
                            sleep=0.05, save=False, label='filter_properties'):
    """Build function for filter properties."""

    if field == 'bandwidth':
        cen = f_range

    inc = incrementer()
    for value in values:

        if field == 'f_range':
            f_range = value
        if field == 'n_cycles':
            n_cycles = value
        if field == 'bandwidth':
            f_range = (cen-value, cen+value)

        filt_coefs = design_fir_filter(fs, pass_type, f_range, n_cycles)
        f_db, db = compute_frequency_response(filt_coefs, 1, fs)

        clear_output(wait=True)

        fig, axes = make_axes_filter_properties()
        add_filter_text(axes[0], pass_type, f_range, n_cycles)
        plot_frequency_response(f_db, db, ax=axes[1],
                                xlim=fr_xlim, ylim=fr_ylim,
                                custom_styler=passer)
        plot_impulse_response(fs, filt_coefs, ax=axes[2],
                              xlim=imp_xlim, ylim=imp_ylim,
                              custom_styler=passer)

        animate_plot(fig, save, next(inc), label=label, sleep=sleep)


def build_filter_ouput(sig, fs, pass_type, f_range, n_cycles, field, values,
                       sleep=0.50, save=False, label='filter_output'):
    """Build function for filter ouput."""

    if field == 'bandwidth':
        cen = f_range

    x_vals = np.arange(len(sig))

    inc = incrementer()
    for value in values:

        if field == 'f_range':
            f_range = value
        if field == 'n_cycles':
            n_cycles = value
        if field == 'bandwidth':
            f_range = (cen-value, cen+value)

        filt_coefs = design_fir_filter(fs, pass_type, f_range, n_cycles)
        filt_output = apply_fir_filter(sig, filt_coefs)

        clear_output(wait=True)

        fig, (ax1, ax2, ax3, ax4) = make_axes_filter_output()

        plot_data(filt_coefs, color='black', ax=ax1)
        ax1.set(xticks=[], yticks=[]);
        ax1.set_title('Filter Kernel')

        ax2.plot(x_vals, sig, color='black', alpha=0.85)
        ax2.set(xticks=[], yticks=[]);
        ax2.set_ylabel('Input')
        ax2.set_xlim(x_vals[0], x_vals[-1])
        ax2.set_ylim([-1.1, 1.1])

        ax3.plot(x_vals, filt_output, color='green', alpha=0.85)
        ax3.set(xticks=[], yticks=[]);
        ax3.set_ylabel('Output')
        ax3.set_xlim(x_vals[0], x_vals[-1])
        ax3.set_ylim([-1.1, 1.1])

        add_filter_text(ax4, pass_type, f_range, n_cycles)

        animate_plot(fig, save, next(inc), label=label, sleep=sleep)
