"""Builder functions for timeseries plots."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.utils.yielders import step_over_time

# Import sigviz code
from sigviz.plts import plot_timeseries, plot_spectra
from sigviz.gif import clear_output, animate_plot

###################################################################################################
###################################################################################################

### AXES

def make_axes_timeseries():
    """Make axes for combined plot."""

    fig = plt.figure()
    ax1 = fig.add_axes([0.0, 0.6, 1.2, 0.5])
    ax2 = fig.add_axes([1.3, 0.6, 0.5, 0.5])

    return fig, [ax1, ax2]

### BUILDERS

def build_timeseries(sig, fs, n_build=np.inf,
                     sleep=0.05, save=False, label='viz'):
    """Build all plots together."""

    size = 750
    step = 2
    start = 2000

    sig_yielder = step_over_time(sig, start=start, size=size, step=step)

    for ind in range(n_build):

        clear_output(wait=True)

        fig, axes = make_axes_timeseries()

        spect_sig = sig[start + step * ind-2000:start + step * ind+2000]
        freqs, powers = trim_spectrum(*compute_spectrum(spect_sig, fs=fs), [1, 50])

        plot_timeseries(next(sig_yielder), ax=axes[0])
        plot_spectra(freqs, powers, log_freqs=True, ax=axes[1])

        animate_plot(fig, save, ind, label=label, sleep=sleep)
