import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

from ring_resonators.cavity_fit.cavity_metrics import g_2_no_delta


FILENAME = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/Pair storage 2025_06_27/Pairs_echo_16hr.txt")
FITTING = False  # add a fit
bin_combine = 100


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 300  # size of x lims
fit_range = 200  # size for fitting
color = 'coral'


# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
coincidence = df["Counts"].to_numpy()
time = df["Time(ps)"].to_numpy(dtype=float)  # unit: ps
time *= 1e-3  # unit: ns
# do binning
if bin_combine > 1:
    num_points = len(coincidence)
    remainder = num_points % bin_combine
    if remainder:
        coincidence = coincidence[:-remainder]
    coincidence = np.sum(coincidence.reshape((-1, bin_combine)),
                         axis=1)
    time = time.reshape((-1, bin_combine))[:, 1]
time_diff = time[1] - time[0]  # spacing of histogram

# fitting
if FITTING:
    model = Model(g_2_no_delta)

    x0_guess = 42.5
    amplitude_guess = 30
    kappa_guess = 5
    g_guess = 2

    # get fit range
    idx_to_fit = np.where(np.abs(time - x0_guess) < (fit_range/2))[0]
    time_to_fit = time[idx_to_fit]
    coincidence_to_fit = coincidence[idx_to_fit]

    params = Parameters()
    params.add('x0', value=x0_guess)
    params.add('amplitude', value=amplitude_guess)
    params.add('kappa', value=kappa_guess)
    params.add('g_factor', value=g_guess/kappa_guess, max=0.5)
    params.add('g', expr='g_factor * kappa')
    res = model.fit(coincidence_to_fit, params, x=time_to_fit)
    print(res.fit_report())

    # res = model.fit(coincidence, x=time,
    #                 x0=x0_guess,
    #                 amplitude=amplitude_guess,
    #                 kappa=kappa_guess,
    #                 g=g_guess)
    # print(res.fit_report())


# plotting
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color, label='Data')

# # TEMP
# x0_guess = 42.5
# amplitude_guess = 30
# kappa_guess = 5
# g_guess = 2
# guess = g_2_no_delta(time, x0_guess, amplitude_guess, kappa_guess, g_guess)
# plt.plot(time, guess, ls=':', color='k', label='Guess')

if FITTING:
    ax.plot(time_to_fit, res.best_fit, 'k--', label='Fit')

    # extract relevant info
    kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    kappa /= 2*np.pi
    kappa *= 1e3  # unit: MHz
    kappa_err = res.params['kappa'].stderr  # unit: GHz
    kappa_err /= 2*np.pi
    kappa_err *= 1e3  # unit: MHz

    label = r"$\kappa$: {:0.3f} $\pm$ {:0.3f} MHz".format(
        kappa, kappa_err)
    t = ax.text(0.05, 0.95, label,
                horizontalalignment='left', verticalalignment='top')
    t.set_transform(ax.transAxes)

if FITTING:
    center = res.params['x0'].value
else:
    center = time[np.argmax(coincidence)]
xlim = (center - xlim_range/2, center + xlim_range/2)
ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Two-Photon Coincidence")
plt.legend(shadow=True)

fig.tight_layout()
fig.show()
