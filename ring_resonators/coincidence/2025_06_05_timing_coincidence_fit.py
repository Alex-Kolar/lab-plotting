import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from lmfit.models import GaussianModel

from ring_resonators.cavity_fit.cavity_metrics import g_2_exp


FILENAME = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/coincidence_2025_06_05/Correlation_2025-06-05_14-56-20.txt")
FILENAME_PULSE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/New_mounted_device/10mK/06052025/SDS00001.csv")
FITTING = True  # add a fit
integration_time = 600  # units: s


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 20  # size of x lims
fit_range = 500  # size for fitting
color = 'coral'
color_pulse = 'cornflowerblue'


# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
coincidence = df["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]  # spacing of histogram

# get pulse data
df_pulse = pd.read_csv(FILENAME_PULSE, header=10, skiprows=[11])
pulse = df_pulse["CH1"]  # unit: W (10^2 V/W, 20 dB attenuation)
time_pulse = df_pulse["Source"]  # unit: s
time_pulse *= 1e9  # unit: ns


# fitting
if FITTING:
    # fit pulse
    model_pulse = GaussianModel()
    res_pulse = model_pulse.fit(pulse, x=time_pulse,
                                center=10, sigma=10, amplitude=20)
    sigma_pulse = res_pulse.params['sigma'].value
    print("Pulse Fitting:")
    print(res_pulse.fit_report())

    # fit coincidences
    model = Model(g_2_exp) + GaussianModel(prefix='bg_')

    x0_guess = 32
    amplitude_guess = 20
    T_1_guess = 3
    T_2_guess = 3
    bg_amplitude_guess = 50
    bg_sigma_guess = np.sqrt(2) * sigma_pulse

    # get fit range
    idx_to_fit = np.where(np.abs(time - x0_guess) < (fit_range/2))[0]
    time_to_fit = time[idx_to_fit]
    coincidence_to_fit = coincidence[idx_to_fit]

    params = Parameters()
    params.add('x0', value=x0_guess)
    params.add('amplitude', value=amplitude_guess)
    params.add('T_1', value=T_1_guess)
    params.add('T_2', value=T_2_guess)
    params.add('bg_center', expr='x0')
    params.add('bg_amplitude', value=bg_amplitude_guess)
    params.add('bg_sigma', value=bg_sigma_guess, vary=False)

    res = model.fit(coincidence_to_fit, params, x=time_to_fit)
    print("Coincidence Fitting:")
    print(res.fit_report())

    # # extract integral (numerically)
    # only_coincidence = res.best_fit - res.params['amplitude'].value  # everything above bg
    # integral = np.sum(only_coincidence[:-1])  # riemann sum
    # integral /= integration_time  # convert to pairs/s
    # print('Integral:', integral)


# plotting pulse
fig, ax = plt.subplots()
ax.plot(time_pulse, pulse, color=color_pulse, label='Data')

if FITTING:
    ax.plot(time_pulse, res_pulse.best_fit, 'k--', label='Fit')

ax.set_xlabel('Time (ns)')
ax.set_ylabel('Optical Power (W)')
ax.set_title("Coincidence Pump Pulse")
ax.legend(shadow=True)

fig.tight_layout()
fig.show()

# plotting coincidence
fig, ax = plt.subplots()
ax.bar(time, coincidence, width=time_diff, color=color, label='Data')

if FITTING:
    ax.plot(time_to_fit, res.best_fit, 'k--', label='Fit')

    # extract relevant info
    T_1 = res.params['T_1'].value
    T_1_err = res.params['T_1'].stderr
    label = rf"$\tau_1$ = {T_1:.2f} $\pm$ {T_1_err:.2f} ns"

    kappa_1 = 1e3 / (2*np.pi*T_1)
    label += "\n"
    label += rf"($\kappa_1$ = {kappa_1:.2f} MHz)"

    T_2 = res.params['T_2'].value
    T_2_err = res.params['T_2'].stderr
    label += "\n"
    label += rf"$\tau_2$ = {T_2:.2f} $\pm$ {T_2_err:.2f} ns"

    kappa_2 = 1e3 / (2 * np.pi * T_2)
    label += "\n"
    label += rf"($\kappa_2$ = {kappa_2:.2f} MHz)"

    # # extract relevant info
    # kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    # kappa /= 2*np.pi
    # kappa *= 1e3  # unit: MHz
    # kappa_err = res.params['kappa'].stderr  # unit: GHz
    # kappa_err /= 2*np.pi
    # kappa_err *= 1e3  # unit: MHz
    #
    # label = r"$\kappa$: {:0.3f} $\pm$ {:0.3f} MHz".format(
    #     kappa, kappa_err)

    t = ax.text(0.05, 0.95, label,
                horizontalalignment='left', verticalalignment='top')
    t.set_transform(ax.transAxes)

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
