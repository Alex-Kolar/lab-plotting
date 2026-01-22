import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model

from ring_resonators.cavity_fit.cavity_metrics import g_2_exp_bg, g_2_no_delta


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device_mk_3/4K/2025_07_24/pairs")
integration_time = 5 * 60  # units: s
current_to_power = {100: 22.3,  # power (in uW) at BS before sample
                    150: 49.3,
                    200: 75.2,
                    250: 103.7,
                    300: 130.5}

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color = 'cornflowerblue'
color_area = 'mediumpurple'
color_coincidence = 'coral'
x_range = 50
PLOT_ALL_COINCIDENCE = True  # plot all fitted coincidence curves


# efficiencies (measured)
power_BS_ref = 103.7  # unit: uW
power_input_ref = 7030  # unit: uW
power_output_ref = 10.13  # unit: uW
efficiency_input_feedthrough = 0.823  # purple feedthrough
efficiency_output_feedthrough = 0.849  # green feedthrough

# input laser light
pump_multiplier = power_input_ref / power_BS_ref
efficiency_total = power_output_ref / power_input_ref
efficiency_chip_coupling = np.sqrt(efficiency_total / efficiency_input_feedthrough / efficiency_output_feedthrough)
print('Efficiency of Chip Coupling:', efficiency_chip_coupling)
efficiency_input = efficiency_input_feedthrough * efficiency_chip_coupling
# efficiencies for output light
# path taken by photons at 195.2 THz
efficiency_52 = efficiency_chip_coupling * efficiency_output_feedthrough * 0.7274 * 0.4532
# path taken by photons at 194.8 THz
efficiency_48 = efficiency_chip_coupling * efficiency_output_feedthrough * 0.4790 * 0.9260
efficiency_pairs = efficiency_48 * efficiency_52


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


# gather data for coincidences
coincidence_files = sorted(glob.glob('pairs_5min_*', root_dir=DATA_DIR))
dfs_coinc = []  # store all dataframes
powers = []  # store all pump powers
int_times = []  # store all integration times
all_res = []  # store all fit result objects

# set up model for fitting
# model = Model(g_2_exp_bg)
model = Model(g_2_no_delta)

for filename in coincidence_files:
    file_str = os.path.splitext(filename)[0]
    file_path = os.path.join(DATA_DIR, filename)

    # get metadata from file
    file_info = file_str.split('_')  # 'pairs', 5mW, XXXmA
    current = int(file_info[2][:3])
    int_time = float(file_info[1][:1])
    power_BS = current_to_power[current]
    power_pump = pump_multiplier * power_BS
    power_on_chip = power_pump * efficiency_input
    powers.append(power_on_chip)
    int_times.append(int_time)

    # read coincidence data
    df = pd.read_csv(file_path, sep='\t')
    dfs_coinc.append(df)
    time = df['Time(ps)'] * 1e-3  # convert to ns
    coincidences = df['Counts']

    # fitting
    print(f'Fitting {current} mA pump')
    # x0_guess = time[np.argmax(coincidences)]
    # amplitude_guess = np.max(coincidences)
    # bg_guess = np.mean(coincidences[:100])
    #
    # T_1_guess = 3
    # T_2_guess = 3
    # res = model.fit(coincidences, x=time,
    #                 x0=x0_guess,
    #                 bg=bg_guess,
    #                 amplitude=amplitude_guess,
    #                 T_1=T_1_guess,
    #                 T_2=T_2_guess)
    # all_res.append(res)
    #
    # T_1 = res.params['T_1'].value
    # T_1_err = res.params['T_1'].stderr
    # T_2 = res.params['T_2'].value
    # T_2_err = res.params['T_2'].stderr
    #
    # kappa_1 = 1/(T_1 * 2 * np.pi) * 1e3  # convert to MHz
    # kappa_1_err = (T_1_err / T_1) * kappa_1
    # kappa_2 = 1/(T_2 * 2 * np.pi) * 1e3  # convert to MHz
    # kappa_2_err = (T_2_err / T_2) * kappa_2
    #
    # print(f'\tT_1: {T_1:.2f} +/- {T_1_err:.2f} ns ({kappa_1:.2f} +/- {kappa_1_err:.2f} MHz)')
    # print(f'\tT_2: {T_2:.2f} +/- {T_2_err:.2f} ns ({kappa_2:.2f} +/- {kappa_2_err:.2f} MHz)')

    # fitting
    x0_guess = time[np.argmax(coincidences)]
    amplitude_guess = np.max(coincidences)
    kappa_guess = 1
    g_guess = 0.125
    res = model.fit(coincidences, x=time,
                    x0=x0_guess,
                    amplitude=amplitude_guess,
                    kappa=kappa_guess,
                    g=g_guess)
    all_res.append(res)

    kappa = res.params['kappa'].value
    print('correlation time:', np.pi/kappa)

    # plot data for coincidences
    if PLOT_ALL_COINCIDENCE:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=400)

        plt.bar(time, coincidences, color=color_coincidence,
                label='Data')
        plt.plot(time, res.best_fit,
                 color='k', ls='--',
                 label='Fit')

        # plt.title(rf'Coincidences with {power_on_chip:.0f} $\mathrm{{\mu}}$W Pump Power On-Chip')
        plt.title('Two-Photon Coincidence')
        plt.xlabel('Time (ns)')
        plt.ylabel('Coincidences')
        plt.legend()

        x_start = res.params['x0'].value - (x_range / 2)
        x_end = res.params['x0'].value + (x_range / 2)
        plt.xlim((x_start, x_end))

        plt.tight_layout()
        plt.show()


# gather all fit data
all_amps = [res.params['amplitude'].value for res in all_res]
all_amps_err = [res.params['amplitude'].stderr for res in all_res]
all_integral = []
for res in all_res:
    # extract integral (numerically)
    only_coincidence = res.best_fit - res.params['amplitude'].value  # everything above bg
    integral = np.sum(only_coincidence)
    integral /= integration_time  # convert to pairs/s
    integral /= efficiency_pairs  # get on chip pair rate
    all_integral.append(integral)


# fit amplitudes versus pump power
model = Model(quadratic)
res = model.fit(all_amps, x=powers,
                a=0.16)
print(res.fit_report())


# fit areas versus pump power
res_area = model.fit(all_integral, x=powers,
                     a=0.16)
print(res_area.fit_report())


# calculate pair rate
pair_rate_1mW = res_area.params['a'].value * 1e6  # convert from uW^2 to mW^2
print('Pair rate (s^-1 mW^-2):', pair_rate_1mW)


# plot of amplitudes versus pump power
powers_for_fit = np.linspace(min(powers), max(powers), 100)
plt.errorbar(powers, all_amps, yerr=all_amps_err,
             color='cornflowerblue', ls='', marker='o', capsize=3,
             label='Fitted Data')
plt.plot(powers_for_fit, quadratic(powers_for_fit, **res.best_values),
         color='k', ls='--',
         label='Fitted Trend')

plt.title('Coincidences versus Pump Power')
plt.xlabel(r'Pump Power ($\mathrm{\mu}$W)')
plt.ylabel('Coincidence Fit Amplitude')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# plot of areas versus pump power
fig, ax = plt.subplots(figsize=(4, 2), dpi=400)

plt.errorbar(powers, all_integral,
             color='mediumpurple', ls='', marker='o', capsize=3,
             label='Data')
plt.plot(powers_for_fit, quadratic(powers_for_fit, **res_area.best_values),
         color='k', ls='--',
         label='Fit')

# # add text
# label = (rf'$a$ = {res_area.params['a'].value * 1e6:.0f} $\pm$ {res_area.params['a'].stderr * 1e6:.0f} '
#            '$\mathrm{{s}}^{{-1}}\mathrm{{mW}}^{{-2}}$')
# t = ax.text(0.95, 0.05, label,
#             horizontalalignment='right', verticalalignment='bottom')
# t.set_transform(ax.transAxes)

plt.title('Coincidence Area versus Pump Power')
plt.xlabel(r'Pump Power On-Chip ($\mathrm{\mu}$W)')
plt.ylabel(r'Coincidence Area')
plt.legend()

plt.tight_layout()
plt.show()
