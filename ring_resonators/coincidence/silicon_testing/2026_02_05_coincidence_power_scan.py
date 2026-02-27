import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model

from ring_resonators.cavity_fit.cavity_metrics import g_2_exp_bg, g_2_single_exp_bg


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Silicon_test_devices/mk_5/chip_2/2026_02_05/pair_generation')
integration_time = 60  # units: s

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color = 'cornflowerblue'
color_area = 'mediumpurple'
color_coincidence = 'coral'
x_range = 50
PLOT_ALL_COINCIDENCE = False  # plot all fitted coincidence curves

# parameters from experiment
input_power = 1.5e3  # unit: uW
output_power = 32  # unit: uW
total_efficiency = output_power / input_power
efficiency_input = np.sqrt(total_efficiency)
efficiency_output = efficiency_input
efficiency_pairs = efficiency_output ** 2
power_on_chip_ref = input_power * efficiency_input


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


# gather data for coincidences
coincidence_files = sorted(glob.glob('silicon_pairs_*.txt', root_dir=DATA_DIR))
dfs_coinc = []  # store all dataframes
powers = []  # store all pump powers
all_res = []  # store all fit result objects

# set up model for fitting
# model = Model(g_2_exp_bg)
model = Model(g_2_single_exp_bg)

for filename in coincidence_files:
    file_str = os.path.splitext(filename)[0]
    file_path = os.path.join(DATA_DIR, filename)

    # get metadata from file
    file_info = file_str.split('_')  # silicon, pairs, XXdB
    attenuation = int(file_info[2][:2])
    power_on_chip = power_on_chip_ref * 10**(-attenuation/10)
    powers.append(power_on_chip)

    # read coincidence data
    df = pd.read_csv(file_path, sep='\t')
    dfs_coinc.append(df)
    time = df['Time(ps)'] * 1e-3  # convert to ns
    coincidences = df['Counts']

    # fitting
    print(f'Fitting {power_on_chip} uW pump')
    x0_guess = time[np.argmax(coincidences)]
    bg_guess = np.mean(coincidences[:100])
    amp_guess = np.max(coincidences)/bg_guess - 1
    tau_guess = 1.2
    res = model.fit(coincidences, x=time,
                    x0=x0_guess,
                    bg=bg_guess,
                    amplitude=amp_guess,
                    tau=tau_guess)
    all_res.append(res)

    tau = res.params['tau'].value
    tau_err = res.params['tau'].stderr
    bw = 1/(tau * 2 * np.pi)
    bw *= 1e3  # convert to MHz
    bw_err = (tau_err / tau) * bw
    g2 = res.params['amplitude'].value + 1
    g2_err = res.params['amplitude'].stderr
    print(f'\tBandwidth: {bw:.2f} +/- {bw_err:.2f} MHz')
    print(f'\tg2: {g2:.2f} +/- {g2_err:.2f}')

    # plot data for coincidences
    if PLOT_ALL_COINCIDENCE:
        fig, ax = plt.subplots(dpi=400)

        plt.bar(time, coincidences, color=color_coincidence,
                label='Data')
        plt.plot(time, res.best_fit,
                 color='k', ls='--',
                 label='Fit')

        plt.title(rf'Coincidences with {power_on_chip:.0f} $\mathrm{{\mu}}$W Pump Power On-Chip')
        # plt.title('Two-Photon Coincidence')
        plt.xlabel('Time (ns)')
        plt.ylabel('Coincidences')
        plt.legend()

        x_start = res.params['x0'].value - (x_range / 2)
        x_end = res.params['x0'].value + (x_range / 2)
        plt.xlim((x_start, x_end))

        plt.tight_layout()
        plt.show()


# gather all fit data
all_bg = [res.params['bg'].value for res in all_res]
all_bg_err = [res.params['bg'].stderr for res in all_res]
all_g2 = np.array([res.params['amplitude'].value + 1 for res in all_res])
all_g2_err = np.array([res.params['amplitude'].stderr for res in all_res])
all_integral = []
for bg, df in zip(all_bg, dfs_coinc):
    coincidences = df['Counts']
    integral_bg = bg * len(coincidences)
    integral = np.sum(coincidences) - integral_bg
    integral /= integration_time  # convert to pairs/s
    integral /= efficiency_pairs  # get on chip pair rate
    all_integral.append(integral)

# fit areas versus pump power
model = Model(quadratic)
res_area = model.fit(all_integral[2:], x=powers[2:], a=0.05)
print(res_area.fit_report())

# calculate pair rate
pair_rate_1mW = res_area.params['a'].value * 1e6  # convert from uW^2 to mW^2
print('Pair rate (s^-1 mW^-2):', pair_rate_1mW)


# plot of amplitudes versus pump power
powers_for_fit = np.linspace(min(powers), max(powers), 100)
plt.errorbar(powers, all_integral,
             color='cornflowerblue', ls='', marker='o', capsize=3,
             label='Fitted Data')
plt.plot(powers_for_fit, quadratic(powers_for_fit, **res_area.best_values),
         color='k', ls='--',
         label='Fitted Trend')

plt.title('Coincidences versus Pump Power')
plt.xlabel(r'Pump Power ($\mathrm{\mu}$W)')
plt.ylabel('Coincidence Counts above Background')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()

# plot of g(2)(0)
fig, ax = plt.subplots(dpi=400)
ax.errorbar(powers, all_g2, yerr=all_g2_err,
            ls='', marker='o', capsize=3, color='coral')
ax.set_xlabel(r'Pump Power ($\mathrm{\mu}$W)')
ax.set_ylabel(r'$g^{(2)}(0)$')

fig.tight_layout()
fig.show()

# plot of rate and bg
fig, ax = plt.subplots(dpi=400)
ax.errorbar(powers, all_integral,
            ls='', marker='o', capsize=3, color='cornflowerblue')
ax.errorbar(powers, all_bg, yerr=all_bg_err,
            ls='', marker='o', capsize=3, color='coral')
ax.set_xlabel(r'Pump Power ($\mathrm{\mu}$W)')
ax.set_ylabel(r'Rate ($\mathrm{s}^{-1}$)')
ax.set_xscale('log')
ax.set_yscale('log')

fig.tight_layout()
fig.show()
