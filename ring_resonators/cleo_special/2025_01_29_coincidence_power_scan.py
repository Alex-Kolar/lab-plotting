import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model

from ring_resonators.cavity_fit.cavity_metrics import g_2_no_delta


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/coincidence/01292025_pair_gen_power_scan")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color = 'cornflowerblue'
color_coincidence = 'coral'
PLOT_ALL = False


# efficiencies
efficiency_laser_coupling = 0.5
efficiency_spectrometer = 0.5
efficiency_chip_coupling = 0.05413
total_efficiency = efficiency_laser_coupling * efficiency_spectrometer * efficiency_chip_coupling
efficiency_out = efficiency_chip_coupling ** 2
idler_eff = 0.4532 * 0.7274  # channel 52
signal_eff = 0.9260 * 0.4790  # channel 48
efficiency_out = efficiency_out * idler_eff * signal_eff
int_time = 5 * 60  # units: s


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


# gather data for coincidences
coincidence_files = sorted(glob.glob('pairs_*', root_dir=DATA_DIR))
dfs_coinc = []  # store all dataframes
powers = []  # store all pump powers
int_times = []  # store all integration times
all_res = []  # store all fit result objects

# set up model for fitting
model = Model(g_2_no_delta)

for filename in coincidence_files:
    file_str = os.path.splitext(filename)[0]
    file_path = os.path.join(DATA_DIR, filename)

    # get metadata from file
    file_info = file_str.split('_')  # 'pairs', XXmW, Xmin
    power = float(file_info[1][:2])
    int_time = float(file_info[2][:1])
    power *= total_efficiency
    powers.append(power)
    int_times.append(int_time)

    # read coincidence data
    df = pd.read_csv(file_path, sep='\t')
    dfs_coinc.append(df)
    time = df['Time(ps)'] * 1e-3  # convert to ns
    coincidences = df['Counts']

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

    # plot data for coincidences
    if PLOT_ALL:
        plt.bar(time, coincidences, color=color_coincidence)
        plt.plot(time, res.best_fit,
                 color='k', ls='--')

        plt.title(f'Coincidences with {power} mW Pump Power')
        plt.xlabel('Time (ns)')
        plt.ylabel('Coincidences')
        plt.xlim((-5, 5))

        plt.tight_layout()
        plt.show()


# gather all fit data
all_amps = [res.params['amplitude'].value for res in all_res]
all_amps_err = [res.params['amplitude'].stderr for res in all_res]
all_kappa = []
all_kappa_err = []
all_g = []
all_g_err = []
all_integral = []
for res in all_res:
    # extract kappa data
    kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    kappa /= 2*np.pi
    kappa *= 1e3  # unit: MHz
    kappa_err = res.params['kappa'].stderr
    kappa_err /= 2*np.pi
    kappa_err *= 1e3
    all_kappa.append(kappa)
    all_kappa_err.append(kappa_err)

    # extract g data
    g = res.params['g'].value  # unit: 2*pi*GHz
    g /= 2*np.pi
    g *= 1e3  # unit: MHz
    g_err = res.params['g'].stderr
    g_err /= 2*np.pi
    g_err *= 1e3
    all_g.append(g)
    all_g_err.append(g_err)

    # extract integral (numerically)
    only_coincidence = res.best_fit - res.params['amplitude'].value
    times = res.userkws['x']
    time_diff = times[1] - times[0]  # assume uniform difference
    integral = np.sum(only_coincidence[:-1])  # * time_diff
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
pairs = res_area.params['a'].value / efficiency_out
pair_rate_1mW = pairs / int_time
print('Pair rate at 1 mW (/s):', pair_rate_1mW)


# plot of amplitudes versus pump power
powers_for_fit = np.linspace(min(powers), max(powers), 100)
plt.errorbar(powers, all_amps, yerr=all_amps_err,
             color='cornflowerblue', ls='', marker='o', capsize=3,
             label='Fitted Data')
plt.plot(powers_for_fit, quadratic(powers_for_fit, **res.best_values),
         color='k', ls='--',
         label='Fitted Trend')

plt.title('Coincidences versus Pump Power')
plt.xlabel('Pump Power (mW)')
plt.ylabel('Coincidence Fit Amplitude')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# plot of widths versus pump power
plt.errorbar(powers, all_kappa, yerr=all_kappa_err,
             color='coral', ls='', marker='o', capsize=3,
             label='Fitted Data')

plt.title('Coincidence Width versus Pump Power')
plt.xlabel('Pump Power (mW)')
plt.ylabel(r'Coincidence Fit $\kappa$ (MHz)')
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

# add text
# label = rf'$a$ = {res_area.params['a'].value:.3f} $\pm$ {res_area.params['a'].stderr:.3f}'
# t = ax.text(0.95, 0.05, label,
#             horizontalalignment='right', verticalalignment='bottom')
# t.set_transform(ax.transAxes)

plt.title('Coincidence Area versus Pump Power')
plt.xlabel(r'Pump Power On-Chip (mW)')
plt.ylabel(r'Coincidence Area')
plt.legend(framealpha=1)

plt.tight_layout()
plt.show()
