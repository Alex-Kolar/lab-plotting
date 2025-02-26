import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/coincidence/01292025_pair_gen_power_scan")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_coincidence = 'coral'


# fitting function for coincidences
def g_2_no_delta(x, x0, amplitude, kappa, g):
    x = np.abs(x - x0)
    exp_term = g*np.sinh(g*x) + (kappa/2)*np.cosh(g*x)
    g_2 = 1 + (np.exp(-kappa * x) / (g ** 2)) * (np.abs(exp_term) ** 2)
    return amplitude * g_2


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
for res in all_res:
    kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    kappa /= 2*np.pi
    kappa *= 1e3  # unit: MHz
    kappa_err = res.params['kappa'].stderr
    kappa_err /= 2*np.pi
    kappa_err *= 1e3
    all_kappa.append(kappa)
    all_kappa_err.append(kappa_err)

# fit amplitudes versus pump power
model = Model(quadratic)
res = model.fit(all_amps, x=powers,
                a=0.16)
print(res.fit_report())


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
