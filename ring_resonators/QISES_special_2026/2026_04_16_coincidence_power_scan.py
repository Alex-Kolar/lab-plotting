import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Unmounted_device_mk_3/2026_04_15/pair_generation')
POWER_FILE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
              '/Unmounted_device_mk_3/2026_04_15/pair_generation/correlation_power.csv')
file_fmt = 'correlation_{:02}.txt'
file_num_center_bin = 1
num_bins = 10  # number of bins to use for coincidence calculation
num_bins_exclude_bg = 200
integration_time = 5 * 60  # seconds

# power parameters
power_99 = 5320  # power measured at 99% arm of beamsplitter (in uW)
power_1 = 54.3  # power measured at 1% arm of beamsplitter (in uW)
power_ratio = power_99 / power_1

# efficiency parameters
snspd_52_efficiency = 0.4532 * 0.7274   # includes filter loss and detector inefficiency
snspd_48_efficiency = 0.4790 * 0.9260

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
PLOT_AND_FIT_FILE = True
fit_file = 1
coincidence_xlim = (0, 50)
color_coincidence = 'cornflowerblue'
color_car = 'coral'


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


# coincidence fitting function
def g_2_exp_bg(x, x0, bg, amplitude, T_1, T_2):
    """Another fitting function for cavity decay.

    Fit as two exponentials with a constant background offset.

    Args:
        x (np.ndarray): time-domain x-values
        x0 (float): center of coincidence peak
        bg (float): value of accidental coincidences
        amplitude (float): amplitude of coincidence peak
        T_1 (float): decay constant for right (positive) side of peak
        T_2 (float): decay constant for left (negative) side of peak

    Return:
        np.ndarray: coincidence histogram
    """
    right_decay = np.heaviside(x-x0, 0.5) * np.exp(-(x-x0) / T_1)
    left_decay = np.heaviside(x0-x, 0.5) * np.exp(-(x0-x) / T_2)
    bg_arr = bg * np.ones_like(x)
    return (amplitude * right_decay) + (amplitude * left_decay) + bg_arr


# read power file
power_df = pd.read_csv(POWER_FILE)
file_numbers = power_df['File']

# get on-chip power
measured_bs_power = power_df['Beamsplitter Power (uW)']
input_power = measured_bs_power * power_ratio
output_power = power_df['Output Power (uW)']
total_efficiency = output_power / input_power
on_chip_power = np.sqrt(total_efficiency) * input_power
print(on_chip_power)

# loop over files to extract coincidence power
data_dfs = []
center_bin = 0
for file_num in file_numbers:
    file_path = os.path.join(DATA_DIR, file_fmt.format(file_num))
    data_df = pd.read_csv(file_path, sep='\t')
    data_dfs.append(data_df)

    if file_num == file_num_center_bin:
        center_bin = np.argmax(data_df['Counts'])
        print(f'Center bin: {center_bin}')
        print(f'Center time: {data_df["Time(ps)"][center_bin] * 1e-3:.2f} ns')

    if file_num == fit_file and PLOT_AND_FIT_FILE:
        # fitting
        coincidence_model = Model(g_2_exp_bg)
        coincidence = data_df['Counts'] / integration_time  # convert to counts/s
        time_coincidence = data_df['Time(ps)'] * 1e-3  # convert to ns
        x0_guess = 25
        bg_guess = 1
        amplitude_guess = 10
        T_1_guess = 1
        T_2_guess = 1
        res_coincidence = coincidence_model.fit(coincidence, x=time_coincidence,
                                                x0=x0_guess,
                                                bg=bg_guess,
                                                amplitude=amplitude_guess,
                                                T_1=T_1_guess,
                                                T_2=T_2_guess)
        print(res_coincidence.fit_report())

        fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
        ax.plot(time_coincidence, coincidence, color=color_coincidence)
        ax.plot(time_coincidence, res_coincidence.best_fit,
                 color='k', ls='--')
        ax.set_xlim(coincidence_xlim)
        ax.set_xlabel('Timing Offset (ns)')
        ax.set_ylabel('Counts')
        ax.set_title('Coincidence Histogram')
        fig.tight_layout()
        fig.show()

# extract coincidences and CAR
idx_start = center_bin - num_bins // 2
idx_end = center_bin + num_bins // 2 + 1
idx_bg_exclude_start = center_bin - num_bins_exclude_bg // 2
idx_bg_exclude_end = center_bin + num_bins_exclude_bg // 2 + 1
coincidences = []
car = []
for data_df in data_dfs:
    counts = np.sum(data_df['Counts'][idx_start:idx_end])
    coincidences.append(counts)

    num_bg_bins = len(data_df['Counts']) - num_bins_exclude_bg
    bg = np.sum(data_df['Counts'][:idx_bg_exclude_start]) + np.sum(data_df['Counts'][idx_bg_exclude_end:])
    bg /= num_bg_bins
    bg *= num_bins
    # print(f'Background: {bg:.2f}')
    car.append((counts - bg) / bg)

# fit coincidence rate
model = Model(quadratic)
coincidence_rate = np.array(coincidences) / integration_time
res = model.fit(coincidence_rate, x=on_chip_power,
                a=0.5)
print(res.fit_report())

# fit coincidence rate on-chip
coincidence_rate_on_chip = coincidence_rate / (total_efficiency * snspd_48_efficiency * snspd_52_efficiency)
res_on_chip = model.fit(coincidence_rate_on_chip, x=on_chip_power/1e3, # convert to mW
                        a=3e5)
print(res_on_chip.fit_report())

# final plotting
error = np.sqrt(coincidences) / integration_time
powers_for_fit = np.linspace(min(on_chip_power), max(on_chip_power), 100)

fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
ax_r = ax.twinx()
ax.errorbar(on_chip_power/1e3, coincidence_rate, yerr=error,
            capsize=3, marker='o', linestyle='', color=color_coincidence)
ax.plot(powers_for_fit/1e3, res.eval(x=powers_for_fit),
        color=color_coincidence, ls='--')
ax_r.plot(on_chip_power/1e3, car,
          color=color_car, marker='s', ls='')
ax.set_title('Pair Generation Rate and CAR')
ax.set_xlabel(r'On-Chip Power (mW)')
ax.set_ylabel('Coincidence Counts (s$^{-1}$)', color=color_coincidence)
ax_r.spines['left'].set_color(color_coincidence)
ax.tick_params(axis='y', colors=color_coincidence)
ax_r.set_ylabel('CAR', color=color_car)
ax_r.spines['right'].set_color(color_car)
ax_r.tick_params(axis='y', colors=color_car)

fig.tight_layout()
fig.show()
