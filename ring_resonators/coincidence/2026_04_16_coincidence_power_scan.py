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

# power parameters
power_99 = 5320  # power measured at 99% arm of beamsplitter (in uW)
power_1 = 54.3  # power measured at 1% arm of beamsplitter (in uW)
power_ratio = power_99 / power_1

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
PLOT_ALL_FILES = True
coincidence_xlim = (0, 50)
color_coincidence = 'cornflowerblue'
color_car = 'coral'


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


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

    # plotting for each file
    if PLOT_ALL_FILES:
        time_series = data_df['Time(ps)'] * 1e-3  # convert to ns
        plt.plot(time_series, data_df['Counts'])
        plt.xlabel('Time (ns)')
        plt.ylabel('Counts')
        plt.xlim(coincidence_xlim)
        plt.title(f'File {file_num}')
        plt.show()

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

# fit coincidences
model = Model(quadratic)
res = model.fit(coincidences, x=on_chip_power,
                a=0.5)
print(res.fit_report())

# final plotting
error = np.sqrt(coincidences)
powers_for_fit = np.linspace(min(on_chip_power), max(on_chip_power), 100)

fig, ax = plt.subplots()
ax_r = ax.twinx()
ax.errorbar(on_chip_power/1e3, coincidences, yerr=error,
            capsize=3, marker='o', linestyle='', color=color_coincidence)
ax.plot(powers_for_fit/1e3, res.eval(x=powers_for_fit),
        color=color_coincidence, ls='--')
ax_r.plot(on_chip_power/1e3, car,
          color=color_car, marker='o', ls='')
ax.set_xlabel(r'On-Chip Power (mW)')
ax.set_ylabel('Coincidence Counts', color=color_coincidence)
ax_r.set_ylabel('CAR', color=color_car)

fig.tight_layout()
fig.show()
