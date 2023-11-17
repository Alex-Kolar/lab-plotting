import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators/07072023_power dependent"
OUTPUT_FILENAME = "output_figs/ring_power_scan.png"

# data taken for frequency
F_MIN = 193395.328
F_MAX = 193426.015

# range for fitting
FIT_RANGE = 3  # unit: GHz

# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
x_range = (15.7, 19)  # unit: GHz
y_range = (0, 0.175)


# locate all files
csv_files = glob.glob('*_*MW.csv', root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]

# sort all files and get powers
powers = np.zeros(len(csv_files))
for i, file in enumerate(csv_files):
    power_str = file[:5]
    power_str = power_str.replace("_", ".")
    power = float(power_str) / 9
    powers[i] = power
csv_paths = sorted(csv_paths)
powers = sorted(powers)

dfs = [pd.read_csv(path, header=11) for path in csv_paths]

# get frequency scan info
df = dfs[0]
scan = df['Volt']
id_min = scan.idxmin()
id_max = scan.idxmax()


# plotting
fig, ax = plt.subplots()

cmap = mpl.cm.get_cmap('Blues')
for power, df in zip(powers, dfs):
    scan = df['Volt']
    id_min = scan.idxmin()
    id_max = scan.idxmax()
    trans = df['Volt.1'][id_min:id_max]
    trans = trans.reset_index(drop=True)

    # convert to frequency
    freq_start = 0
    freq_stop = F_MAX - F_MIN
    freq = np.linspace(freq_start, freq_stop, num=(id_max-id_min))

    # get range for fitting
    min_idx = trans.idxmin()
    center_freq = freq[min_idx]
    lower_freq = center_freq - (FIT_RANGE / 2)
    higher_freq = center_freq + (FIT_RANGE / 2)
    lower_idx = np.abs(freq - lower_freq).argmin()
    higher_idx = np.abs(freq - higher_freq).argmin()

    freq_fit = freq[lower_idx:higher_idx]
    trans_fit = trans[lower_idx:higher_idx]

    # fit
    model = LinearModel() + BreitWignerModel()
    out = model.fit(trans_fit, x=freq_fit,
                    center=center_freq)

    # get fit info
    center = out.params['center'].value
    width = out.params['sigma'].value
    Q = (center + F_MIN) / width
    print("Q for {:0.0f} uW:".format(power * 1e3), Q)

    min_trans = trans.min()
    min_freq = freq[trans.idxmin()]

    color = cmap(power / max(powers))
    ax.plot(freq, trans,
            color=color)
    ax.axvline(x=min_freq, color='k')
    ax.text(min_freq-0.05, min_trans-0.015, r'{:0.0f} $\mu$W'.format(power * 1e3),
            bbox={'facecolor': 'white', 'pad': 3}, fontsize=10)

ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title("Cavity Resonance Shift")
ax.set_xlabel("Detuning from {:0.0f} GHz (GHz)".format(F_MIN))
ax.set_ylabel("Transmission (A.U.)")

fig.tight_layout()
plt.savefig(OUTPUT_FILENAME)


# ## testing of fitting
# df = dfs[0]
#
# # get data for frequency scan
# scan = df['Volt']
# id_min = scan.idxmin()
# id_max = scan.idxmax()
# trans = df['Volt.1'][id_min:id_max]
# trans = trans.reset_index(drop=True)
#
# # convert to frequency
# freq_start = 0
# freq_stop = F_MAX - F_MIN
# freq = np.linspace(freq_start, freq_stop, num=(id_max-id_min))
#
# # get range for fitting
# min_idx = trans.idxmin()
# center_freq = freq[min_idx]
# lower_freq = center_freq - (FIT_RANGE / 2)
# higher_freq = center_freq + (FIT_RANGE / 2)
# lower_idx = np.abs(freq - lower_freq).argmin()
# higher_idx = np.abs(freq - higher_freq).argmin()
#
# freq_fit = freq[lower_idx:higher_idx]
# trans_fit = trans[lower_idx:higher_idx]
#
# # fit
# model = LinearModel() + BreitWignerModel()
# out = model.fit(trans_fit, x=freq_fit,
#                 center=center_freq)
# print(out.fit_report())
#
# plt.clf()
# plt.plot(freq, trans)
# plt.plot(freq_fit, out.best_fit, 'k--')
#
# plt.xlim(x_range)
# plt.show()
