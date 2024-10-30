import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


# for data handling
DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Original_device/Coincidence Count Measurement/08102023"

# measured experimental parameters
MEAS_POWER = np.array([28.63, 56.49, 86.20, 95.05])  # unit: uW
BS_RATIO = 99  # for measured power with power meter
ETA_FACET = 0.178  # coupling efficiency for one facet

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_signal = 'cornflowerblue'
color_idler = 'coral'


# get data for singles
power_in = (MEAS_POWER * BS_RATIO) * 1e-3  # unit: mW
on_chip_power = power_in * ETA_FACET  # unit: mW

counts_on = sorted(glob.glob(os.path.join(DATA_DIR, "countrate_on_res*")))
counts_off = sorted(glob.glob(os.path.join(DATA_DIR, "countrate_off_res*")))
assert len(counts_on) == len(counts_off) == len(MEAS_POWER)

df_counts_on = [pd.read_csv(name, sep='\t') for name in counts_on]
df_counts_off = [pd.read_csv(name, sep='\t') for name in counts_off]

signal_counts_on_avg = np.array([np.mean(df['counts']) for df in df_counts_on])
signal_counts_off_avg = np.array([np.mean(df['counts']) for df in df_counts_off])
idler_counts_on_avg = np.array([np.mean(df['counts.1']) for df in df_counts_on])
idler_counts_off_avg = np.array([np.mean(df['counts.1']) for df in df_counts_off])


# plotting of singles data
plt.plot(on_chip_power, signal_counts_off_avg,
         'o--', color=color_signal, label="Signal (off-resonance)")
plt.plot(on_chip_power, signal_counts_on_avg,
         'o-', color=color_signal, label="Signal (on-resonance)")
plt.plot(on_chip_power, idler_counts_off_avg,
         's--', color=color_idler, label="Idler (off-resonance)")
plt.plot(on_chip_power, idler_counts_on_avg,
         's-', color=color_idler, label="Idler (on-resonance)")

plt.title("Singles Counts")
plt.xlabel("On-Chip Power (mW)")
plt.ylabel("Counts/s")
plt.grid('on')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# get data for coincidences
coinc_on = sorted(glob.glob(os.path.join(DATA_DIR, "correlation_on_res*")))
coinc_off = sorted(glob.glob(os.path.join(DATA_DIR, "correlation_off_res*")))
assert len(coinc_on) == len(coinc_off) == len(MEAS_POWER)

df_coinc_on = [pd.read_csv(name, sep='\t') for name in coinc_on]
df_coinc_off = [pd.read_csv(name, sep='\t') for name in coinc_off]

coinc_real = np.array([np.max(df["Counts"]) for df in df_coinc_on])
coinc_accident = np.zeros(coinc_real.size)
for i, df in enumerate(df_coinc_on):
    idx_to_keep = np.abs(df["Time(ps)"]) > 10e3  # discard times within 10 ns
    coinc_accident[i] = np.mean(df["Counts"][idx_to_keep])
CAR = (coinc_real - coinc_accident) / coinc_accident
coinc_real = coinc_real / 60  # convert to counts/sec (integration time 1 min.)

time = df_coinc_on[0]["Time(ps)"]
time_diff = time[1] - time[0]  # spacing of histogram
time_diff *= 1e-3  # convert to ns


# plotting of coincidence data
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.plot(on_chip_power, coinc_real,
        'o-', color=color_signal, label="Coincidences")
ax2.plot(on_chip_power, CAR,
         's-', color=color_idler, label="CAR")

ax.set_title("Coincidence Data ({} ns Window)".format(time_diff))
ax.set_xlabel("On-Chip Power (mW)")
ax.set_ylabel("Coincidences (Counts/s)")
ax2.set_ylabel("CAR")
ax.grid('on')
ax.legend(shadow=True)
ax2.legend(shadow=True)

fig.tight_layout()
plt.show()


# determine efficiency
signal_eff = coinc_real / (signal_counts_on_avg - signal_counts_off_avg)
idler_eff = coinc_real / (idler_counts_on_avg - idler_counts_off_avg)
# convert to dB
# signal_eff = 10*np.log10(signal_eff)
# idler_eff = 10*np.log10(idler_eff)
plt.plot(on_chip_power, signal_eff,
         label='Signal')
plt.plot(on_chip_power, idler_eff,
         label='Idler')

plt.legend()
plt.xlabel("On-Chip Power (mW)")
plt.ylabel("Ratio of coincidences to singles")

plt.tight_layout()
plt.show()
