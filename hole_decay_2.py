import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Parameters, Model
import matplotlib as mpl
import matplotlib.pyplot as plt


# for data
DATA_DIR = "/Users/alexkolar/Desktop/Projects/AFC/6Amp/hole"
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope

# for peak finding
PROMINENCE = 0.01
DISTANCE = 100  # TODO: better way to explicitly calculate this?
PROMINENCE_SCAN = 1

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
xlim_all_plots = (-1, 11)
PLOT_BG = True


# fit functions
def decay_double(x, amp_fast, amp_slow, tau_fast, tau_slow, offset):
    return amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset


def decay_double_log(x, amp_fast, amp_slow, tau_fast, tau_slow, offset):
    return np.exp(amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset)


# locate all files
csv_files = glob.glob('*/center.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/CH3.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

bg_path = DATA_DIR + "/bg_offres/CH1.CSV"
bg_path_freq = DATA_DIR + "/bg_offres/CH3.CSV"


# read wait times
t_wait = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    t_wait_str = path[-2]
    t_wait_str = t_wait_str[:-2]  # remove "ms"
    t_wait_str = t_wait_str.replace('p', '.')
    t_wait[i] = float(t_wait_str)

# sort
csv_paths = [path for _, path in sorted(zip(t_wait, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(t_wait, csv_paths_freq))]
t_wait.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
df_bg = pd.read_csv(bg_path, names=TEK_HEADER)
df_bg_freq = pd.read_csv(bg_path_freq, names=TEK_HEADER)

# read starting times, peaks, and single scan
all_peaks = []  # NOTE: this is the INDEX of the peak in the array
all_starts = []  # NOTE: this is also the INDEX of the first scan in the array
for df, df_freq in zip(dfs, dfs_freq):
    scan_peaks = find_peaks(df_freq["Volts"], prominence=PROMINENCE_SCAN)[0]
    scan_first_peak = scan_peaks[0]

    all_starts.append(scan_first_peak)
    peaks = find_peaks(df["Volts"],
                       prominence=PROMINENCE, distance=DISTANCE)[0]
    peaks = peaks[peaks > scan_first_peak]
    all_peaks.append(peaks)

start_bg = find_peaks(df_bg_freq["Volts"], prominence=PROMINENCE_SCAN)[0][0]

# get background
max_bg = max(df_bg["Volts"][start_bg:])
min_bg = min(df_bg["Volts"][start_bg:])

# accumulate all peaks
all_peaks_combine = []
all_times_combine = []
for i, df in enumerate(dfs):
    start_idx = all_starts[i]
    peak_heights = df["Volts"][all_peaks[i]]
    time = df["Seconds"]
    time += (t_wait[i]/1e3 - time[start_idx])  # add offset
    peak_times = time[all_peaks[i]]

    peak_heights = peak_heights.tolist()
    peak_times = peak_times.tolist()

    all_peaks_combine += peak_heights
    all_times_combine += peak_times

# fitting
model = Model(decay_double_log)
params = Parameters()
params.add('amp_fast', value=0.2, min=0)
params.add('amp_slow', value=0.2, min=0)
params.add('tau_fast', value=0.0005, min=0)
params.add('tau_slow', value=1, min=0)
params.add('offset', value=0)
result = model.fit(all_peaks_combine, params=params, x=all_times_combine)
print(result.fit_report())


# for looking at all scans
color = 'tab:blue'
for i, df in enumerate(dfs):
    start_idx = all_starts[i]
    time = df["Seconds"][start_idx:]
    transmission = df["Volts"][start_idx:]

    # time *= 1e3  # convert to ms
    time += (t_wait[i]/1e3 - time[start_idx])  # add offset

    if i == 0:
        plt.plot(time, transmission, label="Transmission", color=color)
    else:
        plt.plot(time, transmission, color=color)

if PLOT_BG:
    plt.fill_between(xlim_all_plots, max_bg, min_bg, label="Background",
                     color='tab:gray', alpha=0.2)

plt.xlim(xlim_all_plots)
plt.title("Hole Transmission Decay (6A B-Field)")
plt.xlabel("Time (s)")
plt.ylabel("Transmission (A.U.)")
plt.legend()
plt.grid('on')

plt.tight_layout()
plt.show()


# for looking at all peaks + fit
plt.semilogy(all_times_combine, all_peaks_combine,
             'o', label='Data')
plt.semilogy(all_times_combine, result.best_fit,
             'k--', label='Fit')

plt.xlim((-0.1, 0.6))
plt.title("Hole Transmission Decay (6A B-Field)")
plt.xlabel("Time (s)")
plt.ylabel("Transmission (A.U.)")
plt.legend()
plt.grid('on')

plt.tight_layout()
plt.show()


# for looking at individual scan + fit


# # for studying one round of peaks
# SCAN_TO_PLOT = 0
# # plt.plot(dfs[SCAN_TO_PLOT]["Seconds"], dfs[SCAN_TO_PLOT]["Volts"])
# plt.plot(dfs[SCAN_TO_PLOT]["Volts"])
# # plt.plot(dfs_freq[0]["Seconds"], dfs_freq[0]["Volts"])
# plt.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_peaks[SCAN_TO_PLOT]],
#          dfs[SCAN_TO_PLOT]["Volts"][all_peaks[SCAN_TO_PLOT]],
#          'x')
# print([len(a) for a in all_peaks])
# print(all_peaks[SCAN_TO_PLOT])


# # for looking at detected peaks
# SCAN_TO_PLOT = [0, 1, 2]
# colors = ['tab:blue', 'tab:orange', 'tab:green']
# for i, scan in enumerate(SCAN_TO_PLOT):
#     plt.plot(dfs[scan]["Seconds"][all_starts[i]:], dfs[scan]["Volts"][all_starts[i]:],
#              color=colors[i])
#     plt.plot(dfs[scan]["Seconds"][all_peaks[scan]],
#              dfs[scan]["Volts"][all_peaks[scan]],
#              'x', color=colors[i])
#
# plt.show()
