import numpy as np
import csv
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Model
import matplotlib.pyplot as plt


# for data
TRANSITION = 2  # specifies transitions to plot decay for
SCANTIME = 14  # specifies number of scan periods covered

# for peak finding
PROMINENCE = 0.1
PROMINENCE_SCAN = 10

# for plotting
plt.rc('font', family='Calibri', size=20)
PLOT_BCKGRND = True

# for fitting
def decay_single(x, amp, tau, offset):
    return amp * np.exp(-x / tau) + offset

def decay_double(x, amp_fast, amp_slow, tau_fast, tau_slow, offset):
    return amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset


# load all files
transition_str = f"tr{TRANSITION}"
base_dir = f"{transition_str}/probe_{SCANTIME}/"
center_file = base_dir + f"{transition_str}_pr{SCANTIME}.csv"
up_file = base_dir + f"probe_center_shifted/{transition_str}_up_pr{SCANTIME}.csv"
down_file = base_dir + f"probe_center_shifted/{transition_str}_down_pr{SCANTIME}.csv"

all_df = [pd.read_csv(center_file, skiprows=[1]),
          pd.read_csv(up_file, skiprows=[1]),
          pd.read_csv(down_file, skiprows=[1])]

offres_center_file = "offres/probe_1/off.csv"
offres_up_file = "offres/probe_1/off_up.csv"
offres_down_file = "offres/probe_1/off_down.csv"
all_df_offres = [pd.read_csv(offres_center_file, skiprows=[1]),
                 pd.read_csv(offres_up_file, skiprows=[1]),
                 pd.read_csv(offres_down_file, skiprows=[1])]

# get data on background range
all_max_bckgrnd = []
all_min_bckgrnd = []
for df in all_df_offres:
    trig_max = df['CH3'].idxmax()
    trig_min = df['CH3'].idxmin()
    bckgrnd = df['CH1'][trig_max:trig_min]
    all_max_bckgrnd.append(bckgrnd.max())
    all_min_bckgrnd.append(bckgrnd.min())
# max_bckgrnd = max([df['CH1'].max() for df in all_df_offres])
# min_bckgrnd = min([df['CH1'].min() for df in all_df_offres])
max_bckgrnd = max(all_max_bckgrnd)
min_bckgrnd = min(all_min_bckgrnd)
print("max/min: {} {}".format(max_bckgrnd, min_bckgrnd))

# get info on timing
with open(center_file, 'r') as f:
    reader = csv.reader(f)
    _ = next(reader)  # skip first line
    row = next(reader)
    time_inc = float(row[-1])

# find peaks and times
all_peaks = []
all_times = []
for df in all_df:
    scan_first_peak = find_peaks(df['CH3'], prominence=PROMINENCE_SCAN)[0][0]
    peaks = find_peaks(df['CH1'], prominence=PROMINENCE)[0]
    peaks = peaks[peaks > scan_first_peak]
    times = (df['X'] - scan_first_peak) * time_inc  # unit: s
    # times = (peaks - scan_first_peak) * time_inc  # unit: s
    times *= 1e3  # unit: ms
    all_peaks.append(peaks)
    all_times.append(times)

# do fitting
results = []
labels = ["Center", "Up", "Down"]
for i, df in enumerate(all_df):
    model = Model(decay_single)
    # if switching to double exponential, uncomment below and change result printout
    # params = model.make_params(amp_fast=0.5, amp_slow=0.5, tau_fast=1, tau_slow=10, offset=0.5)
    params = model.make_params(amp=0.5, tau=1, offset=0.5)
    result = model.fit(df['CH1'][all_peaks[i]], params, x=all_times[i][all_peaks[i]])
    tau = result.params['tau'].value
    sigma = result.params['tau'].stderr
    print(f"{labels[i]}: tau = {tau:.2f} +/- {sigma:.2f} ms")
    results.append(result)


# plotting
if PLOT_BCKGRND:
    plt.fill_between(all_times[0][all_times[0] > 0], max_bckgrnd, min_bckgrnd, label="Background",
                     color='tab:gray', alpha=0.2)

colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', 'v']
markers_fit = [':', '--', '-.']
for i, df in enumerate(all_df):
    timing = all_times[i]
    # plt.plot(df['CH1'], color=color, alpha=0.5)
    plt.plot(timing[all_peaks[i]], df['CH1'][all_peaks[i]],
             markers[i], color=colors[i], label=labels[i])
    plt.plot(timing[all_peaks[i]], results[i].best_fit,
             markers_fit[i], color=colors[i], label=f"{labels[i]} Fit")
    # plt.plot(timing[timing > 0], df['CH1'][timing > 0], color=colors[i], alpha=0.2)

plt.title(f"Transition {TRANSITION} Hole Decay")
plt.xlabel("Time (ms)")
plt.ylabel("Transmission (A.U.)")
plt.legend(shadow=True)
plt.grid('on')

plt.tight_layout()
plt.show()
