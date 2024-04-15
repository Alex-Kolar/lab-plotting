import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
import warnings


# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
            "/04_13_24/probe_wo_hdawg/nopump/changing_probe_power")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 50  # Unit: MHz
SCAN_RATE = 1  # Units: Hz (options 1 or 10)
# data for changing gain settings
# units: V/W
GAIN_RES = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e8, 1e8, 1e8]
GAIN_OFF = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8]

# plotting params
CMAP_OFFSET = 0.3
CMAP = cm.Blues
max_low_plot = 0.5  # for low amplitude pumps
xlim = (-25, 25)
ylim = (0, 6)

PLOT_OD = True  # plot as optical depth
LOG_CMAP = True  # use log scale for colormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


"""
FILE PROCESSING
"""

print("Gathering files...")

# locate all files
csv_files = glob.glob(f'*/{SCAN_RATE}Hz/res/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob(f'*/{SCAN_RATE}Hz/res/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]
# background files
csv_files_bg = glob.glob(f'*/{SCAN_RATE}Hz/offres/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq_bg = glob.glob(f'*/{SCAN_RATE}Hz/offres/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths_bg = [os.path.join(DATA_DIR, file) for file in csv_files_bg]
csv_paths_freq_bg = [os.path.join(DATA_DIR, file) for file in csv_files_freq_bg]

# read probe powers
probe_pows = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    probe_str = path[-4]
    probe_str = probe_str[:-2]  # remove 'nW'
    probe_str = probe_str.replace('p', '.')
    probe_pows[i] = float(probe_str)
print(probe_pows)

# convert to log scale if necessary
if LOG_CMAP:
    zero_idx = np.where(probe_pows == 0)[0]
    probe_pows = np.delete(probe_pows, zero_idx)
    csv_paths = np.delete(csv_paths, zero_idx)
    csv_paths_freq = np.delete(csv_paths_freq, zero_idx)
    csv_paths_bg = np.delete(csv_paths_bg, zero_idx)
    csv_paths_freq_bg = np.delete(csv_paths_freq_bg, zero_idx)

    probe_pows = np.log10(probe_pows)

cmap_vmin = min(probe_pows)
cmap_vmax = max(probe_pows)

# sort
csv_paths = [path for _, path in sorted(zip(probe_pows, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(probe_pows, csv_paths_freq))]
csv_paths_bg = [path for _, path in sorted(zip(probe_pows, csv_paths_bg))]
csv_paths_freq_bg = [path for _, path in sorted(zip(probe_pows, csv_paths_freq_bg))]
probe_pows.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
dfs_bg = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_bg]
dfs_freq_bg = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq_bg]
print(f"Found {len(dfs)} data files.")
print(f"Found {len(dfs_freq)} frequency files.")
print(f"Found {len(dfs_bg)} background data files.")
print(f"Found {len(dfs_freq_bg)} background frequency files.")


"""
DATA PROCESSING
"""

scan_time = (1 / SCAN_RATE) / 2  # unit: s

print("Gathering background...")

all_bg_trans = []
all_bg_freq = []
for i, (df, df_freq) in enumerate(zip(dfs_bg, dfs_freq_bg)):
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -1]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > 1]
    if len(scan_edge) > 1:
        warnings.warn("Multiple scan edges found for background, defaulting to first.")
    center_idx = scan_edge[0]

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5 * scan_time, 6)
    stop_time = np.round(center_time + 0.5 * scan_time, 6)

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]

    transmission = df["Volts"][start_idx:stop_idx]
    gain = GAIN_OFF[i]
    transmission = (transmission / gain) * 1e9  # convert to nW
    all_bg_trans.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_bg_freq.append(freq)

print("Gathering transmission peaks and background...")

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = []
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []
max_trans = 0
for i, (df, df_freq) in enumerate(zip(dfs, dfs_freq)):
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -1]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > 1]
    assert len(scan_edge) == 1, "Problem finding single falling edge for frequency channel."
    center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx)

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5 * scan_time, 6)
    stop_time = np.round(center_time + 0.5 * scan_time, 6)

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    transmission = df["Volts"][start_idx:stop_idx]
    gain = GAIN_RES[i]
    transmission = (transmission / gain) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq)

for trans, bg in zip(all_scan_transmission, all_bg_trans):
    trans_arr = np.array(trans)
    all_scan_od.append(np.log(bg / trans_arr))


"""
PLOTTING
"""


lines = []
if PLOT_OD:
    plot_lines = all_scan_od
else:
    plot_lines = all_scan_transmission

for freq, trans in zip(all_scan_freq, plot_lines):
    line = np.column_stack((freq, trans))
    lines.append(line)

cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
line_coll = LineCollection(lines, cmap=cmap)
line_coll.set_array(probe_pows)
line_coll.set_clim(cmap_vmin, cmap_vmax)

fig, ax1 = plt.subplots(1, 1)
im1 = ax1.add_collection(line_coll, autolim=True)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.grid(True)

# labeling
ax1.set_xlabel("Detuning (MHz)")
if PLOT_OD:
    ax1.set_ylabel("Optical Depth")
else:
    ax1.set_ylabel("Transmission (nW)")
ax1.set_title(rf"Probe Power Change")

plt.tight_layout()

# add colorbar
# axcb = fig.colorbar(line_coll_low, ax=ax1)
# axcb.set_label("Pump Amplitude")
# axcb = fig.colorbar(line_coll_high, ax=ax2)
# axcb.set_label("Pump Amplitude")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cb = fig.colorbar(im1, cax=cbar_ax)
if LOG_CMAP:
    cb.set_label("Log (base 10) Probe Power (nW)")
else:
    cb.set_label("Probe Power (nW)")

plt.show()
