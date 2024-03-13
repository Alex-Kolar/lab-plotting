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
            "/02_13_24/burnnprobe/16amp_Bfield/changing_N_wait")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 250  # Unit: MHz
SCAN_TIME = 0.0064  # Unit: s
GAIN = 1e8  # Unit: V/W
WAIT_TIME = 0.128  # Unit: ms (total wait time = N_wait * wait_time)

EDGE_THRESH = 1

# plotting params
CMAP_OFFSET = 0.3
CMAP = cm.Blues
max_low_plot = 0.2  # for low amplitude pumps
ylim = (0, 3)
PLOT_OD = True  # plot as optical depth
LOG_CMAP = False  # use log scale for colormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


"""
FILE PROCESSING
"""

print("Gathering files...")

# if ZOOMIN:
#     DATA_DIR = os.path.join(DATA_DIR, "zoomin")
# else:
#     DATA_DIR = os.path.join(DATA_DIR, "zoomout")

# locate all files
csv_files = glob.glob('*/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

print(f"Found {len(csv_paths)} data files.")
print(f"Found {len(csv_paths_freq)} frequency files.")
if len(csv_paths) == 0:
    raise Exception("No valid files found.")
if len(csv_paths) != len(csv_paths_freq):
    raise Exception("Mismatch in number of frequency and transmission channels.")

# read timing
wait_times = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    wait_str = path[-2]
    wait_str = wait_str.replace('p', '.')
    wait_times[i] = float(wait_str) + 8  # constant offset from experiment
wait_times *= (WAIT_TIME / 1e3)  # Unit: s

if LOG_CMAP:
    zero_idx = np.where(wait_times == 0)[0]
    wait_times = np.delete(wait_times, zero_idx)
    csv_paths = np.delete(csv_paths, zero_idx)
    csv_paths_freq = np.delete(csv_paths_freq, zero_idx)

    wait_times = np.log10(wait_times)

cmap_vmin = min(wait_times)
cmap_vmax = max(wait_times)

# sort
csv_paths = [path for _, path in sorted(zip(wait_times, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(wait_times, csv_paths_freq))]
wait_times.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]


"""
DATA PROCESSING
"""

print("Gathering transmission peaks and background...")

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = []
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []
max_trans = 0
for df, df_freq in zip(dfs, dfs_freq):
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]
    if len(scan_edge) > 1:
        warnings.warn("Multiple scan edges found, defaulting to first.")
    center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx)

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5*SCAN_TIME, 6)
    stop_time = np.round(center_time + 0.5 * SCAN_TIME, 6)

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    transmission = df["Volts"][start_idx:stop_idx]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq)

    max_trans = max(max_trans, max(transmission))

for trans in all_scan_transmission:
    trans_arr = np.array(trans)
    all_scan_od.append(np.log(max_trans / trans_arr))


"""
PLOTTING
"""


lines_low = []
amps_low = []
lines_high = []
amps_high = []
if PLOT_OD:
    plot_lines = all_scan_od
else:
    plot_lines = all_scan_transmission

for freq, trans, amp in zip(all_scan_freq, plot_lines, wait_times):
    line = np.column_stack((freq, trans))
    if amp <= max_low_plot:
        lines_low.append(line)
        amps_low.append(amp)
    else:
        lines_high.append(line)
        amps_high.append(amp)

cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
line_coll_low = LineCollection(lines_low, cmap=cmap)
line_coll_low.set_array(amps_low)
line_coll_low.set_clim(cmap_vmin, cmap_vmax)
line_coll_high = LineCollection(lines_high, cmap=cmap)
line_coll_high.set_array(amps_high)
line_coll_high.set_clim(cmap_vmin, cmap_vmax)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
im1 = ax1.add_collection(line_coll_low, autolim=True)
ax1.set_xlim((-SCAN_RANGE/2, SCAN_RANGE/2))
ax1.set_ylim(ylim)
im2 = ax2.add_collection(line_coll_high, autolim=True)
ax2.set_xlim((-SCAN_RANGE/2, SCAN_RANGE/2))
ax2.set_ylim(ylim)

ax2.tick_params(axis='y', which='both', left=False, labelleft=False)
ax1.grid(True)
ax2.grid(True)

# labeling
ax1.set_xlabel("Detuning (MHz)")
ax2.set_xlabel("Detuning (MHz)")
if PLOT_OD:
    ax1.set_ylabel("Optical Depth")
else:
    ax1.set_ylabel("Transmission (nW)")
if LOG_CMAP:
    ax1.set_title(rf"Wait Time Change (Log(T_wait) $\leq$ {max_low_plot})")
    ax2.set_title(rf"Wait Time Change (Log(T_wait) > {max_low_plot})")
else:
    ax1.set_title(rf"Wait Time Change (T_wait $\leq$ {max_low_plot})")
    ax2.set_title(rf"Wait Time Change (T_wait > {max_low_plot})")

plt.tight_layout()

# add colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(im1, cax=cbar_ax)
if LOG_CMAP:
    cb.set_label("Log Wait Duration T_wait (s)")
else:
    cb.set_label("Wait Duration T_wait (s)")
# axcb = fig.colorbar(line_coll_low, ax=ax1)
# axcb.set_label("Pump Amplitude")
# axcb = fig.colorbar(line_coll_high, ax=ax2)
# axcb.set_label("Pump Amplitude")

plt.show()
